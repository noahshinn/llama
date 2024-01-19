# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    probs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        torch.cuda.set_device("cuda:0")

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()
        checkpoint = torch.load(os.path.join(ckpt_dir, "consolidated.00.pth"), map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int = 1,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)

        all_probs = []
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            probs = torch.softmax(logits[:, -1] / 0.6, dim=-1)
            all_probs.append(probs.squeeze().cpu().tolist())
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)
        return (out_tokens, all_probs)

    def chat_completion(
        self,
        dialogs: List[Dialog],
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        """
        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_probs = self.generate(
            prompt_tokens=prompt_tokens,
        )
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
                "tokens": [self.tokenizer.decode(x) for x in t],
                "probs": probs_i,
            }
            for t, probs_i in zip(
                generation_tokens, generation_probs
            )
        ]
