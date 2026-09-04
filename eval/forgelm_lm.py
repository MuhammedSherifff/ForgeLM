"""Official lm-evaluation-harness adapter for a ForgeLM checkpoint.

The harness evaluates causal language models through token log-likelihood.
This wrapper deliberately keeps that bridge visible: it tokenizes each
context/continuation pair, runs ForgeLM, selects the continuation logits, and
returns ``(log_likelihood, is_greedy)`` as required by the harness.

Import this module only from the official-evaluation entry point because
``lm_eval`` is an optional dependency of ForgeLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from config import load_config
from training.common import build_model_and_tokenizer, load_model_state


def _load_harness_lm_base():
    try:
        from lm_eval.api.model import LM
    except ImportError as error:
        raise ImportError(
            "Official evaluation requires lm-eval. Install it with "
            "pip install 'lm-eval[hf]==0.4.8' 'transformers>=4.45,<5'"
        ) from error
    return LM


LMBase = _load_harness_lm_base()


class ForgeLMLM(LMBase):
    """Expose ForgeLM through the official harness model interface."""

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_gen_toks: int = 128,
    ) -> None:
        super().__init__()
        experiment = load_config(config)
        self._device = torch.device(device)
        self.model, self.tokenizer, self.model_config = build_model_and_tokenizer(
            experiment, self._device
        )
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)

        if self._batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._max_gen_toks < 0:
            raise ValueError("max_gen_toks cannot be negative")

        saved = torch.load(checkpoint, map_location=self._device, weights_only=False)
        if "model" not in saved:
            raise ValueError(f"checkpoint has no model state: {checkpoint}")
        load_model_state(self.model, saved["model"])
        self.model.eval()

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self.model_config.block_size

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def eot_token_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("ForgeLM tokenizer must define eos_token_id")
        return token_id

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize a pair while preserving the harness word-boundary rule."""
        if not context:
            context_ids = [self.eot_token_id]
            continuation_ids = self.tok_encode(continuation, add_special_tokens=False)
        else:
            trailing_spaces = len(context) - len(context.rstrip())
            if trailing_spaces:
                continuation = context[-trailing_spaces:] + continuation
                context = context[:-trailing_spaces]
            whole = self.tok_encode(context + continuation, add_special_tokens=False)
            context_ids = self.tok_encode(context, add_special_tokens=False)
            continuation_ids = whole[len(context_ids):]
        if not continuation_ids:
            raise ValueError("continuation produced no tokens")
        return context_ids, continuation_ids

    @torch.inference_mode()
    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        encoded: list[tuple[list[int], list[int]]] = []
        for context, continuation in pairs:
            context_ids, continuation_ids = self._encode_pair(context, continuation)
            available = self.max_length - len(continuation_ids)
            if available < 1:
                raise ValueError("continuation is too long for ForgeLM context length")
            context_ids = context_ids[-available:]
            encoded.append((context_ids, continuation_ids))

        results: list[tuple[float, bool]] = []
        for offset in range(0, len(encoded), self.batch_size):
            batch = encoded[offset:offset + self.batch_size]
            lengths = [len(context) + len(cont) for context, cont in batch]
            width = max(lengths)
            input_ids = torch.full(
                (len(batch), width), self.eot_token_id,
                dtype=torch.long, device=self.device,
            )
            for row, (context_ids, continuation_ids) in enumerate(batch):
                tokens = context_ids + continuation_ids
                input_ids[row, :len(tokens)] = torch.tensor(tokens, device=self.device)

            logits = self.model(input_ids)
            for row, (context_ids, continuation_ids) in enumerate(batch):
                start = len(context_ids) - 1
                end = start + len(continuation_ids)
                selected = logits[row, start:end].float()
                targets = torch.tensor(continuation_ids, device=self.device)
                log_probs = F.log_softmax(selected, dim=-1)
                token_log_probs = log_probs.gather(-1, targets[:, None]).squeeze(-1)
                greedy = bool((selected.argmax(dim=-1) == targets).all().item())
                results.append((float(token_log_probs.sum().item()), greedy))
        return results

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        return self._score_pairs([request.args for request in requests])

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        results = []
        for request in requests:
            tokens = [self.eot_token_id] + self.tok_encode(
                request.args[0], add_special_tokens=False
            )
            total = 0.0
            # Each window scores its target tokens once. Consecutive windows
            # overlap by one token so the first target has a valid predecessor.
            target_start = 1
            while target_start < len(tokens):
                target_end = min(target_start + self.max_length - 1, len(tokens))
                input_start = target_start - 1
                window = tokens[input_start:target_end]
                input_tensor = torch.tensor([window], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor)[0, :-1].float()
                targets = torch.tensor(window[1:], dtype=torch.long, device=self.device)
                total += float(
                    F.log_softmax(logits, dim=-1)
                    .gather(-1, targets[:, None]).sum().item()
                )
                target_start = target_end
        results.append(total)
        return results

    @torch.inference_mode()
    def generate_until(self, requests: list[Any]) -> list[str]:
        outputs = []
        for request in requests:
            context, gen_kwargs = request.args
            context_ids = self.tok_encode(context, add_special_tokens=False)
            if not context_ids:
                context_ids = [self.eot_token_id]
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            generated = self.model.generate(
                torch.tensor([context_ids], dtype=torch.long, device=self.device),
                max_new_tokens=int(gen_kwargs.get("max_gen_toks", self.max_gen_toks)),
                block_size=self.max_length,
                eos_token_id=self.eot_token_id,
                do_sample=False,
            )
            text = self.tokenizer.decode(
                generated[0, len(context_ids):].tolist(), skip_special_tokens=True
            )
            for stop in until:
                if stop in text:
                    text = text.split(stop, 1)[0]
            outputs.append(text)
        return outputs
