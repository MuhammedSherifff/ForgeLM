from collections.abc import Iterable


class LlamaTokenizer:

    def __init__(
        self,
        name: str,
        revision: str = "main",
        local_files_only: bool = False,
    ):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LlamaTokenizer requires the 'transformers' package"
            ) from error

        self._tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=True,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def save_pretrained(self, path: str) -> tuple[str, ...]:
        return self._tokenizer.save_pretrained(path)
