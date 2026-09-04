from types import SimpleNamespace

import pytest

from data_pipeline.sft_data import (
    ConversationFormatError,
    format_conversation,
    format_generation_prompt,
    normalize_messages,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) % self.vocab_size for character in text]


def test_normalize_messages_accepts_common_role_aliases():
    messages = normalize_messages(
        [
            {"role": "human", "content": "Hello"},
            {"role": "gpt", "content": "Hi"},
        ]
    )

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "answer"}],
        [
            {"role": "user", "content": "question"},
            {"role": "user", "content": "another question"},
        ],
        [
            {"role": "user", "content": "question"},
        ],
    ],
)
def test_normalize_messages_rejects_invalid_conversations(messages):
    with pytest.raises(ConversationFormatError):
        normalize_messages(messages)


def test_format_conversation_masks_non_assistant_tokens():
    formatted = format_conversation(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        FakeTokenizer(),
    )

    assert len(formatted.token_ids) == len(formatted.loss_mask)
    assert formatted.token_ids[0] == 1
    assert formatted.token_ids[-1] == 2
    assert formatted.loss_mask[-1] == 1
    assert formatted.assistant_token_count == 3  # response, newline, and EOS

    assistant_text_start = formatted.token_ids.index(ord("4"))
    assert formatted.loss_mask[assistant_text_start] == 1
    assert all(mask == 0 for mask in formatted.loss_mask[:assistant_text_start])


def test_formatter_preserves_multiple_turns_and_supervises_each_answer():
    formatted = format_conversation(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second"},
        ],
        FakeTokenizer(),
    )

    assert formatted.roles == ("user", "assistant", "user", "assistant")
    assert formatted.assistant_token_count == len("first") + len("second") + 4


def test_format_conversation_requires_eos_token():
    tokenizer = SimpleNamespace(
        bos_token_id=1,
        eos_token_id=None,
        encode=lambda text, add_special_tokens=False: [1],
    )

    with pytest.raises(ConversationFormatError, match="eos_token_id"):
        format_conversation(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            tokenizer,
        )


def test_generation_prompt_uses_shared_role_format():
    prompt = format_generation_prompt(
        [{"role": "user", "content": "Hello"}],
        FakeTokenizer(),
    )
    text = "".join(chr(token) for token in prompt[1:])
    assert text == "user: Hello\nassistant: "
