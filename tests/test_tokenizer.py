import os
import tempfile
import unittest
from unittest.mock import patch

from tokenizer import LlamaTokenizer


class FakeTokenizerBackend:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def __init__(self):
        self.saved_path = None

    def __len__(self):
        return 8

    def encode(self, text, add_special_tokens=True):
        self.encode_args = (text, add_special_tokens)
        return [1, 3, 4, 2] if add_special_tokens else [3, 4]

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_args = (token_ids, skip_special_tokens)
        return "decoded text"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
    ):
        self.chat_args = (
            messages,
            tokenize,
            add_generation_prompt,
        )
        return [1, 3, 4, 2] if tokenize else "formatted chat"

    def save_pretrained(self, path):
        self.saved_path = path
        return (path,)


class TokenizerTestCase(unittest.TestCase):
    def setUp(self):
        self.backend = FakeTokenizerBackend()
        self.loader = patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=self.backend,
        )
        self.from_pretrained = self.loader.start()
        self.addCleanup(self.loader.stop)

        self.tokenizer = LlamaTokenizer(
            name="test/llama-tokenizer",
            revision="test-revision",
            local_files_only=True,
        )

    def test_loads_pretrained_tokenizer_with_configuration(self):
        self.from_pretrained.assert_called_once_with(
            "test/llama-tokenizer",
            revision="test-revision",
            local_files_only=True,
            use_fast=True,
        )

    def test_special_token_ids(self):
        self.assertEqual(self.tokenizer.bos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertIsNone(self.tokenizer.pad_token_id)

    def test_vocab_size_includes_added_tokens(self):
        self.assertEqual(self.tokenizer.vocab_size, 8)

    def test_encode_forwards_special_token_option(self):
        self.assertEqual(
            self.tokenizer.encode("hello", add_special_tokens=False),
            [3, 4],
        )
        self.assertEqual(
            self.backend.encode_args,
            ("hello", False),
        )

    def test_decode_accepts_any_integer_iterable(self):
        token_ids = (token_id for token_id in [1, 3, 2])

        self.assertEqual(
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            ),
            "decoded text",
        )
        self.assertEqual(
            self.backend.decode_args,
            ([1, 3, 2], False),
        )

    def test_chat_template_can_return_tokens(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        self.assertEqual(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            ),
            [1, 3, 4, 2],
        )
        self.assertEqual(
            self.backend.chat_args,
            (messages, True, True),
        )

    def test_chat_template_can_return_formatted_text(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        self.assertEqual(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            ),
            "formatted chat",
        )

    def test_save_pretrained_forwards_path(self):
        self.assertEqual(
            self.tokenizer.save_pretrained("artifacts/tokenizer"),
            ("artifacts/tokenizer",),
        )
        self.assertEqual(
            self.backend.saved_path,
            "artifacts/tokenizer",
        )


@unittest.skipUnless(
    os.getenv("RUN_TOKENIZER_INTEGRATION") == "1",
    "set RUN_TOKENIZER_INTEGRATION=1 to load the real tokenizer",
)
class RealLlamaTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LlamaTokenizer(
            name=(
                "TinyLlama/"
                "TinyLlama-1.1B-intermediate-step-1431k-3T"
            )
        )

    def test_real_tinyllama_tokenizer_round_trip(self):
        text = "Hello, world!"
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        self.assertGreater(self.tokenizer.vocab_size, 0)
        self.assertGreater(len(token_ids), 0)
        self.assertEqual(
            self.tokenizer.decode(token_ids),
            text,
        )

    def test_real_tokenizer_handles_text_edge_cases(self):
        examples = [
            "",
            "   ",
            "Hello,   world!\nNew line.",
            "Numbers: 12345; symbols: @#$%^&*()",
            "Unicode: café — 世界 — مرحبا",
        ]

        for text in examples:
            with self.subTest(text=text):
                token_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                decoded = self.tokenizer.decode(token_ids)

                self.assertIsInstance(token_ids, list)
                self.assertIsInstance(decoded, str)

                if text.strip():
                    self.assertEqual(decoded, text)
                else:
                    # SentencePiece-style tokenizers may normalize runs of
                    # whitespace, but must still handle them safely.
                    self.assertIsInstance(decoded, str)

    def test_real_token_ids_are_within_vocabulary(self):
        token_ids = self.tokenizer.encode(
            "A long enough sentence with numbers 42 and Unicode 世界.",
            add_special_tokens=True,
        )

        self.assertTrue(
            all(
                0 <= token_id < self.tokenizer.vocab_size
                for token_id in token_ids
            )
        )

    def test_real_encoding_is_deterministic(self):
        text = "The same text must produce the same IDs."

        first = self.tokenizer.encode(text)
        second = self.tokenizer.encode(text)

        self.assertEqual(first, second)

    def test_real_special_token_configuration_is_valid(self):
        special_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]

        for token_id in special_token_ids:
            if token_id is not None:
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, self.tokenizer.vocab_size)

        with_special_tokens = self.tokenizer.encode(
            "hello",
            add_special_tokens=True,
        )
        without_special_tokens = self.tokenizer.encode(
            "hello",
            add_special_tokens=False,
        )

        self.assertGreaterEqual(
            len(with_special_tokens),
            len(without_special_tokens),
        )

    def test_real_long_text_can_be_encoded(self):
        text = "hello world " * 1000
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        self.assertGreater(len(token_ids), 0)
        self.assertTrue(
            all(
                0 <= token_id < self.tokenizer.vocab_size
                for token_id in token_ids
            )
        )

    def test_real_tokenizer_save_and_reload_is_consistent(self):
        text = "Saving and reloading must preserve tokenization."
        original_ids = self.tokenizer.encode(text)

        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save_pretrained(directory)
            reloaded = LlamaTokenizer(
                name=directory,
                local_files_only=True,
            )

            self.assertEqual(
                reloaded.vocab_size,
                self.tokenizer.vocab_size,
            )
            self.assertEqual(
                reloaded.encode(text),
                original_ids,
            )


if __name__ == "__main__":
    unittest.main()
