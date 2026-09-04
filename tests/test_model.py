import unittest
from dataclasses import replace

import torch

from config import (
    ModelConfig,
    load_config,
    resolve_model_config,
)
from model.model import (
    CausalSelfAttention,
    GPT,
    LMHead,
    RmsNorm,
    RotaryEmbedding,
    SwiGLU,
    TokenEmbedding,
    TransformerBlock,
    count_parameters,
)


class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig(
            vocab_size=50,
            block_size=16,
            n_layer=2,
            n_embd=32,
            n_head=8,
            n_head_kv=2,
            intermediate_size=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            tie_embeddings=True,
        )

    def setUp(self):
        torch.manual_seed(0)

    def test_yaml_config_loads(self):
        experiment = load_config("configs/local_3050.yml")

        self.assertEqual(experiment.model.n_head, 8)
        self.assertEqual(experiment.model.n_head_kv, 2)
        self.assertIsNone(experiment.model.vocab_size)
        self.assertEqual(
            experiment.tokenizer.name,
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        )
        self.assertGreater(experiment.training.learning_rate, 0)

        base_experiment = load_config("configs/base.yml")
        self.assertEqual(base_experiment.model.block_size, 2048)
        self.assertEqual(base_experiment.model.n_layer, 30)
        self.assertEqual(base_experiment.training.max_steps, 1_464_844)
        self.assertTrue(base_experiment.model.use_sdpa)

    def test_token_embedding_shape(self):
        ids = torch.randint(0, self.config.vocab_size, (2, 7))
        output = TokenEmbedding(self.config)(ids)

        self.assertEqual(output.shape, (2, 7, self.config.n_embd))

    def test_rope_shape_and_norm_preservation(self):
        rope = RotaryEmbedding(self.config)
        x = torch.randn(2, self.config.n_head, 7, 4)
        output = rope(x)

        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(
            output.norm(dim=-1),
            x.norm(dim=-1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rope_position_zero_is_identity(self):
        rope = RotaryEmbedding(self.config)
        x = torch.randn(2, self.config.n_head, 1, 4)

        torch.testing.assert_close(rope(x), x)

    def test_gqa_head_configuration(self):
        attention = CausalSelfAttention(self.config)
        x = torch.randn(2, 7, self.config.n_embd)

        q = attention.q_proj(x).reshape(
            2, 7, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        k = attention.k_proj(x).reshape(
            2, 7, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)
        v = attention.v_proj(x).reshape(
            2, 7, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)

        self.assertEqual(q.shape, (2, 8, 7, 4))
        self.assertEqual(k.shape, (2, 2, 7, 4))
        self.assertEqual(v.shape, (2, 2, 7, 4))

        repeated_k = k.repeat_interleave(attention.repeat_factor, dim=1)
        repeated_v = v.repeat_interleave(attention.repeat_factor, dim=1)
        self.assertEqual(repeated_k.shape, q.shape)
        self.assertEqual(repeated_v.shape, q.shape)

    def test_attention_shape_and_gradients(self):
        attention = CausalSelfAttention(self.config)
        x = torch.randn(2, 7, self.config.n_embd, requires_grad=True)
        output = attention(x)

        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_manual_attention_matches_sdpa(self):
        manual_config = replace(self.config, use_sdpa=False)
        sdpa_config = replace(self.config, use_sdpa=True)
        manual = CausalSelfAttention(manual_config).eval()
        optimized = CausalSelfAttention(sdpa_config).eval()
        optimized.load_state_dict(manual.state_dict())
        x = torch.randn(2, 7, self.config.n_embd)

        torch.testing.assert_close(
            manual(x),
            optimized(x),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_attention_is_causal(self):
        attention = CausalSelfAttention(self.config).eval()
        x = torch.randn(1, 7, self.config.n_embd)
        changed_x = x.clone()
        changed_x[:, 6, :] += 100.0

        original = attention(x)
        changed = attention(changed_x)

        # Changing the final token cannot affect earlier positions.
        torch.testing.assert_close(
            original[:, :6, :],
            changed[:, :6, :],
        )

    def test_swiglu_shape_and_gradients(self):
        mlp = SwiGLU(self.config)
        x = torch.randn(2, 7, self.config.n_embd, requires_grad=True)
        output = mlp(x)

        self.assertEqual(output.shape, x.shape)
        output.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rmsnorm_known_input(self):
        norm = RmsNorm(self.config)
        x = torch.ones(2, 7, self.config.n_embd)

        torch.testing.assert_close(norm(x), x)

    def test_lm_head_shape(self):
        head = LMHead(self.config)
        x = torch.randn(2, 7, self.config.n_embd)

        self.assertEqual(
            head(x).shape,
            (2, 7, self.config.vocab_size),
        )

    def test_transformer_block_shape_and_gradients(self):
        block = TransformerBlock(self.config)
        x = torch.randn(2, 7, self.config.n_embd, requires_grad=True)
        output = block(x)

        self.assertEqual(output.shape, x.shape)
        output.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_full_model_logits_loss_and_gradients(self):
        model = GPT(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        targets = torch.randint(0, self.config.vocab_size, (2, 7))

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()

        self.assertEqual(
            logits.shape,
            (2, 7, self.config.vocab_size),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_generate_appends_tokens_and_respects_context(self):
        model = GPT(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 7))

        output = model.generate(
            input_ids,
            max_new_tokens=4,
            block_size=5,
        )

        self.assertEqual(output.shape, (1, 11))

    def test_sampled_generation_is_reproducible_with_seed(self):
        model = GPT(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))

        torch.manual_seed(123)
        first = model.generate(
            input_ids,
            max_new_tokens=3,
            block_size=5,
            do_sample=True,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )
        torch.manual_seed(123)
        second = model.generate(
            input_ids,
            max_new_tokens=3,
            block_size=5,
            do_sample=True,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )

        torch.testing.assert_close(first, second)

    def test_input_and_output_embeddings_are_tied(self):
        model = GPT(self.config)

        self.assertIs(
            model.token_embedding.embedding.weight,
            model.lm_head.proj.weight,
        )

    def test_model_parameter_count_is_positive(self):
        model = GPT(self.config)

        self.assertGreater(count_parameters(model), 0)

    def test_tokenizer_resolves_model_vocabulary_size(self):
        tokenizer = type("TokenizerStub", (), {"vocab_size": 123})()
        config = replace(self.config, vocab_size=None)

        resolved = resolve_model_config(config, tokenizer)

        self.assertEqual(resolved.vocab_size, 123)
        self.assertIsNone(config.vocab_size)

    def test_mismatched_tokenizer_vocabulary_is_rejected(self):
        tokenizer = type("TokenizerStub", (), {"vocab_size": 123})()

        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_model_config(self.config, tokenizer)

    def test_parameter_count_can_include_frozen_parameters(self):
        model = GPT(self.config)
        total_parameters = count_parameters(model, trainable_only=False)

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.assertEqual(count_parameters(model), 0)
        self.assertEqual(
            count_parameters(model, trainable_only=False),
            total_parameters,
        )

    def test_invalid_model_config_is_rejected(self):
        invalid_configs = [
            {"n_embd": 31},
            {"n_head": 0},
            {"n_head_kv": 0},
            {"n_head_kv": 3},
            {"vocab_size": 0},
            {"n_layer": 0},
            {"block_size": 0},
            {"rope_theta": 0.0},
            {"rms_norm_eps": 0.0},
        ]

        for changes in invalid_configs:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    ModelConfig(**{
                        "vocab_size": 50,
                        "block_size": 16,
                        "n_layer": 2,
                        "n_embd": 32,
                        "n_head": 8,
                        "n_head_kv": 2,
                        "intermediate_size": 64,
                        "rope_theta": 10000.0,
                        "rms_norm_eps": 1e-6,
                        "attention_bias": False,
                        "tie_embeddings": True,
                        **changes,
                    })

    def test_model_rejects_sequence_longer_than_block_size(self):
        model = GPT(self.config)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.block_size + 1),
        )

        with self.assertRaisesRegex(ValueError, "exceeds block size"):
            model(input_ids)

    def test_model_rejects_non_2d_input_ids(self):
        model = GPT(self.config)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (2, 4, 4),
        )

        with self.assertRaisesRegex(ValueError, "input_ids"):
            model(input_ids)

    def test_gqa_repeats_each_kv_head(self):
        attention = CausalSelfAttention(self.config)

        k = torch.tensor([
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        ])

        repeated = k.repeat_interleave(4, dim=1)

        expected = torch.tensor([
            [
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
            ]
        ])

        torch.testing.assert_close(repeated, expected)

    def test_model_accepts_max_sequence_length(self):
        model = GPT(self.config)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.block_size),
        )

        logits = model(input_ids)

        self.assertEqual(
            logits.shape,
            (2, self.config.block_size, self.config.vocab_size),
        )
    def test_embeddings_are_not_tied_when_disabled(self):
        config = replace(
            self.config,
            tie_embeddings=False
        )

        model = GPT(config)

        self.assertIsNot(
            model.token_embedding.embedding.weight,
            model.lm_head.proj.weight,
        )

    def test_attention_allows_current_token(self):
        attention = CausalSelfAttention(self.config).eval()

        x = torch.randn(1, 7, self.config.n_embd)
        changed_x = x.clone()
        changed_x[:, 6, :] += 100.0

        original = attention(x)
        changed = attention(changed_x)

        self.assertFalse(
            torch.allclose(
                original[:, 6, :],
                changed[:, 6, :],
            )
        )

if __name__ == "__main__":
    unittest.main()
