import torch
from torch import nn 
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size,config.n_embd)

    def forward(self, x):
        return self.embedding(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.n_embd // config.n_head
        self.base = config.rope_theta

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2).float() / self.head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, H, T, D = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        freqs = torch.outer(
            positions,
            self.inv_freq
        )

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        x = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        )

        return x.flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads = config.n_head
        self.num_kv_heads = config.n_head_kv 
        self.head_dim = config.n_embd // config.n_head
        self.repeat_factor = self.num_heads // self.num_kv_heads
        self.use_sdpa = config.use_sdpa
        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.n_embd,
            self.head_dim * self.num_kv_heads,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.n_embd,
            self.head_dim * self.num_kv_heads,
            bias=config.attention_bias
        )
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.attention_bias
        )

        self.rope = RotaryEmbedding(config)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                    dtype=torch.bool
                )
            )
        )

    def forward(self, x):
        B, T, C = x.size()

        if T > self.causal_mask.size(0):
            raise ValueError(
                f"sequence length {T} exceeds block size "
                f"{self.causal_mask.size(0)}"
            )

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_head, T, head_dim)
        q = self.rope(q)
        k = self.rope(k)
        k=k.repeat_interleave(self.repeat_factor, dim=1)  
        v=v.repeat_interleave(self.repeat_factor, dim=1)  
        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
        else:

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = attn_weights.masked_fill(
                ~self.causal_mask[:T, :T],
                float("-inf"),
            )
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        output = self.out_proj(attn_output)  # (B, T, C)
        return output   

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        #down projection
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class RmsNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        norm_x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight



class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.proj = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

    def forward(self, x):
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RmsNorm(config)
        self.attention = CausalSelfAttention(config)

        self.mlp_norm = RmsNorm(config)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attention(
            self.attn_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = TokenEmbedding(config)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.final_norm = RmsNorm(config)

        self.lm_head = LMHead(config)

        self.apply(self._init_weights)

        if config.tie_embeddings:
            self.lm_head.proj.weight = self.token_embedding.embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def hidden_states(self, input_ids):
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        block_size: int,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens without a KV cache.

        At each step the model sees the last ``block_size`` tokens. This is
        intentionally the simple reference implementation: it is easy to
        understand, but slower than cached inference because the prefix is
        recomputed after every generated token.
        """
        generated = input_ids
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = generated[:, -block_size:]
            logits = self(context)[:, -1, :]

            if not do_sample:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1]
                    logits = logits.masked_fill(
                        logits < threshold.unsqueeze(-1),
                        float("-inf"),
                    )

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove = cumulative_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(
                        remove,
                        float("-inf"),
                    )
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(
                        -1,
                        sorted_indices,
                        sorted_logits,
                    )

                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and torch.all(
                next_token == eos_token_id
            ):
                break

        if was_training:
            self.train()
        return generated

    def forward(self, input_ids, targets=None):
        x = self.hidden_states(input_ids)

        if targets is None:
            return self.lm_head(x)

        logits = self.lm_head(x)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()

    if trainable_only:
        parameters = (
            parameter
            for parameter in parameters
            if parameter.requires_grad
        )

    return sum(parameter.numel() for parameter in parameters)
