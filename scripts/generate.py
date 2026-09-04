"""Generate text from a ForgeLM checkpoint.

Examples:

    python -m scripts.generate \
        --config configs/base.yml \
        --checkpoint checkpoints/pretrain_base/step_006000.pt \
        --prompt "The future of artificial intelligence is"

Interactive chat mode:

    python -m scripts.generate --config configs/base.yml \
        --checkpoint checkpoints/sft/best.pt --chat
"""

import argparse
from pathlib import Path

import torch

from config import load_config
from data_pipeline.sft_data import (
    ConversationFormatError,
    format_generation_prompt,
)
from training.common import (
    build_model_and_tokenizer,
    load_model_state,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with ForgeLM")
    parser.add_argument("--config", default="configs/base.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    experiment = load_config(config_path)
    model, tokenizer, model_config = build_model_and_tokenizer(experiment, device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    load_model_state(model, checkpoint["model"])
    model.eval()
    return model, tokenizer, model_config


def _decode_reply(tokenizer, generated_ids: list[int]) -> str:
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not text.strip() and generated_ids == [tokenizer.eos_token_id]:
        return "[EOS]"
    return text


def generate_reply(
    model,
    tokenizer,
    model_config,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("prompt produced no tokens")

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )
    output = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        block_size=model_config.block_size,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    generated_ids = output[0, input_tensor.size(1):].tolist()
    return _decode_reply(tokenizer, generated_ids)


def run_chat(model, tokenizer, model_config, args, device) -> None:
    messages: list[dict[str, str]] = []
    print("ForgeLM chat. Type /exit to stop and /clear to reset.\n")

    while True:
        user_text = input("you> ").strip()
        if user_text == "/exit":
            break
        if user_text == "/clear":
            messages.clear()
            print("conversation cleared")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            prompt_ids = format_generation_prompt(messages, tokenizer)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output = model.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                block_size=model_config.block_size,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=args.sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            reply = _decode_reply(
                tokenizer, output[0, input_tensor.size(1):].tolist()
            )
        except ConversationFormatError:
            messages.pop()
            continue
        print(f"assistant> {reply}\n", flush=True)
        messages.append({"role": "assistant", "content": reply})


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)

    device = resolve_device(args.device)
    model, tokenizer, model_config = load_model(
        args.config,
        args.checkpoint,
        device,
    )
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.chat:
        run_chat(model, tokenizer, model_config, args, device)
        return

    prompt = args.prompt
    if prompt is None:
        prompt = input("prompt> ")
    print(generate_reply(model, tokenizer, model_config, prompt, args, device), flush=True)


if __name__ == "__main__":
    main()
