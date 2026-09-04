"""Run ForgeLM through EleutherAI's official evaluation harness.

Example:
    python -m eval.evaluate_official \
        --config configs/base.yml \
        --checkpoint checkpoints/pretrain_base/step_006800.pt \
        --tasks hellaswag \
        --output reports/forgelm_official.json

Install the optional dependency first:
    pip install "lm-eval[hf]==0.4.8"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official ForgeLM evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-gen-toks", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        import lm_eval
        from lm_eval.utils import handle_non_serializable
    except (ImportError, AttributeError) as error:
        raise SystemExit(
            "The official evaluator dependencies are incompatible. Run: "
            "pip install --upgrade --force-reinstall "
            "'lm-eval[hf]==0.4.8' 'transformers>=4.45,<5'"
        ) from error

    from eval.forgelm_lm import ForgeLMLM

    args = parse_args()
    model = ForgeLMLM(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, default=handle_non_serializable),
        encoding="utf-8",
    )
    print(json.dumps(results.get("results", results), indent=2, default=handle_non_serializable))
    print(f"official evaluation saved to: {output}")


if __name__ == "__main__":
    main()
