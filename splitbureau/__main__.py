"""
CLI entry point for splitbureau:
  - python -m splitbureau bench
  - python -m splitbureau serve --port 8017
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import uvicorn

from .bench import format_benchmark_report, run_full_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splitbureau",
        description="Vertically Partitioned DH-PSI & Split-Learning Credit Bureau CLI",
    )
    sub = parser.add_subparsers(dest="command")

    bench_p = sub.add_parser("bench", help="Run full DH-PSI + SplitNN + Inversion benchmark")
    bench_p.add_argument("--samples", type=int, default=900, help="Number of common customers")
    bench_p.add_argument("--epochs", type=int, default=160, help="Training epochs")
    bench_p.add_argument("--seed", type=int, default=42, help="Random seed")
    bench_p.add_argument("--json", action="store_true", help="Emit JSON output")

    serve_p = sub.add_parser("serve", help="Start FastAPI server")
    serve_p.add_argument("--host", type=str, default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8017)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "serve":
        if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("SPLITBUREAU_API_KEY"):
            print("Refusing to listen on a non-loopback address without SPLITBUREAU_API_KEY set.", file=sys.stderr)
            return 2
        uvicorn.run("splitbureau.api:app", host=args.host, port=args.port, reload=False)
        return 0

    # Default to `bench`
    samples = getattr(args, "samples", 900)
    epochs = getattr(args, "epochs", 160)
    seed = getattr(args, "seed", 42)
    as_json = getattr(args, "json", False)

    summary = run_full_benchmark(n_common=samples, epochs=epochs, seed=seed)
    if as_json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(format_benchmark_report(summary))
    return 0 if summary.all_assertions_passed else 1


if __name__ == "__main__":
    sys.exit(main())
