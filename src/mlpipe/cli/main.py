from __future__ import annotations
import argparse
from pathlib import Path
from mlpipe.core.registry import list_blocks
from mlpipe.pipelines.xgb_basic.run import run_pipeline
import mlpipe.blocks  # Import to register all blocks

def main():
    parser = argparse.ArgumentParser("mlpipe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a pipeline")
    p_run.add_argument("--pipeline", default="xgb_basic")
    p_run.add_argument("--config-path", default="configs")
    p_run.add_argument("--config-name", default="pipeline")

    sub.add_parser("list-blocks", help="List available blocks")

    args = parser.parse_args()
    if args.cmd == "run":
        run_pipeline(pipeline=args.pipeline, config_path=args.config_path, config_name=args.config_name)
    elif args.cmd == "list-blocks":
        for name in list_blocks():
            print(name)

if __name__ == "__main__":
    main()
