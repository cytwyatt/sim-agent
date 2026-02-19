from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sim_agent.config import load_config
from sim_agent.pipeline import run_topic
from sim_agent.store.json_store import JsonStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sim-agent", description="Unified simulation literature agent.")
    parser.add_argument("--config", default="sim_agent.toml", help="Path to config TOML (default: sim_agent.toml).")

    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="Run a new literature search and extraction job.")
    run_cmd.add_argument("--topic", required=True, help="Research topic/query.")
    run_cmd.add_argument("--top-n", type=int, default=None, help="Number of papers to deeply process.")
    run_cmd.add_argument("--years", type=int, default=None, help="Publication window in years.")
    run_cmd.add_argument("--output-dir", default=None, help="Output root directory.")
    run_cmd.add_argument("--deep-profiles", default=None, help="Comma-separated deep profiles (e.g., MD,QMMM).")
    run_cmd.add_argument(
        "--custom-field",
        action="append",
        default=[],
        help="Custom extraction field name. Repeat flag to provide multiple values.",
    )
    run_cmd.add_argument("--no-sqlite", action="store_true", help="Disable SQLite indexing.")

    inspect_cmd = sub.add_parser("inspect", help="Inspect one saved paper JSON record.")
    inspect_cmd.add_argument("--run-id", required=True)
    inspect_cmd.add_argument("--paper-id", required=True)
    inspect_cmd.add_argument("--output-dir", default=None, help="Output root directory.")

    export_cmd = sub.add_parser("export", help="Export a saved run summary.")
    export_cmd.add_argument("--run-id", required=True)
    export_cmd.add_argument("--format", choices=["markdown", "json", "html"], default="markdown")
    export_cmd.add_argument("--output-dir", default=None, help="Output root directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    config = load_config(args.config)
    output_dir = args.output_dir or config.defaults.output_dir
    store = JsonStore(output_dir)

    if args.command == "run":
        top_n = args.top_n or config.defaults.top_n
        years = args.years or config.defaults.years
        deep_profiles = (
            [x.strip().upper() for x in args.deep_profiles.split(",") if x.strip()]
            if args.deep_profiles
            else [x.upper() for x in config.defaults.deep_profiles]
        )
        use_sqlite = not args.no_sqlite and config.defaults.use_sqlite
        result = run_topic(
            topic=args.topic,
            config=config,
            top_n=top_n,
            years=years,
            output_dir=output_dir,
            deep_profiles=deep_profiles,
            custom_fields=args.custom_field,
            use_sqlite=use_sqlite,
        )
        print(f"Run complete: {result.run_id}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Markdown: {result.markdown_path}")
        print(f"HTML: {result.html_path}")
        print(f"JSON summary: {result.summary_json_path}")
        print(f"Candidate titles: {result.candidate_titles_path}")
        print(f"Papers analyzed: {len(result.records)}")
        return 0

    if args.command == "inspect":
        try:
            payload = store.load_paper(args.run_id, args.paper_id)
        except FileNotFoundError:
            print("Record not found.", file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "export":
        run_dir = Path(output_dir) / "runs" / args.run_id
        if not run_dir.exists():
            print(f"Run not found: {args.run_id}", file=sys.stderr)
            return 2
        if args.format == "markdown":
            print(store.load_markdown(args.run_id))
            return 0
        if args.format == "html":
            print(store.load_html(args.run_id))
            return 0
        print(json.dumps(store.load_aggregate_json(args.run_id), indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
