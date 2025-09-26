from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from importlib import metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata as metadata  # type: ignore[assignment]

from Marketing_analytics.ai import (
    BRIEF_JSON_SCHEMA,
    generate_brief_md,
    generate_verified_brief,
    lint_brief_schema,
)
from Marketing_analytics.daily_master import is_daily_master_config

SCHEMA_FAIL_HEADER = "[Brief] Schema check failed – fix these first:"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "daily_master.json"
ARTIFACT_ROOT = PROJECT_ROOT / "reports" / "daily_master"


def _print_schema_violations(violations: Iterable[str]) -> None:
    print(SCHEMA_FAIL_HEADER)
    for violation in violations:
        print(f"- {violation}")


def _schema_check(verbose: bool = True) -> int:
    violations = lint_brief_schema(deepcopy(BRIEF_JSON_SCHEMA))
    if violations:
        if verbose:
            _print_schema_violations(violations)
        return 1
    if verbose:
        print("OK")
    return 0


def _fmt_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _run_etl(args: argparse.Namespace) -> int:
    if args.quick and args.full:
        print("[ETL] Use either --quick or --full, not both.")
        return 2
    config_path = Path(args.config or DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        print(f"[ETL] Config file not found: {config_path}")
        return 1
    if not is_daily_master_config(config_path):
        print(f"[ETL] Config does not look like a daily master spec: {config_path}")
        return 1

    until = "kpis" if args.quick else "write"
    cli_script = PROJECT_ROOT / "cli" / "marketing_analysis.py"
    cmd = [sys.executable, str(cli_script), "--config", str(config_path), "--until", until]
    if args.debug:
        cmd.append("--debug")
    print(f"[ETL] Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _run_brief(args: argparse.Namespace) -> int:
    config_path = Path(args.config or DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        print(f"[Brief] Config file not found: {config_path}")
        return 1

    overrides: Dict[str, Optional[str]] = {}
    if args.model:
        overrides["OPENAI_MODEL_ANALYSIS"] = args.model
    if args.tokens:
        overrides["OPENAI_MAX_OUTPUT_TOKENS"] = str(args.tokens)
    if not args.md and args.days:
        overrides["OPENAI_SERIES_DAYS"] = str(args.days)

    previous_env: Dict[str, Optional[str]] = {}
    def _apply_override(key: str, value: Optional[str]) -> None:
        previous_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        for key, value in overrides.items():
            _apply_override(key, value)

        if args.md:
            results = generate_brief_md(str(config_path))
        else:
            if _schema_check(verbose=False) != 0:
                _print_schema_violations(lint_brief_schema(deepcopy(BRIEF_JSON_SCHEMA)))
                return 1
            _apply_override("BRIEF_SKIP_VERIFIER", None if args.verify else "1")
            results = generate_verified_brief(str(config_path))
    except Exception as exc:
        print(f"[Brief] Failed: {exc}")
        return 1
    finally:
        for key, original in previous_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    print("Brief generation complete:")
    if 'brief_md' in results:
        print(f"- brief_md: {results['brief_md']}")
        print(f"- brief_raw: {results['brief_raw']}")
    else:
        for label, path in results.items():
            print(f"- {label}: {path}")
    return 0


def _run_schema(args: argparse.Namespace) -> int:
    if not args.check:
        print("Use --check to lint the schema.")
        return 2
    return _schema_check(verbose=True)


def _run_ui(_: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "dashboard.py"
    if not script.exists():
        print(f"[UI] dashboard.py not found at {script}")
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(script)]
    print("[UI] Launching Streamlit dashboard...")
    return subprocess.call(cmd)


def _status_payload() -> Optional[Dict[str, Any]]:
    status_path = ARTIFACT_ROOT / "status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_status(_: argparse.Namespace) -> int:
    payload = _status_payload()
    if not payload:
        print(f"[Status] No status.json found in {_fmt_rel(ARTIFACT_ROOT)}")
        return 0
    stage = payload.get("stage", "unknown")
    pct = payload.get("pct")
    stamp = payload.get("updated_at") or payload.get("timestamp")
    pct_text = f"{pct:.0%}" if isinstance(pct, (int, float)) else "n/a"
    print(f"[Status] stage={stage} pct={pct_text} updated_at={stamp}")
    return 0


def _run_doctor(_: argparse.Namespace) -> int:
    print("[Doctor] Environment check")
    print(f"- Python: {sys.version.split()[0]}")
    packages = ["pandas", "numpy", "streamlit", "openai", "anthropic"]
    for pkg in packages:
        try:
            version = metadata.version(pkg)
            print(f"- {pkg}: {version}")
        except metadata.PackageNotFoundError:
            print(f"- {pkg}: missing")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"- {key}: {'set' if os.getenv(key) else 'missing'}")
    print(f"- Config: {'ok' if DEFAULT_CONFIG.exists() else 'missing'} ({_fmt_rel(DEFAULT_CONFIG)})")
    print(f"- Artifacts: {'ok' if ARTIFACT_ROOT.exists() else 'missing'} ({_fmt_rel(ARTIFACT_ROOT)})")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    if not ARTIFACT_ROOT.exists():
        print(f"[Clean] Nothing to remove in {_fmt_rel(ARTIFACT_ROOT)}")
        return 0
    items = sorted(ARTIFACT_ROOT.iterdir())
    if not items:
        print(f"[Clean] {_fmt_rel(ARTIFACT_ROOT)} already empty")
        return 0
    print("[Clean] The following artifacts will be removed:")
    for item in items:
        marker = "/" if item.is_dir() else ""
        print(f"  - {_fmt_rel(item)}{marker}")
    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("[Clean] Aborted")
            return 0
    for item in items:
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    print(f"[Clean] Removed artifacts in {_fmt_rel(ARTIFACT_ROOT)}")
    return 0


def _run_artifacts(_: argparse.Namespace) -> int:
    if not ARTIFACT_ROOT.exists():
        print(f"[Artifacts] Directory not found: {_fmt_rel(ARTIFACT_ROOT)}")
        return 1
    entries = sorted(ARTIFACT_ROOT.glob('*'))
    if not entries:
        print(f"[Artifacts] No files found in {_fmt_rel(ARTIFACT_ROOT)}")
        return 0
    print(f"[Artifacts] Listing contents of {_fmt_rel(ARTIFACT_ROOT)}")
    for path in entries:
        label = f"{path.name}/" if path.is_dir() else path.name
        stats = path.stat()
        mtime = datetime.fromtimestamp(stats.st_mtime).isoformat(timespec='seconds')
        size = stats.st_size
        print(f"  - {label:25s} {mtime} ({size} bytes)")
    return 0


def _run_test(_: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    return subprocess.call(cmd)


def _run_models(_: argparse.Namespace) -> int:
    try:
        from openai import OpenAI  # type: ignore[misc]
    except ImportError:
        print("[Models] openai package not installed.")
        return 1
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[Models] OPENAI_API_KEY not set.")
        return 1
    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
    except Exception as exc:
        print(f"[Models] Failed to list models: {exc}")
        return 1
    current = os.getenv("OPENAI_MODEL_ANALYSIS", "gpt-5")
    print(f"[Models] Available models (highlighting OPENAI_MODEL_ANALYSIS={current}):")
    for model in models.data:
        name = getattr(model, "id", getattr(model, "name", "unknown"))
        marker = "*" if name == current else "-"
        print(f"{marker} {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project management utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Validate the brief JSON schema")
    schema_parser.add_argument("--check", action="store_true", help="Run schema lint")
    schema_parser.set_defaults(func=_run_schema)

    etl_parser = subparsers.add_parser("etl", help="Execute the daily master pipeline")
    etl_parser.add_argument("--config", help="Config file", default=str(DEFAULT_CONFIG))
    etl_parser.add_argument("--quick", action="store_true", help="Fast KPI smoke run")
    etl_parser.add_argument("--full", action="store_true", help="Complete run (default)")
    etl_parser.add_argument("--debug", action="store_true", help="Print stage banners and artifact timestamps")
    etl_parser.set_defaults(func=_run_etl)

    brief_parser = subparsers.add_parser("brief", help="Generate an executive brief")
    brief_parser.add_argument("--config", help="Config file", default=str(DEFAULT_CONFIG))
    brief_parser.add_argument("--model", help="OpenAI model", default="gpt-5-chat-latest")
    brief_parser.add_argument("--days", type=int, help="Series window days (JSON route only)", default=120)
    brief_parser.add_argument("--tokens", type=int, help="Max output tokens", default=3000)
    brief_parser.add_argument("--md", action="store_true", help="Write Markdown-only brief (chat route, no verifier)")
    brief_parser.add_argument("--verify", action="store_true", default=True, help="Run verifier (default)")
    brief_parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip Anthropic verifier")
    brief_parser.set_defaults(func=_run_brief)

    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit dashboard")
    ui_parser.set_defaults(func=_run_ui)

    status_parser = subparsers.add_parser("status", help="Print latest run status")
    status_parser.set_defaults(func=_run_status)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect environment readiness")
    doctor_parser.set_defaults(func=_run_doctor)

    artifacts_parser = subparsers.add_parser("artifacts", help="List generated artifacts")
    artifacts_parser.set_defaults(func=_run_artifacts)

    clean_parser = subparsers.add_parser("clean", help="Clear generated artifacts")
    clean_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    clean_parser.set_defaults(func=_run_clean)

    test_parser = subparsers.add_parser("test", help="Run pytest -q")
    test_parser.set_defaults(func=_run_test)

    models_parser = subparsers.add_parser("models", help="List available OpenAI models")
    models_parser.set_defaults(func=_run_models)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.error("No command specified")
    return int(func(args))


if __name__ == "__main__":
    sys.exit(main())


