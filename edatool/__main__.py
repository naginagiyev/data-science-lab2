import os
import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import get_args
from edatool.schemas import Task
from dotenv import find_dotenv, load_dotenv
from edatool.api import audit, presplit, verify
from edatool.settings import deep_merge, read_json
from edatool.core.progress import detail_line, label_width

# ANSI color codes for the words that the terminal output highlights
colors = {
    "✓": "\033[32m✓",
    "✗": "\033[31m✗",
    "must_fix": "\033[31mmust_fix\033[0m",
    "should_fix": "\033[33mshould_fix\033[0m",
    "consider": "\033[36mconsider\033[0m",
}

# formats the short terminal lines, and adds colors and bold stage names when...
# the terminal supports it
class ScreenFormatter(logging.Formatter):
    def __init__(self, color):
        super().__init__("%(message)s")
        self.color = color

    def format(self, record):
        text = super().format(record)
        if not self.color:
            return text
        lines = []
        for line in text.split("\n"):
            for plain, painted in colors.items():
                line = line.replace(plain, painted)
            if "\033[3" in line and not line.endswith("\033[0m"):
                line += "\033[0m"
            if line[:2] == "  " and line[2:3].strip():
                line = "  \033[1m" + line[2 : label_width + 2] + "\033[0m" + line[label_width + 2 :]
            lines.append(line)
        return "\n".join(lines)

# checks if the terminal can show colors (windows terminal, VS Code or a unix...
# terminal)
def supports_color():
    terminal = os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") or os.environ.get("TERM")
    return sys.stdout.isatty() and bool(terminal) and "NO_COLOR" not in os.environ

# collects every log line in memory and saves them as run.log in the run...
# folder at the end
class RunLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        self.lines = []
        self.folder = None

    def emit(self, record):
        self.lines.append(self.format(record))
        if getattr(record, "run_folder", None) is not None:
            self.folder = record.run_folder

    def save(self, fallback):
        path = (self.folder / "run.log") if self.folder else fallback
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path

# builds the command line interface with the presplit, audit and verify...
# commands
def build_parser():
    parser = argparse.ArgumentParser(prog="edatool", description="Audit a tabular dataset.")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ["presplit", "audit"]:
        command = commands.add_parser(name)
        command.add_argument("data", help="path to the data file")
        command.add_argument("--target", required=True)
        command.add_argument("--task", required=True, choices=get_args(Task))
        command.add_argument("--settings", help="JSON file with settings overrides")
        command.add_argument("--out", default="runs", help="folder for the run files")
        if name == "audit":
            command.add_argument("--no-agent", action="store_true", help="deterministic checks only")
            command.add_argument("--sample-rows", type=int)
            command.add_argument("--cache", help="folder for cached agent decisions (default: <out>/agentcache)")

    verify_command = commands.add_parser("verify")
    verify_command.add_argument("report", help="path to the previous report.json")
    verify_command.add_argument("--snapshots", required=True, help="folder with step snapshots and steps.json")
    verify_command.add_argument("--out", default="runs", help="folder for the run files")
    verify_command.add_argument("--settings", help="JSON file with settings overrides")
    for command in commands.choices.values():
        command.add_argument("--quiet", action="store_true", help="show no progress, only the result path")
        command.add_argument("--verbose", action="store_true", help="also show detectors that found nothing")
    return parser

# entry point of the CLI: loads .env, sets up the terminal output and the log...
# file, then runs the chosen command
def main():
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logger = logging.getLogger("edatool")
    logger.setLevel(logging.DEBUG)
    screen = logging.StreamHandler(sys.stdout)
    screen.setFormatter(ScreenFormatter(supports_color()))
    screen.setLevel(logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO)
    logger.addHandler(screen)
    run_log = RunLog()
    logger.addHandler(run_log)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.command == "audit":
        settings = args.settings
        if not args.no_agent:
            cache = args.cache or str(Path(args.out) / "agentcache")
            settings = deep_merge(read_json(args.settings) if args.settings else {}, {"agent": {"cache_dir": cache}})
        audit(
            args.data,
            target=args.target,
            task=args.task,
            use_agent=not args.no_agent,
            settings=settings,
            sample_rows=args.sample_rows,
            run_dir=args.out,
        )
    elif args.command == "presplit":
        advice = presplit(args.data, target=args.target, task=args.task, settings=args.settings)
        logger.debug(advice.model_dump_json(indent=2))
    else:
        folder = Path(args.snapshots)
        steps = read_json(folder / "steps.json")
        snapshots = [(step["name"], pd.read_csv(folder / step["file"])) for step in steps["steps"]]
        step_issues = {step["name"]: step.get("issues", []) for step in steps["steps"]}
        test = pd.read_csv(folder / steps["test"]) if steps.get("test") else None
        verify(args.report, snapshots, step_issues, settings=args.settings, run_dir=args.out, lineage=steps.get("lineage"), test=test)
    path = run_log.save(Path(args.out) / f"{stamp}-{args.command}.log")
    if not args.quiet:
        print(detail_line(str(path), "full log"))
        print()

if __name__ == "__main__":
    main()
