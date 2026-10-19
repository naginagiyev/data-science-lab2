import sys
import argparse
import questionary
from pathlib import Path
from schemas import TASK_ALIASES
from settings import load_settings
from tools.runfolder import prepare_console
from modelling.fittedmodel import FittedModel
from orchestrator import VARIANTS, StopRun, run

# builds the command-line parser with the run, predict and report commands
def build_parser():
    parser = argparse.ArgumentParser(prog="main.py", description="Agentic AutoML v2")
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="train a model on one table")
    run_parser.add_argument("data", help="path of the data file")
    run_parser.add_argument("--target", help="target column (skips the framer)")
    run_parser.add_argument("--task", choices=sorted(TASK_ALIASES) + sorted(set(TASK_ALIASES.values())))
    run_parser.add_argument("--variant", default="full", choices=VARIANTS)
    run_parser.add_argument("--budget", type=float, help="total seconds of the run")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--backend", choices=["own", "autogluon", "flaml"])
    run_parser.add_argument("--ask", action="store_true", help="ask the user when the framer is unsure")
    run_parser.add_argument("--mode", default="demo", choices=["demo", "benchmark"])
    run_parser.add_argument("--out", help="folder for run folders (default runs/)")
    run_parser.add_argument("--settings", help="JSON file merged over configuration/settings.json")
    run_parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, e.g. gate.enabled=false")
    predict_parser = commands.add_parser("predict", help="predict new rows with a saved model")
    predict_parser.add_argument("model", help="path of model.joblib")
    predict_parser.add_argument("data", help="path of the new data")
    predict_parser.add_argument("--out", default="predictions.csv")
    report_parser = commands.add_parser("report", help="print the report of a run folder")
    report_parser.add_argument("run_dir", help="path of the run folder")
    return parser

# turns the run arguments into settings: file, --set overrides and the short flags
def run_settings(arguments):
    assignments = list(arguments.set)
    if arguments.budget is not None:
        assignments.append(f"budget.total_seconds={arguments.budget}")
    if arguments.seed is not None:
        assignments.append(f"seed={arguments.seed}")
    if arguments.backend is not None:
        assignments.append(f"backend={arguments.backend}")
    return load_settings(arguments.settings, assignments)

# runs the pipeline from the parsed arguments
def run_command(arguments):
    settings = run_settings(arguments)
    task = TASK_ALIASES.get(arguments.task, arguments.task)
    options = {"ask": arguments.ask}
    context = run(arguments.data, settings, arguments.target, task, arguments.mode, arguments.variant, arguments.out, options=options)
    return 0 if context is not None else 1

# predicts new rows with a saved model and writes the predictions to a CSV file
def predict_command(arguments):
    model = FittedModel.load(arguments.model)
    predictions = model.predict_frame(arguments.data)
    predictions.to_csv(arguments.out, index=False)
    print(f"Wrote {len(predictions)} predictions to {arguments.out}")
    return 0

# prints report.md of a run folder
def report_command(arguments):
    report_file = Path(arguments.run_dir) / "report.md"
    if not report_file.exists():
        print(f"No report.md in {arguments.run_dir}.")
        return 1
    print(report_file.read_text(encoding="utf-8"))
    return 0

# asks for the inputs of one run with questionary and starts it
def interactive():
    data = questionary.path("Path of your data file:").ask()
    if not data:
        return 1
    target = questionary.text("Target column (empty: let the framer decide):").ask() or None
    variant = questionary.select("Variant:", choices=VARIANTS, default="full").ask()
    if variant is None:
        return 1
    arguments = build_parser().parse_args(["run", data, "--variant", variant] + (["--target", target] if target else []) + ["--ask"])
    return run_command(arguments)

# entry point: runs the chosen command, or the interactive mode without arguments
def main():
    prepare_console()
    parser = build_parser()
    if len(sys.argv) == 1:
        return interactive()
    arguments = parser.parse_args()
    commands = {"run": run_command, "predict": predict_command, "report": report_command}
    if arguments.command not in commands:
        parser.print_help()
        return 1
    try:
        return commands[arguments.command](arguments)
    except StopRun as error:
        print(f"Stopped: {error}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
