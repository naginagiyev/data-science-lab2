import sys

# do not let the pycaches to appear
sys.dont_write_bytecode = True

import time
import click
import logging
import questionary
import pandas as pd
from pathlib import Path
from tools.eda import EDA
from rich.text import Text
from rich.panel import Panel
from paths import sandbox_dir
from rich.console import Console
from agents.planner import Planner
from agents.trainagent import TrainAgent
from agents.configfiller import ConfigFiller
from agents.processoragent import ProcessorAgent
from agents.targetdetector import TargetDetector
from tools.datasplitter import split as split_data
from agents.evaluationagent import EvaluationAgent

console = Console()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def run_eda(dataset_path: str, target_col: str, task_type: str):
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    eda = EDA(input_path=dataset_path, target_col=target_col, task_type=task_type)
    stem = Path(dataset_path).stem
    eda.output_path = str(sandbox_dir / f"{stem}eda.md")
    eda.run()
    return eda.output_path

def infer_task_type_from_target(dataset_path: str, target_col: str) -> str:
    df = pd.read_csv(dataset_path)
    target = df[target_col].dropna()
    unique_count = int(target.nunique(dropna=True))
    total_count = int(len(target))
    unique_ratio = (unique_count / total_count) if total_count else 0.0
    numeric_target = pd.to_numeric(target, errors="coerce")
    numeric_ratio = float(numeric_target.notna().mean()) if total_count else 0.0

    if unique_count == 2:
        return "binary-classification"
    if numeric_ratio > 0.95 and unique_count > 20 and unique_ratio > 0.05:
        return "regression"
    return "multi-class-classification"

@click.command()
def main():
    start = time.time()
    text = Text()
    text.append("🤖  Agentic Machine Learning Automation", style="bold cyan")
    console.print(Panel(text, border_style="cyan", padding=(1, 4)))
    console.print()

    while True:
        dataset_path = questionary.path("📂  Path to your dataset:", only_directories=False).ask()

        if dataset_path is None:
            console.print("[yellow]  ⚠️  Cancelled.[/yellow]")
            return

        if not Path(dataset_path).exists():
            console.print("[red]  ❌  I don't see a dataset at that path. Are you sure you entered it right?[/red]\n")
            continue
        break

    console.print(f"[green]  ✅  Dataset exists in[/green] [dim]{dataset_path}[/dim]\n")

    detected_target = TargetDetector().detect(dataset_path)
    inferred_task_type = infer_task_type_from_target(dataset_path, detected_target)

    run_eda(dataset_path, detected_target, inferred_task_type)

    ConfigFiller().fill(dataset_path, detected_target, inferred_task_type)
    console.print()
    console.print(
        Panel(
            "[bold green]✅  Configuration generated![/bold green]\n[dim]Configuration generated from EDA and saved to ./configuration/configuration.json[/dim]",
            border_style="green", padding=(1, 4))
    )
    console.print()
    console.print("[bold magenta]  📋  Generating processing plan...[/bold magenta]")
    planner = Planner()
    preprocessing_path = planner.create_pre_processing_plan()
    console.print()
    console.print(
        Panel(
            f"[bold green]✅  Processing plan created![/bold green]\n[dim]Processing plan saved to {preprocessing_path}[/dim]",
            border_style="green", padding=(1, 4))
    )
    console.print()
    console.print("[bold magenta]  ⚙️  Running preprocessing...[/bold magenta]")
    ProcessorAgent().preprocess()
    console.print()
    console.print("[bold magenta]  ✂️  Splitting data...[/bold magenta]")
    split_data()
    console.print()
    console.print("[bold magenta]  📋  Generating training plan...[/bold magenta]")
    training_path = planner.create_training_plan()
    console.print()
    console.print(
        Panel(
            f"[bold green]✅  Training plan created![/bold green]\n[dim]Training plan saved to {training_path}[/dim]",
            border_style="green", padding=(1, 4))
    )
    console.print()
    console.print("[bold magenta]  🧠  Training model...[/bold magenta]")
    TrainAgent().train()
    console.print()
    console.print("[bold magenta]  🎛️  Tuning model parameters...[/bold magenta]")
    EvaluationAgent().tune()
    elapsed = int(time.time() - start)
    console.print()
    console.print(
        Panel(
            f"[bold green]✅  All done![/bold green]\n[dim]Completed in {elapsed} seconds[/dim]",
            border_style="green", padding=(1, 4))
    )

if __name__ == "__main__":
    main()
