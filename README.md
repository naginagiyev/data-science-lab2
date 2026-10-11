# Agentic AutoML

An agentic machine learning automation tool. You give it a training dataset, and it handles the rest — from analysis to a tuned model.

---

## What It Does

This tool runs a full machine learning pipeline with no input from you other than the dataset path:

1. **Detects the target column** and the task type
2. **Analyzes the dataset** with exploratory data analysis (EDA)
3. **Fills the project configuration** (metric, minimum score, etc.) from the EDA report
4. **Plans and runs preprocessing** — cleaning, encoding, scaling
5. **Splits the data** into training and test sets
6. **Trains a model** using an AI-generated training plan
7. **Tunes the model** for better performance

It supports **binary classification**, **multi-class classification** and **regression** tasks.

---

## How to Run

```bash
uv run main.py
```

You will be asked to provide the path to your training dataset. After that, the tool decides everything automatically.

---

## Environment Variables

Create a `.env` file in the project root with the following key:

```
OPENAI_API_KEY=your_openai_key
```

---

## Project Structure

```
agentic-automl/
├── main.py               # Entry point
├── agents/               # AI agents for each pipeline step
├── tools/                # Helper tools (EDA, data split, metrics, etc.)
├── prompts/              # Prompt templates for the agents
├── llms/                 # LLM integration
├── configuration/        # Key project configurations
├── sandbox/              # Workspace for agents and tools
└── models/               # Trained model artifacts
```

### LLM modules (`llms/`)

| File | Role |
|------|------|
| `generation.py` | Planning, reasoning, and structured extraction via the OpenAI Chat API |
| `codex.py` | Code generation and repair |

Both read system prompts from `prompts/` and use API keys from your environment (see **Environment Variables**).

### Agents (`agents/`)

| File | Role |
|------|------|
| `targetdetector.py` | Detects the target column of the dataset |
| `configfiller.py` | Fills the project configuration from the EDA report |
| `planner.py` | Builds preprocessing and training plans |
| `processoragent.py` | Generates and runs preprocessing code (with fix loops) |
| `trainagent.py` | Generates and runs training code (with fix loops) |
| `evaluationagent.py` | Scores the model and tuning loops |

### Tools (`tools/`)

These scripts do **not** call the LLM; they run fixed steps the pipeline depends on.

| File | Role |
|------|------|
| `eda.py` | Exploratory data analysis report |
| `datasplitter.py` | Train / validation / test split |
| `coderunner.py` | Runs generated scripts as subprocesses |
| `datareader.py` | Loads CSV/TSV and basic checks |
| `metrics.py` | Maps metric names to scikit-learn scores |

### Prompts (`prompts/`)

Markdown files used as system or task prompts:

| File | Used for |
|------|----------|
| `targetdetector.md` | Target column detection |
| `configfiller.md` | Filling configuration from EDA |
| `processingplanprompt.md` | Preprocessing plan |
| `trainplanprompt.md` | Training plan |
| `codexcodeprompt.md` | Code generation |
| `codexfixprompt.md` | Code repair |
| `processorreasoningprompt.md` | Preprocessing fix-loop reasoning |
| `trainreasoningprompt.md` | Training fix-loop reasoning |
| `evaluationprompt.md` | Hyperparameter tuning suggestions |

### Other configuration (`configuration/`)

| File | Role |
|------|------|
| `constants.json` | Split ratios, tuning limits, paths |
| `metrics.json` | Allowed metric names per task type |
| `modeloptions.md` | Model shortlist for the planner |
| `edaconfig.py` | EDA settings |

During a run, the active settings for the project are written under `configuration/` (for example `configuration.json` for the run).

---

## Testing the project

1. Install dependencies (see **Requirements**).
2. Set `OPENAI_API_KEY` in the root `.env` file.
3. Run `uv run main.py` and pass a path to a small CSV.
4. Check outputs under `sandbox/` (processed data, scripts, logs, `tuning.json`), `models/` and `configuration/` for the updated JSON.

For prompt or agent changes, edit the matching file in `prompts/` or `agents/`, then repeat a short run to confirm the pipeline still completes.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.10 for you)
- An OpenAI API key

Install dependencies:

```bash
uv sync
```

Add a new dependency with `uv add <package>`, so it is saved in `pyproject.toml` and `uv.lock`.
