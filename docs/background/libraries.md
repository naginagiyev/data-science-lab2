# Libraries & Tooling Review

A review of the Python libraries and packages used in DS Lab 1, and what to keep, change, add, or remove for DS Lab 2 / the MSc thesis. Companion to `overview.md`; issue codes like **B4** or **E2** refer to that file.

---

## Summary

The **core data-science stack is good**. The problems are in **how dependencies are managed**, the **LLM layer**, the choice of **ONNX as the only model format**, and **what's missing for thesis-level experiments**.

The project code itself imports only about 15 third-party packages:

> pandas, numpy, scikit-learn, scipy, openai, pydantic, onnxruntime, paramiko, fastapi, rich, questionary, click, tabulate, python-dotenv, joblib

Everything else in `requirements.txt` is either used only by the **LLM-generated scripts** (catboost, skl2onnx, onnxmltools, cloudpickle) or **not used at all**.

---

## Table of Contents

- [1. Keep](#1-keep)
- [2. Change](#2-change)
  - [2.1 Python version and dependency management](#21-python-version-and-dependency-management)
  - [2.2 The LLM layer](#22-the-llm-layer)
  - [2.3 ONNX as the only model format](#23-onnx-as-the-only-model-format)
  - [2.4 Execution sandbox and deployment](#24-execution-sandbox-and-deployment)
  - [2.5 Smaller points](#25-smaller-points)
- [3. Add (missing for a thesis)](#3-add-missing-for-a-thesis)
- [4. Remove](#4-remove)
- [5. Proposed dependency groups](#5-proposed-dependency-groups)
- [6. Priority order](#6-priority-order)

---

## 1. Keep

| Library | Verdict |
|---|---|
| **pandas, numpy, scipy, scikit-learn** | Standard and correct. The scikit-learn `Pipeline` is the right way to package preprocessing. |
| **pydantic** | Good. Use it more: also for the run config (see E2), not only for LLM outputs. |
| **FastAPI + uvicorn** | Good choice for serving. |
| **rich + questionary** | Fine for the command-line interface. |
| **CatBoost** | Good for tabular data. Add LightGBM next to it (see [section 3](#3-add-missing-for-a-thesis)). |
| **tabulate** | Fine for the EDA Markdown tables. |

---

## 2. Change

### 2.1 Python version and dependency management

**Problem.**
- **Python 3.10 reaches end-of-life in October 2026**, the month after this review.
- `numpy==1.26` is outdated; numpy 2.x is the current line.
- Dependencies are declared in **three places that don't agree**:
  - `pyproject.toml` lists no dependencies.
  - `uv.lock` is effectively empty.
  - `requirements.txt` is a UTF-16 `pip freeze` that includes transitive packages.
- There is no separation between what the pipeline needs, what generated code may import, what the API needs, and development tools.

**Fix.**
- Move to **Python 3.12**.
- Use **uv** properly: declare direct dependencies in `pyproject.toml`, generate a real `uv.lock`, and delete the frozen `requirements.txt`.
- Split dependencies into groups (see [section 5](#5-proposed-dependency-groups)). Keep the `runtime` group in sync with the libraries the prompts allow generated code to use.

### 2.2 The LLM layer

**Current:** the raw `openai` SDK, with the model name `gpt-4.1-mini` hardcoded in two places. The two wrappers are also inconsistent: `llms/codex.py` calls `client.beta.chat.completions.parse` while `llms/generation.py` calls `client.chat.completions.parse`.

**Problem.**
- The system is locked to one provider. For a thesis, *"does the agent still work with a cheaper or different LLM?"* is a valuable experiment.
- API errors aren't retried, and calls aren't logged, so cost and tokens are untracked (C3).

**Fix.**
- Put the LLM behind **your own small interface**, implemented with either:
  - **LiteLLM**: one interface for OpenAI, Anthropic, Gemini, and local models; or
  - **PydanticAI**: typed agents that work naturally with the existing Pydantic schemas.
- Add **tenacity** for retries with exponential backoff.
- Log **every call**: prompt, response, tokens, latency, and cost. The simplest option is JSONL files in the run folder; **Langfuse** or **MLflow tracing** add a UI on top.
- **Avoid LangChain.** It's heavy and mostly hides what the agent does. The thesis should show exactly that.

### 2.3 ONNX as the only model format

This is the most costly choice in the project.

**Problem.** Requiring ONNX export for every model causes several existing problems:
- **ARIMA and Prophet can't be exported**, so those task types can't work (A2).
- ONNX `outputs[0]` holds labels, which broke the probability metrics (B4).
- `skl2onnx` often fails on custom transformers, which wastes fix-loop attempts.
- There are **two separate artifacts**: a preprocessor saved with `cloudpickle` but loaded with `joblib`, plus the ONNX model.

**Fix.**
- Save **one scikit-learn `Pipeline` (preprocessor + model)** as a **single artifact**, using either:
  - **skops**: a safer alternative to pickle for scikit-learn objects; or
  - the **MLflow model** format, which also covers serving.
- Keep ONNX only as an **optional export step** at the end, if you want to show it.

### 2.4 Execution sandbox and deployment

**Problem.**
- Generated code runs through `subprocess` with no isolation (D2).
- Deployment with **paramiko + ngrok** is hand-rolled, insecure, and tied to one VM (D1, D3).

**Fix.**
- Run generated code with the **docker** Python SDK, in a throwaway container per script with no secrets and with CPU, memory, and time limits. Alternatively, use a hosted sandbox such as **E2B**.
- Replace paramiko + ngrok with a **Dockerfile** for the API. Then deploy anywhere (`docker run`, Hugging Face Spaces, Fly.io, Render), or leave deployment out of scope. Deployment isn't the research contribution, so keep it simple.

### 2.5 Smaller points

| Current | Issue | Suggestion |
|---|---|---|
| **click** | Used only as a bare `@click.command()` with no options, so it currently does nothing. | Add real flags (`--auto`, `--config`, `--dataset`), or switch to **Typer**, the more modern option. |
| **python-dotenv** | Works, but settings are untyped and scattered. | **pydantic-settings**: typed, validated settings (API keys, model name, limits) in one place. |
| **cloudpickle + joblib** | Saved with one library, loaded with the other. | Replaced by the single-artifact approach in [section 2.3](#23-onnx-as-the-only-model-format). |
| **Time-series stack** (statsmodels, ONNX) | ARIMA/Prophet don't fit the ONNX design. | If time series stays in scope, use **statsforecast** or **sktime**. Recommended instead: drop it and focus on tabular data. |

---

## 3. Add (missing for a thesis)

| Library | Why |
|---|---|
| **Optuna** | Real hyperparameter optimisation, and the baseline against LLM-guided tuning (B5). |
| **LightGBM** (optionally **XGBoost**) | Most strong tabular baselines use it. |
| **AutoGluon** and/or **FLAML** | AutoML baselines to compare the agent against (A1). |
| **MLflow** | Experiment tracking: one entry per run with parameters, metrics, artifacts, and LLM cost. Solves most of E2. |
| **openml** | Loads benchmark tasks (OpenML-CC18 / AMLB) with their official splits. |
| **tenacity** | Retries with backoff for LLM API calls. |
| **pandera** *(optional)* | Declarative data checks. A natural fit for the deterministic checks that should replace the LLM critic (C1). |
| **pytest, ruff, pyright / mypy, pre-commit** | Tests, linting and formatting, type checking, and automatic checks before each commit (E4, E6). |

---

## 4. Remove

These packages are in `requirements.txt` but nothing uses them:

| Package | Reason |
|---|---|
| `tf2onnx` | Useless without TensorFlow, and TensorFlow isn't installed. |
| `plotly` | Not imported anywhere. |
| `graphviz` | Not imported anywhere. |
| `sympy` | Not imported anywhere. |
| `jupyter_client`, `nbformat` | Not imported anywhere. |
| All pinned **transitive** packages | Let `uv.lock` handle them instead of listing them by hand. |

---

## 5. Proposed dependency groups

An illustrative structure for `pyproject.toml`:

| Group | Contents |
|---|---|
| **core** | pandas, numpy, scipy, scikit-learn, pydantic, pydantic-settings, litellm *or* pydantic-ai, tenacity, rich, questionary, typer, tabulate, mlflow, optuna, openml |
| **runtime** (what generated code may import) | scikit-learn, lightgbm, catboost, (xgboost), skops |
| **baselines** | autogluon *or* flaml |
| **serve** | fastapi, uvicorn, skops *(or the MLflow model runtime)* |
| **sandbox** | docker |
| **dev** | pytest, ruff, pyright *or* mypy, pre-commit |

---

## 6. Priority order

1. **Move to uv, Python 3.12, and a clean `pyproject.toml`.** Remove unused packages.
2. **Save one scikit-learn Pipeline instead of ONNX.** This also fixes B4 and part of A2.
3. **Add Optuna, LightGBM, MLflow, and openml.** They are needed for the thesis experiments.
4. **Put LiteLLM or PydanticAI behind a small interface of your own**, and log every LLM call.
5. **Replace paramiko + ngrok with Docker**, and run generated code in a container sandbox.
