import json
from schemas import Plan, Step
from operations.custom import FREE_CLASS
from agents.cleaner import CodeAnswer, columns_view
from tools.sandbox import SandboxError, allowed_imports_text, check_code, run_code

# the task text of the free-code prompt: what to clean, the audit issues and...
# the columns with their values
def instructions(task, issues_text, columns_text):
    return (
        f"Task: prepare the features for predicting `{task.target}` ({task.task}). The target is not in frame.\n"
        "Clean the table: turn numbers written as text (units, currencies, percent, suffixes) into numbers, dates into\n"
        "numeric parts, yes/no spellings into 0/1, spelling variants into one label, missing-value tokens into empty\n"
        "values, and split packed values into numeric columns. Learn anything you need in fit (training rows only).\n"
        "Keep every useful column; never use the target; return numbers as floats and other values as strings.\n"
        f"Audit issues:\n{issues_text}\nColumns:\n{columns_text}"
    )

# strips a markdown code fence from an answer
def plain_code(text):
    return text.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip() + "\n"

# the problems of the generated cleaning code: the AST check, then a trial...
# run (fit on 300 training rows, transform 100 other rows) that keeps the rows
def code_problems(answer, frame):
    code = plain_code(answer.code)
    problems = check_code(code, FREE_CLASS)
    if problems:
        return problems
    sample = frame.astype(str).reset_index(drop=True)
    try:
        run_code(code, FREE_CLASS, sample.head(300), [sample.iloc[300:400].reset_index(drop=True) if len(sample) > 300 else sample], timeout=60)
    except SandboxError as error:
        return [str(error)]
    return []

# asks the LLM for the cleaning class; returns the free-code plan (protected...
# steps first, then one gated freecode step) or None
def free_code_plan(client, context, train, glance, task, protected, issues_text, settings, show_values=True):
    features = train.drop(columns=[task.target])
    inputs = {
        "class_name": FREE_CLASS, "allowed_imports": allowed_imports_text(),
        "task_specific_instructions": instructions(task, issues_text, columns_view(train, glance, task.target, show_values, settings.cleaner.max_values_per_column)),
    }
    answer = client.ask("freecode", inputs, CodeAnswer, "plan", context, check=lambda answer: code_problems(answer, features))
    if answer is None:
        return None
    step = Step(op="freecode", params={"code": plain_code(answer.code)}, reason="LLM-written cleaning code (free-code ablation)", source="llm")
    return Plan(version=1, steps=list(protected.steps) + [step], row_steps=list(protected.row_steps), target_steps=list(protected.target_steps), notes=f"freecode: {len(json.dumps(step.params))} characters of code")
