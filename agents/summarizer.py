import re
from pydantic import BaseModel

MAX_WORDS = 120
MAX_FACTS = 6000

# the summary answer
class SummaryAnswer(BaseModel):
    text: str

# the numbers of a text without thousands commas ("1,234.5" -> "1234.5")
def numbers(text):
    return {value.replace(",", "").rstrip(".") for value in re.findall(r"\d[\d,]*\.?\d*", text)}

# the problems of a summary: too long, or numbers that are not in the facts...
# (small counts up to 10 are allowed)
def summary_problems(answer, facts):
    problems = []
    words = len(answer.text.split())
    if words > MAX_WORDS:
        problems.append(f"the summary has {words} words; use at most {MAX_WORDS}")
    known = numbers(facts)
    unknown = sorted(value for value in numbers(answer.text) - known if not (value.isdigit() and int(value) <= 10))
    if unknown:
        problems.append(f"these numbers are not in the facts: {', '.join(unknown)}; use only numbers from the facts")
    return problems

# asks the LLM for the summary paragraph of a report (the report text is the...
# facts); returns the text or None when the call fails
def summarize(client, context, report_text):
    facts = report_text[:MAX_FACTS]
    answer = client.ask("summary", {"facts": facts}, SummaryAnswer, "report", context, check=lambda answer: summary_problems(answer, facts))
    return answer.text.strip() if answer is not None else None
