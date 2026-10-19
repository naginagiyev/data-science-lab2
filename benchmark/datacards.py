import sys
import json
from edatool import audit
from benchmark.leaks import leak_table
from paths import docs_dir, results_dir
from benchmark.corruptor import EXPECTED_OPS, TYPE_NUMBERS, corrupt, write_layout
from benchmark.datasets import GROUP_A, GROUP_C, LOCAL_DATASETS, MANIFEST, OPENML_DATASETS, dataset_spec, load_dataset

CARDS_FILE = docs_dir / "datasets.md"
TASK_IDS = {
    "houses": 363678, "Is-this-a-good-customer": 363682, "credit-g": 363626, "Fitness_Club": 363671, "Marketing_Campaign": 363684,
    "Bank_Customer_Churn": 363619, "E-CommereShippingData": 363632, "online_shoppers_intention": 363691, "maternal_health_risk": 363685,
    "students_dropout_and_academic_success": 363704, "SDSS17": 363699, "concrete_compressive_strength": 363625,
    "healthcare_insurance_expenses": 363675, "airfoil_self_noise": 363612, "Another-Dataset-on-used-Fiat-500": 363615,
    "miami_housing": 363686, "diamonds": 363631,
}
POOL_NAMES = {"dev": "Dev pool", "A": "Group A (clean base; Group B adds the corruptor)", "C": "Group C (natural mess)", "spare-A": "Spares for Group A", "spare-C": "Spares for Group C"}
NOTES = {
    "ibmchurn": ("local dev file (IBM Telco churn)", "blank TotalCharges for new customers, customerID"),
    "kinghousedata": ("local dev file (King County house sales)", "date as text, yr_renovated = 0 means never, repeated id (resales)"),
    "winequality": ("local dev file (red wine only)", "240 duplicate rows; quality 3-8, class 3 has 10 rows"),
    "creditcardfraud": ("local dev file; anonymized 'no harm' control", "V1-V28 anonymized; 0.17% fraud; fixed 50,000-row sample"),
    "houses": ("TabArena dev dataset (California housing)", "clean; log target"),
    "Is-this-a-good-customer": ("TabArena dev dataset", "clean"),
    "credit-g": ("small classic credit data with text categories", "clean in this version"),
    "Fitness_Club": ("small binary data with readable names", "clean in this version"),
    "Marketing_Campaign": ("customer data with dates and categories", "Dt_Customer is a date as text"),
    "Bank_Customer_Churn": ("medium binary churn data", "clean"),
    "E-CommereShippingData": ("medium binary data with units in names (Weight_in_gms)", "clean"),
    "online_shoppers_intention": ("medium binary web-session data", "clean"),
    "maternal_health_risk": ("small multi-class medical data", "clean; duplicates possible"),
    "students_dropout_and_academic_success": ("wide multi-class data (37 features)", "many coded categories"),
    "SDSS17": ("large multi-class physics data", "single-letter band names (u g r i z) are standard but short; fixed 30,000-row stratified sample"),
    "concrete_compressive_strength": ("small regression with physical amounts", "clean"),
    "healthcare_insurance_expenses": ("small regression with categories", "clean"),
    "airfoil_self_noise": ("small physical regression", "clean"),
    "Another-Dataset-on-used-Fiat-500": ("small regression on used cars", "clean"),
    "miami_housing": ("medium housing regression; licence is non-commercial (research use only)", "clean"),
    "diamonds": ("large regression; fixed 30,000-row sample", "x, y, z are single-letter dimension names"),
    "HR_Analytics_Job_Change_of_Data_Scientists": ("natural mess (expected)", "experience '>20', '<1'; company size ranges; last_new_job 'never', '>4'"),
    "in_vehicle_coupon_recommendation": ("natural mess (expected)", "income ranges as text; age '50plus', 'below21'; time '10AM'; expiration '1d', '2h'"),
    "heloc": ("natural mess (expected)", "special values -7, -8, -9"),
    "bank-marketing": ("natural mess (expected)", "'unknown' tokens; pdays = -1; the duration leak is not in this version"),
    "Food_Delivery_Time": ("checked for Group C in P6.5 and replaced: the TabArena version is cleaned", "only trailing spaces in categories; fixed 30,000-row sample"),
    "employee_salaries": ("natural mess; replaces Food_Delivery_Time in Group C (P6.5)", "dates as mm/dd/yyyy text; a mostly empty job-title column; many job titles"),
}

# one line per dataset for the overview table
def overview_row(name, entry):
    spec = OPENML_DATASETS[name]
    rows = f"{entry['rows']:,}" + (f" → {spec['sample_rows']:,}" if spec.get("sample_rows") else "")
    return f"| {name} | {spec['pool']} | {'P' if spec.get('pilot') else ''} | {entry['task']} | `{entry['target']}` | {rows} | {entry['columns']} | {entry['licence']} |"

# the data card of one OpenML dataset
def card(name, entry):
    why, mess = NOTES.get(name, ("spare", "-"))
    tid = TASK_IDS.get(name)
    return [
        f"### {name}", "",
        f"- Source: OpenML did {entry['did']}" + (f", TabArena task {tid}" if tid else "") + f", version {entry['version']} ({entry['url']}); OpenML name `{entry['openml_name']}`.",
        f"- Licence: {entry['licence']}.",
        f"- Size: {entry['rows']:,} rows, {entry['columns']} columns (target included)" + (f"; fixed sample of {entry['sample_rows']:,} rows" if entry.get("sample_rows") else "") + ".",
        f"- Task: {entry['task']}, target `{entry['target']}`" + (f", {entry['classes']} classes" if entry.get("classes") else "") + ".",
        f"- Readable feature names: {entry['readable_share']:.0%}. Table hash `{entry['table_hash']}`.",
        f"- Why: {why}.", f"- Known mess: {mess}.", "",
    ]

# writes docs/datasets.md from the manifest, the corruptor examples, the leak...
# table and the natural-mess check
def write_cards():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lines = [
        "# Datasets", "",
        "Generated by `uv run python -m benchmark.datacards` from `benchmark/results/datasets.json` (written by",
        "`python -m benchmark.datasets`, which downloads every dataset into `cache/openml/`). Pools: Appendix D of",
        "`buildplan.md`. Every licence allows research use; miami_housing (CC BY-NC-SA) is non-commercial only.", "",
        "## Overview", "",
        "| Dataset | Pool | Pilot | Task | Target | Rows | Columns | Licence |", "|---|---|---|---|---|---|---|---|",
    ]
    for name, spec in LOCAL_DATASETS.items():
        why, mess = NOTES[name]
        lines.append(f"| {name} | dev (local) | | {spec['task']} | `{spec['target']}` | local file | | see source |")
    lines += [overview_row(name, entry) for name, entry in manifest.items()]
    lines += ["", "## Local dev files", ""]
    for name, spec in LOCAL_DATASETS.items():
        why, mess = NOTES[name]
        lines += [f"### {name}", "", f"- Source: `data/{spec['file']}` (git-ignored). Task {spec['task']}, target `{spec['target']}`.", f"- Why: {why}.", f"- Known mess: {mess}.", ""]
    for pool, title in POOL_NAMES.items():
        names = [name for name in manifest if manifest[name]["pool"] == pool]
        if names:
            lines += [f"## {title}", ""]
            for name in names:
                lines += card(name, manifest[name])
    for title, text in [("Corruptor examples", mess_examples()), ("Leak columns (Group D)", leak_table()), ("Natural mess", natural_mess())]:
        lines += [f"## {title}", "", text.strip(), ""]
    CARDS_FILE.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")

# one realistic example per mess type (level 3, one type at a time) from the...
# first Group A dataset where the type applies
def mess_examples():
    lines = ["Level 3, one type at a time, seed 0; three changed cells per type (before → after).", ""]
    for kind in [kind for kind in TYPE_NUMBERS if kind not in {"layout", "leak"}]:
        for name in GROUP_A:
            spec = dataset_spec(name)
            clean = load_dataset(name).head(3000)
            messy, truth, cells = corrupt(clean, spec["target"], name, 0, 3, [kind])
            if not truth["items"]:
                continue
            item = truth["items"][0]
            lines.append(f"- **{TYPE_NUMBERS[kind]}. {kind}** ({name}, expected `{EXPECTED_OPS[kind]}`):")
            if kind == "ids":
                lines.append(f"  added columns {item['params']['added']}, e.g. `{messy[item['params']['added'][0]].iloc[0]}`")
            else:
                sample = cells[cells["original"] != cells["corrupted"]].head(3)
                pairs = "; ".join(f"`{row.column}`: `{row.original}` → `{row.corrupted}`" for row in sample.itertuples())
                lines.append(f"  {pairs}")
            break
    name = "credit-g"
    spec = dataset_spec(name)
    sample_path = results_dir / "layout-example.csv"
    write_layout(load_dataset(name).head(3), sample_path, 3)
    head = sample_path.read_bytes().decode("cp1252").splitlines()[:5]
    sample_path.unlink()
    lines += ["- **10. layout** (credit-g, level 3: title rows, `;`, decimal commas, cp1252, footer; reader test E6b): first lines:", "", "```", *[line[:110] for line in head], "```"]
    return "\n".join(lines)

# the columns that show the natural mess of each Group C dataset
NATURAL_COLUMNS = {
    "HR_Analytics_Job_Change_of_Data_Scientists": ["experience", "company_size", "last_new_job"],
    "in_vehicle_coupon_recommendation": ["income", "age", "expiration", "time"],
    "heloc": ["ExternalRiskEstimate", "MSinceMostRecentDelq"],
    "bank-marketing": ["contact", "poutcome", "pdays"],
    "employee_salaries": ["date_first_hired", "underfilled_job_title"],
    "Food_Delivery_Time": ["Type_of_order", "Type_of_vehicle", "Delivery_person_Ratings"],
}

# the natural-mess check of P6.5: top values of the messy columns and the...
# deterministic audit findings (B, C and D codes) of every Group C dataset and...
# of the replaced Food_Delivery_Time
def natural_mess():
    lines = ["Checked on the downloaded versions (deterministic audit of the first 10,000 rows).", ""]
    for name in GROUP_C + ["Food_Delivery_Time"]:
        spec = dataset_spec(name)
        raw = load_dataset(name)
        report = audit(raw.head(10000), spec["target"], spec["task"], use_agent=False)
        codes = {}
        for issue in report.issues:
            if issue.code[0] in "BCD":
                codes[issue.code] = codes.get(issue.code, 0) + 1
        status = "replaced by employee_salaries (the TabArena version is cleaned)" if name == "Food_Delivery_Time" else ("pilot" if OPENML_DATASETS[name].get("pilot") else "Group C")
        lines.append(f"- **{name}** ({status}); audit findings: " + ", ".join(f"{code} x{count}" for code, count in sorted(codes.items())))
        for column in NATURAL_COLUMNS.get(name, []):
            if column in raw:
                top = raw[column].value_counts().head(5)
                lines.append(f"  - `{column}`: " + ", ".join(f"`{value}` ({count})" for value, count in top.items()))
    return "\n".join(lines)

if __name__ == "__main__":
    write_cards()
    sys.exit(0)
