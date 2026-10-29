import json
from html import escape
from tools.traceview import step_params

STYLE = """
:root{--bg:#f7f7f9;--card:#fff;--text:#1d1f24;--dim:#6b7280;--line:#e5e7eb;--accent:#0e7490;--llm:#a21caf;--ok:#15803d;--bad:#b91c1c;--warn:#b45309;--code:#f3f4f6}
:root[data-theme=dark]{--bg:#111318;--card:#1a1d24;--text:#e5e7eb;--dim:#9ca3af;--line:#2b303b;--accent:#22d3ee;--llm:#e879f9;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--code:#0d0f14}
 @media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111318;--card:#1a1d24;--text:#e5e7eb;--dim:#9ca3af;--line:#2b303b;--accent:#22d3ee;--llm:#e879f9;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--code:#0d0f14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}.dim{color:var(--dim)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}.grid div b{display:block;font-size:16px}
.timeline{display:flex;height:22px;border-radius:6px;overflow:hidden;margin-top:10px;border:1px solid var(--line)}
.timeline a{display:block;background:var(--accent);opacity:.75;border-right:1px solid var(--card);color:#fff;font-size:11px;overflow:hidden;white-space:nowrap;padding:2px 4px;text-decoration:none}
.timeline a.failed{background:var(--bad)}.timeline a:hover{opacity:1}
.toolbar{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.toolbar input{flex:1;min-width:180px;padding:7px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text)}
button{border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:999px;padding:5px 11px;cursor:pointer}button.on{border-color:var(--accent);color:var(--accent)}
h2{font-size:16px;margin:0;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}h2 .num{color:var(--accent);font-variant-numeric:tabular-nums}h2 .sec{margin-left:auto;color:var(--dim);font-weight:400}
.event{border-top:1px solid var(--line);padding:7px 0}.event:first-of-type{border-top:0}
summary{cursor:pointer}.badge{display:inline-block;font-size:11px;padding:0 7px;border-radius:999px;border:1px solid currentColor;margin-left:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.llm{color:var(--llm)}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;max-height:420px;white-space:pre-wrap;word-break:break-word;font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--dim);font-weight:600}
.reason{border-left:3px solid var(--llm);padding:4px 10px;margin:6px 0;font-style:italic}.hidden{display:none}
 @media (max-width:600px){main{padding:10px 16px}h2 .sec{margin-left:0}}
"""

SCRIPT = """
const q=document.getElementById('q');let kind='all';
function apply(){const t=q.value.toLowerCase();document.querySelectorAll('.event').forEach(e=>{const okKind=kind==='all'||e.dataset.kind===kind;const okText=!t||e.textContent.toLowerCase().includes(t);e.classList.toggle('hidden',!(okKind&&okText));});}
q.addEventListener('input',apply);
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{kind=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('on',x===b));apply();}));
document.getElementById('expand').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
document.getElementById('collapse').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=false));
document.getElementById('theme').addEventListener('click',()=>{const r=document.documentElement;const dark=r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);r.dataset.theme=dark?'light':'dark';try{localStorage.setItem('trace-theme',r.dataset.theme)}catch(e){}});
try{const s=localStorage.getItem('trace-theme');if(s)document.documentElement.dataset.theme=s}catch(e){}
"""

# escapes any value for HTML (dicts and lists as pretty JSON)
def text(value):
    return escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str))

# an HTML table from column names and rows
def table(columns, rows):
    head = "".join(f"<th>{text(name)}</th>" for name in columns)
    body = "".join("<tr>" + "".join(f"<td>{text(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"

# the answer of an LLM call: plan steps as a table, the reasons the model wrote...
# highlighted, and the full JSON
def answer_html(answer):
    try:
        data = json.loads(answer or "")
    except (ValueError, TypeError):
        return f"<pre>{text(answer or '')}</pre>"
    parts = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and "op" in value[0]:
                params = [step_params(step) for step in value]
                rows = [[step.get("op", ""), ", ".join(map(str, param.get("columns") or [param.get("column", "")])), json.dumps(param, ensure_ascii=False), step.get("reason", "")] for step, param in zip(value, params)]
                parts.append(f"<b>{text(key)}</b>: {len(value)} steps" + table(["operation", "columns", "parameters", "reason (written by the LLM)"], rows))
            elif "reason" in key.lower() or key.lower() in ("rationale", "explanation", "notes", "summary", "why"):
                parts.append(f"<div class='reason'><b>reasoning ({text(key)}):</b> {text(value)}</div>")
    return "".join(parts) + f"<details><summary class='dim'>full answer (JSON)</summary><pre>{text(data)}</pre></details>"

# one LLM call: header, prompt and answer (collapsed), errors of an invalid answer
def llm_html(record):
    state = "cached" if record.get("cached") else ("ok" if record.get("ok") else "invalid" if record.get("answer") else "failed")
    badge = {"ok": "ok", "cached": "dim", "invalid": "bad", "failed": "bad"}[state]
    prompts = [message["content"] for message in record.get("messages") or [] if message.get("role") == "user"]
    retry = f"<span class='badge warn'>retry {record['attempt']}</span>" if record.get("attempt") else ""
    head = (f"<span class='llm'>◆ LLM</span> <b>{text(record.get('prompt', '?'))}</b> <span class='dim'>{text(record.get('model', ''))} · "
            f"{record.get('input_tokens', 0):,} in / {record.get('output_tokens', 0):,} out · ${record.get('cost_usd', 0.0):.4f} · {record.get('seconds', 0.0):.1f} s</span>"
            f"<span class='badge {badge}'>{state}</span>{retry}")
    error = f"<pre class='bad'>{text(record['error'])}</pre>" if record.get("error") else ""
    prompt = f"<details><summary>prompt ({len(prompts[-1].splitlines()) if prompts else 0} lines)</summary><pre>{text(prompts[-1] if prompts else '')}</pre></details>"
    return f"<details><summary>{head}</summary>{prompt}<h4>answer</h4>{answer_html(record.get('answer'))}{error}</details>"

# one event as HTML (kind decides the filter button it belongs to)
def event_html(event):
    data, kind = event["data"], event["type"]
    if kind == "llm_call":
        body, group = llm_html(data), "llm"
    elif kind == "decision":
        icon, style = ("✓", "ok") if data.get("accepted") else ("✗", "bad")
        extra = {key: data[key] for key in ("evidence", "proposal") if data.get(key)}
        more = f"<details><summary class='dim'>evidence and proposal</summary><pre>{text(extra)}</pre></details>" if extra else ""
        body = f"<span class='{style}'>{icon}</span> <span class='dim'>{text(data.get('kind'))}</span> <b>{text(data.get('subject'))}</b> {text(data.get('reason', ''))}<span class='badge'>{text(data.get('source', ''))}</span>{more}"
        group = "decision"
    elif kind == "detail":
        inner = table(data["table"]["columns"], data["table"]["rows"]) if data.get("table") else ""
        inner += "".join(f"<div>{text(str(line))}</div>" for line in data.get("lines", []))
        body, group = f"<details open><summary><b>▸ {text(event['title'])}</b></summary>{inner}</details>", "detail"
    elif kind == "warning":
        body, group = f"<span class='warn'>⚑ {text(event['title'])}</span>", "warning"
    elif kind == "error":
        body = f"<span class='bad'>✗ {text(data.get('error', event['title']))}</span><details><summary class='dim'>traceback</summary><pre>{text(data.get('traceback', ''))}</pre></details>"
        group = "error"
    else:
        return ""
    return f"<div class='event' data-kind='{group}'>{body}</div>"

# the summary card from the run_start and run_end events
def summary_html(start, end):
    data, result = start.get("data", {}), end.get("data", {}) if end else {}
    usage = result.get("usage", {})
    cells = [("outcome", result.get("outcome", "running")), ("time", f"{result.get('seconds', 0):.1f} s"), ("CV score", result.get("cv") or "n/a"), ("test score", result.get("test") or "n/a"),
             ("LLM calls", usage.get("calls", 0)), ("LLM cost", f"${usage.get('cost_usd', 0.0):.4f}"), ("variant", data.get("variant", "")), ("LLM model", data.get("model") or "none")]
    grid = "".join(f"<div><span class='dim'>{label}</span><b>{text(str(value))}</b></div>" for label, value in cells)
    return f"<div class='card'><h1>Agentic AutoML v2 · {text(data.get('data', ''))}</h1><div class='dim'>{text(data.get('run_id', ''))}</div><div class='grid' style='margin-top:10px'>{grid}</div>"

# writes trace.html: summary, stage timeline, toolbar and one card per stage
def write_html(events, path):
    start = next((event for event in events if event["type"] == "run_start"), {"data": {}})
    end = next((event for event in reversed(events) if event["type"] == "run_end"), None)
    stages, order = {}, []
    for event in events:
        if event["type"] == "stage_start":
            order.append(event["title"])
            stages[event["title"]] = {"start": event, "end": None, "events": []}
        elif event["type"] == "stage_end" and event["stage"] in stages:
            stages[event["stage"]]["end"] = event
        elif event["stage"] in stages:
            stages[event["stage"]]["events"].append(event)
    total = sum((stages[name]["end"] or {"data": {"seconds": 0}})["data"].get("seconds", 0) for name in order) or 1
    bar, cards = "", ""
    for name in order:
        stage = stages[name]
        end_data = (stage["end"] or {"data": {}})["data"]
        seconds, ok = end_data.get("seconds", 0), end_data.get("ok", True)
        number = stage["start"]["data"].get("number", 0)
        bar += f"<a href='#stage-{name}' class='{'' if ok else 'failed'}' style='flex:{max(seconds / total, 0.02):.3f}' title='{name} {seconds:.1f} s'>{number:02d} {name}</a>"
        note = text(end_data.get("note") or "") if ok else f"<span class='bad'>{text(end_data.get('error') or '')} → {text(end_data.get('fallback') or 'no fallback')}</span>"
        body = "".join(event_html(event) for event in stage["events"]) or "<div class='dim'>no events</div>"
        cards += (f"<section class='card' id='stage-{name}'><h2><span class='num'>{number:02d}</span>{name} <span class='{'ok' if ok else 'bad'}'>{'✓' if ok else '✗'}</span> <span class='dim'>{note}</span>"
                  f"<span class='sec'>{seconds:.1f} s</span></h2><div class='dim'>{text(stage['start']['data'].get('text', ''))}</div>{body}</section>")
    filters = "".join(f"<button data-filter='{key}' class='{'on' if key == 'all' else ''}'>{label}</button>" for key, label in [("all", "all"), ("llm", "LLM calls"), ("decision", "decisions"), ("detail", "details"), ("warning", "warnings"), ("error", "errors")])
    toolbar = f"<div class='toolbar'><input id='q' placeholder='search prompts, answers, decisions...'>{filters}<button id='expand'>expand all</button><button id='collapse'>collapse all</button><button id='theme'>light / dark</button></div>"
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Run trace</title><style>{STYLE}</style></head>"
            f"<body><main>{summary_html(start, end)}<div class='timeline'>{bar}</div></div>{toolbar}{cards}</main><script>{SCRIPT}</script></body></html>")
    path.write_text(page, encoding="utf-8", newline="\n")
    return path
