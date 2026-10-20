"""Render preregistered local tasks and check fresh DOM outside the model policy.

Only fixture definitions enter HTML. Goals and acceptance values stay in the runner.
No function here invokes a model, browser, API, or external resource.
"""

import html
import json


def render_case(case):
    config = case["fixture"]
    escape = html.escape
    kind = config["kind"]
    controls = ""
    initial = config.get("initial", {"field": "", "mode": "", "checked": False})
    if kind in {"form", "repair", "unknown"}:
        controls += (
            f'<label>{escape(config["field_label"])} '
            f'<input id="field" type="text" required value="{escape(initial["field"], quote=True)}"></label>'
        )
    if kind in {"form", "repair"}:
        options = "".join(
            f'<option value="{escape(value, quote=True)}"'
            f'{" selected" if value == initial["mode"] else ""}>{escape(value)}</option>'
            for value in config["options"]
        )
        controls += f'<label>{escape(config["mode_label"])} <select id="mode">{options}</select></label>'
        checked = " checked" if initial["checked"] else ""
        controls += f'<label><input id="checkbox" type="checkbox"{checked}>{escape(config["checkbox_label"])}</label>'
    if kind == "unsupported":
        controls += f'<label>{escape(config["field_label"])} <input id="file" type="file" required></label>'
    disabled = " disabled" if kind == "unsupported" else ""
    # Config is UI-only, with no task goals, expected settings, checker rules, or gold action.
    config_json = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c")
    case_id_json = json.dumps(case["id"])
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(config['heading'])}</title>
<style>body{{max-width:700px;margin:40px auto;font:20px system-ui}}label{{display:block;margin:18px 0}}
input,select,button{{font:inherit}}button{{padding:12px;margin:12px 12px 0 0}}#summary{{white-space:pre-line}}</style>
<h1>{escape(config['heading'])}</h1>
<p id="status" role="status">{escape(config['before'])}</p>
<form id="form">{controls}<button id="primary" type="submit"{disabled}>{escape(config['primary'])}</button>
<button id="reset" type="button">{escape(config['reset'])}</button></form>
<p id="summary" data-applied="null" data-submissions="0" data-interactions="0"></p>
<script>
const config={config_json}, caseId={case_id_json};
document.body.dataset.caseId=caseId;
const summary=document.getElementById('summary'), status=document.getElementById('status');
const field=document.getElementById('field'), mode=document.getElementById('mode');
const checkbox=document.getElementById('checkbox'), file=document.getElementById('file');
const primary=document.getElementById('primary');
const current=()=>({{field:field?.value??null,mode:mode?.value??null,checked:checkbox?.checked??null}});
const interaction=()=>summary.dataset.interactions=String(Number(summary.dataset.interactions)+1);
const show=(values)=>{{
  summary.dataset.applied=JSON.stringify(values);
  summary.textContent=values ? [config.field_label+': '+values.field,config.mode_label+': '+values.mode,
    config.checkbox_label+': '+(values.checked?'enabled':'disabled')].join('\\n') : '';
}};
if(config.kind==='repair')show(current());
for(const control of [field,mode,checkbox,file].filter(Boolean)){{
  control.addEventListener('input',()=>{{interaction();status.textContent='Changes not applied';}});
  if(control===file)control.addEventListener('change',()=>primary.disabled=!file.files.length);
}}
document.getElementById('form').onsubmit=event=>{{
  event.preventDefault();interaction();summary.dataset.submissions=String(Number(summary.dataset.submissions)+1);
  status.textContent=config.after;
  if(config.kind==='completion')primary.disabled=true;
  else if(config.kind==='form'||config.kind==='repair')show(current());
  else summary.dataset.applied=JSON.stringify({{field:field?.value??null,file_count:file?.files.length??0}});
}};
document.getElementById('reset').onclick=()=>{{
  interaction();status.textContent=config.before;primary.disabled=config.kind==='unsupported';
  if(field)field.value=config.initial?.field??'';
  if(mode)mode.value=config.initial.mode;
  if(checkbox)checkbox.checked=config.initial.checked;
  if(file)file.value='';
  if(config.kind==='repair')show(current());else show(null);
}};
</script></html>"""


# Generic inspection only. No case-specific acceptance data or success value is evaluated in the browser.
READ_FRESH_DOM = """(() => {
  const summary=document.getElementById('summary');
  return {
    case_id:document.body.dataset.caseId,
    status:document.getElementById('status')?.textContent,
    field:document.getElementById('field')?.value??null,
    mode:document.getElementById('mode')?.value??null,
    checked:document.getElementById('checkbox')?.checked??null,
    file_count:document.getElementById('file')?.files.length??0,
    primary_disabled:document.getElementById('primary')?.disabled??null,
    applied:JSON.parse(summary?.dataset.applied??'null'),
    submissions:Number(summary?.dataset.submissions??-1),
    semantic_interactions:Number(summary?.dataset.interactions??-1)
  };
})()"""


def verify_case(case, observed, *, agent_status, terminal_choice=None, executed_actions=None, execution_error=None):
    """Independent acceptance; call only after the policy stops, never feed output back."""
    expected = case["acceptance"]
    checks = {
        "correct_fixture": observed.get("case_id") == case["id"],
        "agent_terminal_status": agent_status == expected["agent_status"],
        "explicit_model_terminal_choice": terminal_choice == expected["agent_status"].upper(),
        "negative_no_browser_actions": expected["agent_status"] != "blocked" or executed_actions == 0,
        "no_execution_error": execution_error is None,
    }
    for key, value in expected.items():
        if key == "agent_status":
            continue
        if key == "minimum_submissions":
            checks[key] = type(observed.get("submissions")) is int and observed["submissions"] >= value
        elif key == "applied_settings_match":
            applied = observed.get("applied")
            checks[key] = isinstance(applied, dict) and all(
                applied.get(name) == expected[name] for name in ("field", "mode", "checked")
            )
        else:
            checks[key] = observed.get(key) == value
    return {"verified": all(checks.values()), "checks": checks, "fresh_dom": observed}
