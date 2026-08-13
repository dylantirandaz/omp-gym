"""The embedded dashboard page: one HTML document, no build step."""

DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>omp-gym</title>
<style>
  :root { --bg: #0d1117; --panel: #161b22; --border: #30363d;
          --text: #e6edf3; --dim: #8b949e; --green: #3fb950;
          --red: #f85149; --blue: #58a6ff; --amber: #d29922; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 -apple-system, "SF Mono", Menlo, monospace; }
  header { padding: 16px 24px; border-bottom: 1px solid var(--border);
           display: flex; align-items: baseline; gap: 16px; }
  header h1 { font-size: 18px; margin: 0; }
  header span { color: var(--dim); font-size: 12px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
         padding: 16px 24px; }
  section { background: var(--panel); border: 1px solid var(--border);
            border-radius: 8px; padding: 16px; }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size: 13px; color: var(--dim); margin: 0 0 12px;
       text-transform: uppercase; letter-spacing: .08em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--dim); font-weight: normal;
       padding: 4px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
  .pass { color: var(--green); } .fail { color: var(--red); }
  .dim { color: var(--dim); }
  .episode { cursor: pointer; } .episode:hover td { background: #1c2129; }
  #transcript { max-height: 480px; overflow-y: auto; }
  .step { margin: 8px 0; padding: 8px 12px; border-radius: 6px;
          white-space: pre-wrap; word-break: break-word; font-size: 12px; }
  .step.user { background: #1c2b3a; }
  .step.assistant { background: #1d2b1f; }
  .step.tool { background: #2b2617; color: var(--dim); }
  .step.tool.error { background: #3a1d1d; }
  .tc { color: var(--blue); font-size: 12px; margin-top: 6px; }
  .tc code { color: var(--amber); }
  #empty { color: var(--dim); padding: 24px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>omp-gym</h1>
  <span id="meta">loading…</span>
</header>
<main>
  <section class="wide"><h2>Models</h2><div id="models"></div></section>
  <section><h2>Adapters</h2><div id="adapters"></div></section>
  <section><h2>Timeline</h2><div id="timeline"></div></section>
  <section class="wide"><h2>Episodes</h2><div id="episodes"></div></section>
  <section class="wide"><h2>Transcript</h2>
    <div id="transcript"><div id="empty">select an episode</div></div>
  </section>
</main>
<script>
const fmt = {
  money: v => v == null ? "-" : "$" + v.toFixed(4),
  pct: (p, r) => Math.round(100 * p / r) + "% (" + p + "/" + r + ")",
  int: v => v == null ? "-" : Math.round(v).toLocaleString(),
};

function table(headers, rows) {
  if (!rows.length) return '<div class="dim">nothing recorded yet</div>';
  return "<table><tr>" + headers.map(h => "<th>" + h + "</th>").join("") +
    "</tr>" + rows.join("") + "</table>";
}

async function load() {
  const summary = await (await fetch("/api/summary")).json();
  document.getElementById("meta").textContent =
    summary.entries + " ledger entries" +
    (summary.torn ? " · " + summary.torn + " torn" : "");

  document.getElementById("models").innerHTML = table(
    ["model", "pass rate", "cost/pass", "tokens/solve", "mean tokens"],
    summary.models.map(m =>
      `<tr><td>${m.model}</td>` +
      `<td class="${m.passes ? "pass" : "fail"}">` +
      `${fmt.pct(m.passes, m.runs)}</td>` +
      `<td>${fmt.money(m.cost_per_pass)}</td>` +
      `<td>${fmt.int(m.tokens_per_solve)}</td>` +
      `<td class="dim">${fmt.int(m.mean_tokens)}</td></tr>`));

  document.getElementById("adapters").innerHTML = table(
    ["adapter", "method", "train loss", "val loss", "when"],
    summary.adapters.map(a =>
      `<tr><td>${a.adapter}</td><td>${a.method ?? "sft"}</td>` +
      `<td>${a.metrics.first_train_loss} → ${a.metrics.last_train_loss}</td>` +
      `<td>${a.metrics.first_val_loss == null ? "-" :
        a.metrics.first_val_loss + " → " + a.metrics.last_val_loss}</td>` +
      `<td class="dim">${a.timestamp.slice(0, 16)}</td></tr>`));

  const timeline = await (await fetch("/api/timeline")).json();
  document.getElementById("timeline").innerHTML = table(
    ["when", "kind", "what"],
    timeline.reverse().map(e =>
      `<tr><td class="dim">${e.timestamp.slice(0, 16)}</td>` +
      `<td>${e.kind}</td><td class="dim">` +
      `${(e.config.task ?? e.config.adapter ??
         (e.config.models ?? []).join(",") ?? "").slice(0, 48)}</td></tr>`));

  const episodes = await (await fetch("/api/episodes")).json();
  document.getElementById("episodes").innerHTML = table(
    ["episode", "task", "model", "reward", "seconds"],
    episodes.map(e =>
      `<tr class="episode" onclick="showTranscript('${e.episode}')">` +
      `<td>${e.episode}</td><td>${e.task}</td>` +
      `<td class="dim">${e.model.split("/").pop()}</td>` +
      `<td class="${e.reward >= 1 ? "pass" : "fail"}">${e.reward}</td>` +
      `<td class="dim">${e.duration_seconds}</td></tr>`));
}

async function showTranscript(name) {
  const steps = await (await fetch(
    "/api/transcript?episode=" + encodeURIComponent(name))).json();
  const el = document.getElementById("transcript");
  el.innerHTML = steps.map(s => {
    if (s.role === "user")
      return `<div class="step user">${esc(s.text)}</div>`;
    if (s.role === "assistant") {
      const calls = (s.tool_calls ?? []).map(c =>
        `<div class="tc">→ <code>${c.name}</code> ` +
        `${esc(JSON.stringify(c.arguments)).slice(0, 140)}</div>`).join("");
      return `<div class="step assistant">${esc(s.text ?? "")}${calls}</div>`;
    }
    return `<div class="step tool${s.is_error ? " error" : ""}">` +
      `[${s.tool_name}] ${esc(s.text ?? "")}</div>`;
  }).join("") || '<div id="empty">empty transcript</div>';
  el.scrollIntoView({behavior: "smooth"});
}

function esc(s) {
  return String(s).replace(/[&<>"]/g,
    c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

load();
setInterval(load, 15000);
</script>
</body>
</html>
"""
