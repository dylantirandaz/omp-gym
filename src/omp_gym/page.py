"""The embedded dashboard page: one HTML document, no build step."""

DASHBOARD_PAGE = r"""<!DOCTYPE html>
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
  .in { background: #0d1117; border: 1px solid var(--border);
        color: var(--text); padding: 8px; border-radius: 6px;
        font: inherit; }
  .btn { background: #1c2b3a; border: 1px solid var(--border);
         color: var(--text); padding: 8px 16px; border-radius: 6px;
         cursor: pointer; font: inherit; }
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
  td.cell { text-align: center; }
  .c-pass { background: #12331c; color: var(--green); }
  .c-part { background: #332b12; color: var(--amber); }
  .c-fail { background: #3a1d1d; color: var(--red); }
  .c-err { background: #21262d; color: var(--dim); }
  th.hold { color: var(--amber); }
  tr.diverge td { background: #2b2617; }
  .tok { padding: 1px 2px; border-radius: 3px; white-space: pre-wrap; }
  .chip { cursor: pointer; border: 1px solid var(--border);
          border-radius: 10px; padding: 2px 8px; font-size: 12px;
          color: var(--blue); display: inline-block; margin: 2px; }
  .gen { background: #0d1117; border: 1px solid var(--border);
         border-radius: 6px; padding: 8px; font-size: 12px;
         white-space: pre-wrap; word-break: break-word;
         max-height: 220px; overflow-y: auto; margin: 4px 0 0; }
  .replaypane { border: 1px solid var(--border); border-radius: 8px;
                padding: 12px; }
  .filetab { cursor: pointer; border: 1px solid var(--border);
             border-radius: 6px; padding: 2px 8px; font-size: 12px;
             display: inline-block; margin: 2px; color: var(--dim); }
  .filetab.active { color: var(--text); background: #1c2b3a; }
  input[type=range] { width: 100%; }
</style>
</head>
<body>
<header>
  <h1>omp-gym</h1>
  <span id="meta">loading…</span>
</header>
<main>
  <section class="wide"><h2>Models</h2><div id="models"></div></section>
  <section class="wide"><h2>Run monitor</h2><div id="monitor">
    <div class="dim">loading…</div></div></section>
  <section class="wide"><h2>Bench matrix</h2><div id="matrix">
    <div class="dim">loading…</div></div></section>
  <section><h2>Adapters</h2><div id="adapters"></div></section>
  <section><h2>Timeline</h2><div id="timeline"></div></section>
  <section class="wide"><h2>Episodes</h2><div id="episodes"></div></section>
  <section><h2>Failure modes</h2><div id="failures"></div></section>
  <section><h2>Interpretability (preview)</h2><div id="interp"></div></section>
  <section class="wide"><h2>Lens diff</h2>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
      <input id="lensprompt" class="in" style="flex:2;min-width:240px"
             placeholder="prompt · Enter runs the lens"
             onkeydown="if(event.key==='Enter')runLens()">
      <input id="lensadapter" class="in" style="flex:1;min-width:180px"
             value="adapters/v15-varied-runtime"
             placeholder="adapter dir · empty = base only">
      <input id="lensmodel" class="in" style="flex:1;min-width:180px"
             placeholder="base model · empty = default">
      <button class="btn" onclick="runLens()">run</button>
    </div>
    <div id="lensout"><div class="dim">the lens shows the top predicted
      next token after every decoder layer; with an adapter dir it shows
      base and adapter side by side and marks layers that diverge</div></div>
  </section>
  <section class="wide"><h2>SAE explorer</h2>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
      <input id="saeprompt" class="in" style="flex:2;min-width:240px"
             placeholder="prompt · Enter shows feature activations"
             onkeydown="if(event.key==='Enter')runSae()">
      <button class="btn" onclick="runSae()">activations</button>
      <input id="saefeature" class="in" style="width:110px"
             placeholder="feature id">
      <input type="range" id="saealpha" min="-4" max="4" step="0.5"
             value="2" style="width:160px;align-self:center"
             oninput="document.getElementById('alphaval').textContent=this.value">
      <span style="align-self:center">alpha
        <strong id="alphaval">2</strong></span>
      <button class="btn" onclick="runSteer()">steer</button>
    </div>
    <div id="saeout"><div class="dim">token heat shows the strongest SAE
      feature per token; pick a feature chip, set alpha, then steer</div></div>
    <div id="steerout"></div>
  </section>
  <section class="wide"><h2>Training</h2><div id="training"></div></section>
  <section class="wide"><h2>Replay</h2>
    <div class="dim" id="replayhint">select an episode below to scrub
      through it step by step</div>
    <div id="replaywrap" style="display:none">
      <div style="margin-bottom:8px">compare with:
        <select id="compare" class="in" onchange="pickCompare(this.value)">
        </select></div>
      <div style="display:flex;gap:16px;align-items:flex-start">
        <div class="replaypane" style="flex:1;min-width:0" id="replay-a"></div>
        <div class="replaypane" style="flex:1;min-width:0;display:none"
             id="replay-b"></div>
      </div>
    </div>
  </section>
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
let EPISODES = [];
const REPLAY = { a: null, b: null };

function table(headers, rows) {
  if (!rows.length) return '<div class="dim">nothing recorded yet</div>';
  return "<table><tr>" + headers.map(h => "<th>" + h + "</th>").join("") +
    "</tr>" + rows.join("") + "</table>";
}

function esc(s) {
  return String(s).replace(/[&<>"]/g,
    c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

function jsq(s) {
  return String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'")
    .replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r");
}

function sparklineSVG(series, color, w, h) {
  if (!series || series.length < 2) return "";
  w = w || 220; h = h || 48;
  const min = Math.min(...series), max = Math.max(...series);
  const range = max - min || 1;
  const pts = series.map((v, i) => {
    const x = (i / (series.length - 1)) * (w - 4) + 2;
    const y = h - 4 - ((v - min) / range) * (h - 8);
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
    `<polyline points="${pts}" fill="none" stroke="${color}" ` +
    `stroke-width="1.5"/></svg>`;
}

async function load() {
  const summary = await (await fetch("/api/summary")).json();
  document.getElementById("meta").textContent =
    summary.entries + " ledger entries" +
    (summary.torn ? " · " + summary.torn + " torn" : "");

  document.getElementById("models").innerHTML = table(
    ["model", "pass rate", "cost/pass", "tokens/solve", "mean tokens"],
    summary.models.map(m =>
      `<tr><td>${esc(m.model)}</td>` +
      `<td class="${m.passes ? "pass" : "fail"}">` +
      `${esc(fmt.pct(m.passes, m.runs))}</td>` +
      `<td>${fmt.money(m.cost_per_pass)}</td>` +
      `<td>${fmt.int(m.tokens_per_solve)}</td>` +
      `<td class="dim">${fmt.int(m.mean_tokens)}</td></tr>`));

  document.getElementById("adapters").innerHTML = table(
    ["adapter", "method", "train loss", "val loss", "when"],
    summary.adapters.map(a =>
      `<tr><td>${esc(a.adapter)}</td><td>${esc(a.method ?? "sft")}</td>` +
      `<td>${esc(a.metrics.first_train_loss)} → ` +
      `${esc(a.metrics.last_train_loss)}</td>` +
      `<td>${a.metrics.first_val_loss == null ? "-" :
        esc(a.metrics.first_val_loss) + " → " +
        esc(a.metrics.last_val_loss)}</td>` +
      `<td class="dim">${esc(a.timestamp.slice(0, 16))}</td></tr>`));

  const timeline = await (await fetch("/api/timeline")).json();
  document.getElementById("timeline").innerHTML = table(
    ["when", "kind", "what"],
    timeline.reverse().map(e =>
      `<tr><td class="dim">${esc(e.timestamp.slice(0, 16))}</td>` +
      `<td>${esc(e.kind)}</td><td class="dim">` +
      `${esc((e.config.task ?? e.config.adapter ??
         (e.config.models ?? []).join(",") ?? "").slice(0, 48))}</td></tr>`));

  EPISODES = await (await fetch("/api/episodes")).json();
  document.getElementById("episodes").innerHTML = table(
    ["episode", "task", "model", "reward", "seconds"],
    EPISODES.map(e =>
      `<tr class="episode" ` +
      `onclick="showTranscript('${esc(jsq(e.episode))}')">` +
      `<td>${esc(e.episode)}</td><td>${esc(e.task)}</td>` +
      `<td class="dim">${esc(e.model.split("/").pop())}</td>` +
      `<td class="${e.reward >= 1 ? "pass" : "fail"}">` +
      `${esc(e.reward)}</td>` +
      `<td class="dim">${esc(e.duration_seconds)}</td></tr>`));
}

async function loadMonitor() {
  const data = await (await fetch("/api/monitor")).json();
  let html = "";
  const live = data.live;
  if (live && live.losses.length) {
    const lastIter = live.iterations[live.iterations.length - 1];
    const lastLoss = live.losses[live.losses.length - 1];
    const fresh = live.age_seconds < 120;
    const val = live.validations.length
      ? live.validations[live.validations.length - 1] : null;
    html += `<div style="margin-bottom:12px"><strong>live</strong> ` +
      `<span class="${fresh ? "pass" : "dim"}">` +
      `${fresh ? "running" : "stale"} · ` +
      `${Math.round(live.age_seconds)}s since last write</span><br>` +
      `<span class="dim">iter ${esc(lastIter)} · loss ${esc(lastLoss)}` +
      (val ? ` · val ${esc(val.loss)} @ ${esc(val.iteration)}` : "") +
      (live.tokens_per_second != null
        ? ` · ${esc(live.tokens_per_second)} tok/s` : "") +
      (live.peak_memory_gb != null
        ? ` · ${esc(live.peak_memory_gb)} GB peak` : "") +
      `</span><br>` + sparklineSVG(live.losses, "#58a6ff", 440, 56) +
      `</div>`;
  } else {
    html += `<div class="dim" style="margin-bottom:12px">no live log · ` +
      `mirror a trainer log to runs/live-train.log</div>`;
  }
  if (data.reports.length) {
    html += table(
      ["adapter", "model", "iters", "train loss", "val loss", "device",
       "curve", "when"],
      data.reports.map(r =>
        `<tr><td>${esc((r.adapter_dir ?? "?").split("/").pop())}</td>` +
        `<td class="dim">${esc((r.model ?? "").split("/").pop())}</td>` +
        `<td>${esc(r.iterations ?? "-")}</td>` +
        `<td>${esc(r.first_train_loss)} → ${esc(r.last_train_loss)}</td>` +
        `<td class="dim">${r.first_val_loss == null ? "-" :
          esc(r.first_val_loss) + " → " + esc(r.last_val_loss)}</td>` +
        `<td class="dim">${esc(r.device_name ?? "-")}</td>` +
        `<td>${sparklineSVG(r.train_series, "#3fb950", 120, 28)}</td>` +
        `<td class="dim">${new Date(r.modified * 1000)
          .toISOString().slice(0, 16)}</td></tr>`));
  }
  document.getElementById("monitor").innerHTML =
    html || '<div class="dim">no training runs recorded</div>';
}

async function loadMatrix() {
  const data = await (await fetch("/api/matrix")).json();
  const el = document.getElementById("matrix");
  if (!data.rows.length) {
    el.innerHTML = '<div class="dim">no bench rows recorded yet</div>';
    return;
  }
  const holdout = new Set(data.holdout);
  let html = "<table><tr><th>run</th><th>when</th>" +
    data.tasks.map(t =>
      `<th class="${holdout.has(t) ? "hold" : ""}">${esc(t)}</th>`
    ).join("") + "</tr>";
  html += data.rows.map(r => {
    const cells = data.tasks.map(t => {
      const c = r.cells[t];
      if (!c) return "<td></td>";
      let cls = "c-err", text = "E";
      if (c.trials) {
        text = c.passes + "/" + c.trials;
        cls = c.passes === c.trials ? "c-pass"
          : c.passes ? "c-part" : "c-fail";
      }
      return `<td class="cell ${cls}">${esc(text)}</td>`;
    }).join("");
    return `<tr><td>${esc(r.label)}</td>` +
      `<td class="dim">${esc(String(r.when)).slice(0, 16)}</td>${cells}</tr>`;
  }).join("") + "</table>";
  html += `<div class="dim" style="margin-top:8px">amber headers are ` +
    `sealed holdout tasks · E = provider error only</div>`;
  el.innerHTML = html;
}

async function showTranscript(name) {
  const data = await (await fetch(
    "/api/transcript?episode=" + encodeURIComponent(name))).json();
  const el = document.getElementById("transcript");
  el.innerHTML = (data.steps ?? []).map(renderStep).join("") ||
    '<div id="empty">empty transcript</div>';
  initReplay("a", name, data);
  fillCompare(name);
  document.getElementById("replaywrap").scrollIntoView({behavior: "smooth"});
}

function renderStep(s) {
  if (!s) return "";
  if (s.role === "user")
    return `<div class="step user">${esc(s.text)}</div>`;
  if (s.role === "assistant") {
    const calls = (s.tool_calls ?? []).map(c =>
      `<div class="tc">→ <code>${esc(c.name)}</code> ` +
      `${esc(JSON.stringify(c.arguments)).slice(0, 400)}</div>`).join("");
    return `<div class="step assistant">${esc(s.text ?? "")}${calls}</div>`;
  }
  return `<div class="step tool${s.is_error ? " error" : ""}">` +
    `[${esc(s.tool_name)}] ${esc(s.text ?? "")}</div>`;
}

function initReplay(pane, name, data) {
  if (!data.steps || !data.steps.length) return;
  REPLAY[pane] = { name: name, record: data.record ?? {},
    steps: data.steps, idx: data.steps.length - 1, file: null };
  document.getElementById("replayhint").style.display = "none";
  document.getElementById("replaywrap").style.display = "block";
  document.getElementById("replay-" + pane).style.display = "block";
  renderReplay(pane);
}

function fillCompare(name) {
  const me = EPISODES.find(e => e.episode === name);
  const options = ['<option value="">none</option>'].concat(
    EPISODES.filter(e => me && e.task === me.task && e.episode !== name)
      .map(e => `<option value="${esc(e.episode)}">` +
        `${esc(e.episode)} · ${esc(e.model.split("/").pop())} · ` +
        `reward ${esc(e.reward)}</option>`));
  document.getElementById("compare").innerHTML = options.join("");
}

async function pickCompare(value) {
  if (!value) {
    REPLAY.b = null;
    document.getElementById("replay-b").style.display = "none";
    return;
  }
  const data = await (await fetch(
    "/api/transcript?episode=" + encodeURIComponent(value))).json();
  initReplay("b", value, data);
}

function scrub(pane, value) {
  REPLAY[pane].idx = Number(value);
  renderReplay(pane);
}

function pickFile(pane, index) {
  const state = REPLAY[pane];
  state.file = index;
  renderReplay(pane);
}

function replayFiles(state) {
  const files = {};
  for (let i = 0; i <= state.idx; i++) {
    const step = state.steps[i];
    for (const call of step.tool_calls ?? []) {
      const a = call.arguments ?? {};
      if (call.name === "write" && typeof a.path === "string" &&
          typeof a.content === "string") files[a.path] = a.content;
    }
  }
  return files;
}

function renderReplay(pane) {
  const state = REPLAY[pane];
  if (!state) return;
  const record = state.record ?? {};
  const last = state.idx === state.steps.length - 1;
  const files = replayFiles(state);
  const names = Object.keys(files);
  if (state.file == null || state.file >= names.length)
    state.file = names.length ? names.length - 1 : null;
  const tabs = names.map((n, i) =>
    `<span class="filetab${i === state.file ? " active" : ""}" ` +
    `onclick="pickFile('${pane}', ${i})">${esc(n)}</span>`).join("");
  const body = state.file != null
    ? `<pre class="gen">${esc(files[names[state.file]].slice(0, 4000))}</pre>`
    : '<div class="dim">no full-file writes up to this step</div>';
  const test = last && record.test_output
    ? `<div class="dim" style="margin-top:8px">test output</div>` +
      `<pre class="gen">${esc(record.test_output)}</pre>` : "";
  document.getElementById("replay-" + pane).innerHTML =
    `<div style="margin-bottom:6px"><strong>${esc(state.name)}</strong> ` +
    `<span class="${record.reward >= 1 ? "pass" : "fail"}">` +
    `reward ${esc(record.reward ?? "?")}</span> ` +
    `<span class="dim">${esc((record.model ?? "").split("/").pop())} · ` +
    `${esc(record.duration_seconds ?? "?")}s</span></div>` +
    `<input type="range" min="0" max="${state.steps.length - 1}" ` +
    `value="${state.idx}" oninput="scrub('${pane}', this.value)">` +
    `<div class="dim">step ${state.idx + 1} / ${state.steps.length}</div>` +
    renderStep(state.steps[state.idx]) +
    `<div style="margin-top:6px">${tabs}</div>` + body + test;
}

async function loadFailures() {
  const data = await (await fetch("/api/clusters")).json();
  const clusters = data.clusters ?? {};
  document.getElementById("failures").innerHTML = table(
    ["mode", "count"],
    Object.entries(clusters).map(([mode, c]) =>
      `<tr><td>${esc(mode)}</td><td>${esc(c.count)}</td></tr>`));
}

async function loadInterp() {
  const data = await (await fetch("/api/inspect")).json();
  let html = "";
  if (data.lens) {
    html += "<p class='dim'>logit lens · " + esc(data.lens.model) +
      " · " + esc(data.lens.layers) + " layers</p>";
    html += table(["layer", "top predictions"],
      data.lens.top_by_layer.map((tokens, i) =>
        `<tr><td>${i}</td><td class="dim">` +
        tokens.map(esc).join(" · ") + "</td></tr>"));
  }
  if (data.sae) {
    html += "<p class='dim'>SAE · layer " + esc(data.sae.layer) +
      " · loss " + data.sae.loss_first.toFixed(3) + " → " +
      data.sae.loss_last.toFixed(3) + "</p>";
    html += table(["feature", "rate", "fires on"],
      data.sae.report.slice(0, 12).map(f =>
        `<tr><td>${esc(f.feature)}</td><td>${esc(f.activity_rate)}</td>` +
        `<td class="dim">` +
        `${esc((f.top_samples[0] ?? {}).excerpt ?? "")}</td></tr>`));
  }
  document.getElementById("interp").innerHTML =
    html || '<div class="dim">run omp-gym inspect / sae first</div>';
}

async function runLens() {
  const prompt = document.getElementById("lensprompt").value.trim();
  if (!prompt) return;
  const adapter = document.getElementById("lensadapter").value.trim();
  const model = document.getElementById("lensmodel").value.trim();
  const out = document.getElementById("lensout");
  out.innerHTML =
    '<div class="dim">loading model and computing… (first run is slow)</div>';
  const modelParam = model ? "&model=" + encodeURIComponent(model) : "";
  if (!adapter) {
    const res = await (await fetch("/api/lens?prompt=" +
      encodeURIComponent(prompt) + modelParam)).json();
    if (res.error) {
      out.innerHTML = '<div class="fail">' + esc(res.error) + '</div>';
      return;
    }
    out.innerHTML = table(["layer", "top predictions"],
      res.top_by_layer.map((tokens, i) =>
        `<tr><td>${i}</td><td class="dim">` +
        tokens.map(esc).join(" · ") + "</td></tr>"));
    return;
  }
  const res = await (await fetch("/api/lensdiff?prompt=" +
    encodeURIComponent(prompt) + "&adapter=" +
    encodeURIComponent(adapter) + modelParam)).json();
  if (res.error) {
    out.innerHTML = '<div class="fail">' + esc(res.error) + '</div>';
    return;
  }
  const count = res.diverges.filter(Boolean).length;
  out.innerHTML = `<div class="dim" style="margin-bottom:8px">` +
    `${esc(res.model)} · ${esc(res.adapter)} · ` +
    `${count} of ${res.diverges.length} layers diverge</div>` +
    table(["layer", "base top", "adapter top"],
      res.base.map((tokens, i) =>
        `<tr class="${res.diverges[i] ? "diverge" : ""}"><td>${i}</td>` +
        `<td class="dim">${tokens.map(esc).join(" · ")}</td>` +
        `<td class="dim">${res.adapter_top[i].map(esc).join(" · ")}` +
        `</td></tr>`));
}

async function runSae() {
  const prompt = document.getElementById("saeprompt").value.trim();
  if (!prompt) return;
  const out = document.getElementById("saeout");
  out.innerHTML =
    '<div class="dim">loading model and computing… (first run is slow)</div>';
  const res = await (await fetch("/api/sae/tokens?prompt=" +
    encodeURIComponent(prompt))).json();
  if (res.error) {
    out.innerHTML = '<div class="fail">' + esc(res.error) + '</div>';
    return;
  }
  const max = Math.max(
    ...res.tokens.map(t => (t.features[0] ?? {}).activation ?? 0), 0.001);
  const spans = res.tokens.map(t => {
    const act = (t.features[0] ?? {}).activation ?? 0;
    const title = t.features.map(f => "#" + f.id + ":" + f.activation)
      .join(" ");
    return `<span class="tok" style="background:rgba(88,166,255,` +
      `${(0.75 * act / max).toFixed(2)})" title="${esc(title)}">` +
      `${esc(t.text)}</span>`;
  }).join("");
  const chips = res.top_features.map(f =>
    `<span class="chip" onclick="pickFeature(${Number(f.id)})">` +
    `#${esc(f.id)} · ${esc(f.activation)}</span>`).join("");
  out.innerHTML = `<div class="dim">model ${esc(res.model)}` +
    (res.adapter ? ` + ${esc(res.adapter)}` : "") +
    ` · layer ${esc(res.layer)}</div>` +
    `<div style="margin:8px 0;line-height:2">${spans}</div>` +
    `<div>top features: ${chips}</div>`;
}

function pickFeature(id) {
  document.getElementById("saefeature").value = id;
}

async function runSteer() {
  const prompt = document.getElementById("saeprompt").value.trim();
  const feature = document.getElementById("saefeature").value.trim();
  if (!prompt || feature === "") return;
  const alpha = document.getElementById("saealpha").value;
  const out = document.getElementById("steerout");
  out.innerHTML = '<div class="dim" style="margin-top:8px">steering…</div>';
  const res = await (await fetch("/api/sae/steer?prompt=" +
    encodeURIComponent(prompt) + "&feature=" +
    encodeURIComponent(feature) + "&alpha=" +
    encodeURIComponent(alpha))).json();
  if (res.error) {
    out.innerHTML = '<div class="fail">' + esc(res.error) + '</div>';
    return;
  }
  out.innerHTML = `<div style="display:flex;gap:12px;margin-top:8px">` +
    `<div style="flex:1;min-width:0"><div class="dim">alpha 0</div>` +
    `<pre class="gen">${esc(res.unsteered)}</pre></div>` +
    `<div style="flex:1;min-width:0"><div class="dim">feature #` +
    `${esc(res.feature)} · alpha ${esc(res.alpha)}</div>` +
    `<pre class="gen">${esc(res.steered)}</pre></div></div>`;
}

async function loadTraining() {
  const data = await (await fetch("/api/training")).json();
  let html = "";
  if (data.train && data.train.length) {
    html += data.train.map(t => {
      const curve = t.series && t.series.length > 1
        ? sparklineSVG(t.series, "#3fb950") : "";
      return `<div style="margin-bottom:14px">` +
        `<strong>${esc(t.adapter)}</strong>` +
        ` <span class="dim">${esc(t.method)} · ${esc(t.iters)} iters · ` +
        `${t.first.toFixed(3)} → ${t.last.toFixed(4)}` +
        (t.first_val != null ? ` · val ${t.first_val.toFixed(3)} → ` +
          `${t.last_val.toFixed(4)}` : "") +
        `</span><br>${curve}</div>`;
    }).join("");
  }
  if (data.rl && data.rl.length) {
    html += data.rl.map(r =>
      `<div><strong>${esc(r.adapter)}</strong> <span class="dim">rl · ` +
      `mean reward ${esc(r.mean_reward_first)} → ` +
      `${esc(r.mean_reward_last)}</span></div>`
    ).join("");
  }
  document.getElementById("training").innerHTML =
    html || '<div class="dim">no training runs recorded</div>';
}

load();
loadFailures();
loadInterp();
loadTraining();
loadMonitor();
loadMatrix();
setInterval(load, 15000);
setInterval(loadFailures, 30000);
setInterval(loadMonitor, 5000);
setInterval(loadMatrix, 30000);
</script>
</body>
</html>
"""
