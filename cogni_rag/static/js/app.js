/**
 * CogniRAG Agent Frontend Controller
 * Manages chat interactions, step-by-step reasoning traces, ingestion, benchmarks, and failure diagnostics.
 */

document.addEventListener("DOMContentLoaded", () => {
  fetchHealthAndStats();
  fetchFailureModes();
});

function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".view-content").forEach(v => {
    v.classList.remove("active");
    v.style.display = "none";
  });

  const btn = document.getElementById(`tab-${tab}-btn`);
  const view = document.getElementById(`${tab}-view`);
  if (btn) btn.classList.add("active");
  if (view) {
    view.classList.add("active");
    view.style.display = "block";
  }
}

function setQuery(text) {
  const inp = document.getElementById("chat-input");
  if (inp) {
    inp.value = text;
    inp.focus();
  }
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const inputEl = document.getElementById("chat-input");
  const query = inputEl.value.trim();
  if (!query) return;

  const guardrails = document.getElementById("chk-guardrails").checked;

  // Append user message
  appendMessage("user", query);
  inputEl.value = "";

  const sendBtn = document.getElementById("btn-chat-send");
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.innerText = "Reasoning... ⏳";
  }

  try {
    const res = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, enable_guardrails: guardrails })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    appendAgentResponse(data);
    renderCitations(data.citations);
  } catch (err) {
    console.error("Chat error:", err);
    appendMessage("agent", `⚠️ Request failed: ${err.message}`);
  } finally {
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.innerText = "Execute ReAct Loop ⚡";
    }
  }
}

function appendMessage(sender, text) {
  const container = document.getElementById("chat-history");
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg msg-${sender}`;

  const avatar = sender === "user" ? "👤" : "🧠";
  msgDiv.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body"><p>${text}</p></div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function appendAgentResponse(data) {
  const container = document.getElementById("chat-history");
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-msg msg-agent";

  // Build Reasoning Trace accordion
  let traceHtml = "";
  if (data.reasoning_trace && data.reasoning_trace.length > 0) {
    const stepItems = data.reasoning_trace.map(s => {
      const act = s.action ? `<div class="step-action">Action: ${s.action.tool_name}(${JSON.stringify(s.action.tool_input)})</div>` : "";
      const obs = s.observation ? `<div class="step-obs">Observation: ${typeof s.observation.output === 'object' ? JSON.stringify(s.observation.output, null, 2) : s.observation.output}</div>` : "";
      return `
        <div class="step-card">
          <div class="step-thought">Thought [Step ${s.step_num}]: ${s.thought}</div>
          ${act}
          ${obs}
        </div>
      `;
    }).join("");

    traceHtml = `
      <details class="trace-accordion">
        <summary>View ReAct Reasoning Trace (${data.reasoning_trace.length} steps • ${data.total_latency_ms}ms)</summary>
        <div class="trace-content">${stepItems}</div>
      </details>
    `;
  }

  // Grounding Badge
  let badgeHtml = "";
  if (data.is_grounded) {
    badgeHtml = `<span class="ground-badge badge-grounded">✓ Grounded in Context (${(data.faithfulness_score * 100).toFixed(0)}%)</span>`;
  } else {
    badgeHtml = `<span class="ground-badge badge-ungrounded">⚠️ Grounding Alert (${(data.faithfulness_score * 100).toFixed(0)}%)</span>`;
  }

  // Tools used list
  let toolsHtml = "";
  if (data.tools_used && data.tools_used.length > 0) {
    toolsHtml = `<div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">Tools invoked: <strong>${data.tools_used.join(", ")}</strong></div>`;
  }

  msgDiv.innerHTML = `
    <div class="msg-avatar">🧠</div>
    <div class="msg-body">
      ${traceHtml}
      <div class="answer-text">${data.answer.replace(/\n/g, '<br>')}</div>
      ${toolsHtml}
      ${badgeHtml}
    </div>
  `;

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function renderCitations(citations) {
  const container = document.getElementById("citations-container");
  if (!citations || citations.length === 0) {
    container.innerHTML = `<div class="text-dim text-center" style="padding: 20px;">No search executed for last query.</div>`;
    return;
  }

  container.innerHTML = citations.map(c => `
    <div class="citation-card">
      <div class="citation-head">
        <span>${c.chunk.doc_title}</span>
        <span class="citation-score">${(c.score * 100).toFixed(1)}% Match</span>
      </div>
      <div class="citation-snippet">${c.chunk.text.substring(0, 160)}...</div>
    </div>
  `).join("");
}

async function handleIngestSubmit(e) {
  e.preventDefault();
  const title = document.getElementById("inp-doc-title").value.trim();
  const content = document.getElementById("inp-doc-content").value.trim();
  const statusDiv = document.getElementById("ingest-status");

  try {
    const res = await fetch("/api/v1/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content })
    });

    if (res.status === 409) {
      statusDiv.innerHTML = `<span class="text-rose">⚠️ Duplicate document: Identical content hash already indexed.</span>`;
      return;
    }

    const data = await res.json();
    statusDiv.innerHTML = `<span class="text-green">✓ Ingested successfully: Created ${data.chunks_created} semantic chunks (${data.char_count} chars).</span>`;
    document.getElementById("inp-doc-title").value = "";
    document.getElementById("inp-doc-content").value = "";
    fetchHealthAndStats();
  } catch (err) {
    statusDiv.innerHTML = `<span class="text-rose">Ingestion error: ${err.message}</span>`;
  }
}

async function runBenchmark() {
  const btn = document.getElementById("btn-run-eval");
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Evaluating 15 Golden Test Cases... ⏳";
  }

  try {
    const res = await fetch("/api/v1/evaluate", { method: "POST" });
    const data = await res.json();

    // Update KPIs
    document.getElementById("kpi-hit").innerText = `${(data.hit_rate_at_k * 100).toFixed(1)}%`;
    document.getElementById("kpi-mrr").innerText = data.mean_reciprocal_rank_mrr.toFixed(3);
    document.getElementById("kpi-tool").innerText = `${(data.tool_call_accuracy * 100).toFixed(1)}%`;
    document.getElementById("kpi-sec").innerText = `${(data.adversarial_block_rate * 100).toFixed(1)}%`;
    document.getElementById("kpi-lat").innerText = `${data.average_latency_ms.toFixed(1)} ms`;

    // Render Table
    const tbody = document.getElementById("eval-table-body");
    tbody.innerHTML = data.details.map(d => {
      const statusColor = (d.tool_correct && (!d.is_answerable || d.retrieval_hit || d.is_blocked)) ? "text-green" : "text-amber";
      const statusLabel = statusColor === "text-green" ? "PASSED" : "WARN";
      return `
        <tr>
          <td><strong>${d.id}</strong></td>
          <td>${d.category}</td>
          <td>${d.query.substring(0, 55)}...</td>
          <td>${d.expected_tool || "None (Blocked)"}</td>
          <td>${d.tools_used.join(", ") || "None"}</td>
          <td>${d.retrieval_hit ? '✓ Hit' : (d.is_answerable ? '✗ Miss' : 'N/A')}</td>
          <td class="${statusColor}"><strong>${statusLabel}</strong></td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Eval error:", err);
    alert(`Benchmark evaluation failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "⚡ Run 15-Case Golden Evaluation";
    }
  }
}

async function fetchFailureModes() {
  try {
    const res = await fetch("/api/v1/failures");
    if (!res.ok) return;
    const failures = await res.json();

    const container = document.getElementById("failures-container");
    container.innerHTML = failures.map(f => `
      <div class="failure-card">
        <div class="failure-header">
          <div class="failure-title">${f.title}</div>
          <span class="failure-badge ${f.severity === 'CRITICAL' ? 'badge-sev-crit' : 'badge-sev-high'}">${f.severity}</span>
        </div>
        <div class="failure-vulnerability">${f.vulnerability}</div>
        <div class="failure-comparison">
          <div class="comp-unmitigated">
            <div class="comp-title">❌ Unmitigated Failure Mode</div>
            <pre style="white-space: pre-wrap; font-family: var(--font-mono); font-size: 0.72rem;">${f.unmitigated_behavior}</pre>
          </div>
          <div class="comp-mitigated">
            <div class="comp-title">🛡️ Engineered Mitigation & Result</div>
            <div style="margin-bottom: 6px;">${f.mitigation_strategy.replace(/\n/g, '<br>')}</div>
            <strong>Mitigated Output:</strong> ${f.mitigated_result}
          </div>
        </div>
      </div>
    `).join("");
  } catch (e) {
    console.warn("Could not fetch failure modes:", e);
  }
}

async function fetchHealthAndStats() {
  try {
    const res = await fetch("/api/v1/health");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("sys-chunks").innerText = `${data.total_chunks} Chunks`;
    if (document.getElementById("stat-docs")) document.getElementById("stat-docs").innerText = data.total_documents;
    if (document.getElementById("stat-chunks")) document.getElementById("stat-chunks").innerText = data.total_chunks;
  } catch (e) {
    console.warn("Health check error:", e);
  }
}
