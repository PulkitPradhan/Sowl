// Sowl Agent Harness Frontend (Vanilla JS, Zero Dependencies, textContent Only)

(function () {
  "use strict";

  // State
  let ws = null;
  let activeSessionId = null;
  let activeRunId = null;
  let activeConfirmHash = null;
  let activeToolCards = {}; // step -> DOM element
  let tokenStats = { prompt: 0, completion: 0, reasoning: 0, cached: 0 };
  let runStartTime = null;

  // DOM Elements
  const chatContainer = document.getElementById("chat-container");
  const taskInput = document.getElementById("task-input");
  const btnSend = document.getElementById("btn-send");
  const btnCancel = document.getElementById("btn-cancel");
  const sessionList = document.getElementById("session-list");
  const btnNewSession = document.getElementById("btn-new-session");
  const profileSelect = document.getElementById("profile-select");
  const thinkingSelect = document.getElementById("thinking-select");
  const connectionStatus = document.getElementById("connection-status");
  const modelIndicator = document.getElementById("model-indicator");
  const hardwareHint = document.getElementById("hardware-hint");
  const btnTogglePanel = document.getElementById("btn-toggle-panel");
  const rightPanel = document.getElementById("right-panel");
  const skillsList = document.getElementById("skills-list");
  const runStepTrace = document.getElementById("run-step-trace");

  // Token Stats Elements
  const statPrompt = document.getElementById("stat-prompt");
  const statCompletion = document.getElementById("stat-completion");
  const statReasoning = document.getElementById("stat-reasoning");
  const statCached = document.getElementById("stat-cached");
  const statSpeed = document.getElementById("stat-speed");

  // Modals
  const confirmModal = document.getElementById("confirm-modal");
  const confirmTool = document.getElementById("confirm-tool");
  const confirmRisk = document.getElementById("confirm-risk");
  const confirmHostRow = document.getElementById("confirm-host-row");
  const confirmHost = document.getElementById("confirm-host");
  const confirmActionHash = document.getElementById("confirm-action-hash");
  const confirmArgs = document.getElementById("confirm-args");
  const btnConfirmApprove = document.getElementById("btn-confirm-approve");
  const btnConfirmDeny = document.getElementById("btn-confirm-deny");

  const settingsModal = document.getElementById("settings-modal");
  const btnSettings = document.getElementById("btn-settings");
  const btnCloseSettings = document.getElementById("btn-close-settings");
  const profilesTableContainer = document.getElementById("profiles-table-container");

  const paletteModal = document.getElementById("palette-modal");
  const paletteInput = document.getElementById("palette-input");
  const paletteResults = document.getElementById("palette-results");

  // Sanitizer: ANSI, control chars, BiDi overrides
  const ANSI_REGEX = /\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;
  const CONTROL_REGEX = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g;
  const BIDI_REGEX = /[\u202A-\u202E\u2066-\u2069]/g;

  function sanitize(text) {
    if (typeof text !== "string") {
      text = String(text != null ? text : "");
    }
    return text.replace(ANSI_REGEX, "").replace(CONTROL_REGEX, "").replace(BIDI_REGEX, "");
  }

  // Cap DOM items at 200
  function enforceDomCap() {
    while (chatContainer.children.length > 200) {
      chatContainer.removeChild(chatContainer.firstElementChild);
    }
  }

  function appendChatMessage(type, text) {
    const el = document.createElement("div");
    el.className = "chat-msg chat-msg-" + type;
    el.textContent = sanitize(text);
    chatContainer.appendChild(el);
    enforceDomCap();
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return el;
  }

  // Token Fragment Auth
  async function checkAuthAndInit() {
    const hash = window.location.hash;
    if (hash && hash.startsWith("#token=")) {
      const token = hash.slice(7);
      window.history.replaceState(null, "", window.location.pathname);
      try {
        const res = await fetch("/api/auth", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token: token }),
        });
        if (!res.ok) {
          appendChatMessage("ask", "Authentication failed: Invalid or expired startup token.");
          return;
        }
      } catch (err) {
        appendChatMessage("ask", "Auth network error: " + sanitize(err.message));
        return;
      }
    }

    // Initialize application
    initApp();
  }

  async function initApp() {
    await loadStatus();
    await loadModels();
    await loadSkills();
    await loadSessions();
    connectWebSocket();
  }

  // WebSocket Connection
  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = proto + "//" + window.location.host + "/ws";

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      connectionStatus.textContent = "Connected";
      connectionStatus.className = "status-pill status-connected";
    };

    ws.onclose = function () {
      connectionStatus.textContent = "Disconnected";
      connectionStatus.className = "status-pill status-disconnected";
      // Auto-reconnect after 3s
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function () {
      connectionStatus.textContent = "Error";
      connectionStatus.className = "status-pill status-disconnected";
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        handleWsEvent(data);
      } catch (err) {
        console.error("WS Parse Error:", err);
      }
    };
  }

  function handleWsEvent(evt) {
    const evtType = evt.type;

    if (evtType === "queued") {
      appendChatMessage("agent", "Agent run is queued waiting for model lock...");
    } else if (evtType === "step") {
      const action = evt.action || {};
      const kind = action.kind;

      if (kind === "tool") {
        renderToolCard(evt.step, action.tool, action.args, evt.risk || "R1");
      } else if (kind === "done") {
        appendChatMessage("agent", action.summary || "Task completed.");
      } else if (kind === "ask") {
        appendChatMessage("ask", "Agent asked: " + (action.question || ""));
      }

      // Update trace list in right panel
      addStepTrace(evt.step, kind === "tool" ? action.tool : kind);
    } else if (evtType === "observation") {
      updateToolCardObservation(evt.step, evt.observation);
    } else if (evtType === "confirm") {
      showConfirmModal(evt);
    } else if (evtType === "done") {
      finishRun(evt);
    } else if (evtType === "error") {
      appendChatMessage("ask", "Error: " + sanitize(evt.message || "Unknown error"));
      resetRunState();
    }
  }

  // Tool Card Rendering
  function renderToolCard(step, toolName, args, risk) {
    const card = document.createElement("div");
    card.className = "tool-card";

    const header = document.createElement("div");
    header.className = "tool-card-header";

    const title = document.createElement("div");
    title.className = "tool-card-title";
    const badge = document.createElement("span");
    badge.className = "risk-badge risk-" + sanitize(risk);
    badge.textContent = sanitize(risk);
    const nameSpan = document.createElement("span");
    nameSpan.textContent = "[Step " + step + "] " + sanitize(toolName);
    title.appendChild(badge);
    title.appendChild(nameSpan);

    const status = document.createElement("span");
    status.className = "tool-status status-running";
    status.textContent = "Running";

    header.appendChild(title);
    header.appendChild(status);
    card.appendChild(header);

    const body = document.createElement("div");
    body.className = "tool-card-body";

    const argsPre = document.createElement("pre");
    argsPre.className = "tool-args-pre";
    argsPre.textContent = sanitize(JSON.stringify(args, null, 2));
    body.appendChild(argsPre);

    const obsContainer = document.createElement("div");
    obsContainer.className = "tool-obs-container";
    body.appendChild(obsContainer);

    card.appendChild(body);
    chatContainer.appendChild(card);
    enforceDomCap();
    chatContainer.scrollTop = chatContainer.scrollHeight;

    activeToolCards[step] = card;
  }

  function updateToolCardObservation(step, obsText) {
    const card = activeToolCards[step];
    if (!card) return;

    const status = card.querySelector(".tool-status");
    if (status) {
      status.className = "tool-status status-ok";
      status.textContent = "Done";
    }

    const obsContainer = card.querySelector(".tool-obs-container");
    if (obsContainer) {
      const summaryDiv = document.createElement("div");
      const cleanObs = sanitize(obsText);
      const preview = cleanObs.length > 240 ? cleanObs.slice(0, 240) + "..." : cleanObs;
      summaryDiv.textContent = "Result: " + preview;
      obsContainer.appendChild(summaryDiv);

      if (cleanObs.length > 240) {
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "tool-details-toggle";
        toggleBtn.textContent = "Show full raw result";
        const fullPre = document.createElement("pre");
        fullPre.className = "tool-raw-pre";
        fullPre.textContent = cleanObs;
        fullPre.style.display = "none";

        toggleBtn.addEventListener("click", function () {
          if (fullPre.style.display === "none") {
            fullPre.style.display = "block";
            toggleBtn.textContent = "Hide raw result";
          } else {
            fullPre.style.display = "none";
            toggleBtn.textContent = "Show full raw result";
          }
        });

        obsContainer.appendChild(toggleBtn);
        obsContainer.appendChild(fullPre);
      }
    }
  }

  // Confirmation Modal
  function showConfirmModal(evt) {
    activeConfirmHash = evt.action_hash;

    confirmTool.textContent = sanitize(evt.tool || "");
    confirmRisk.textContent = sanitize(evt.risk || "R1");
    confirmRisk.className = "risk-badge risk-" + sanitize(evt.risk || "R1");

    if (evt.hostname) {
      confirmHost.textContent = sanitize(evt.hostname);
      confirmHostRow.classList.remove("meta-hidden");
    } else {
      confirmHostRow.classList.add("meta-hidden");
    }

    confirmActionHash.textContent = sanitize(evt.action_hash || "");
    confirmArgs.textContent = sanitize(JSON.stringify(evt.args || {}, null, 2));

    confirmModal.classList.remove("modal-hidden");
  }

  function hideConfirmModal() {
    confirmModal.classList.add("modal-hidden");
    activeConfirmHash = null;
  }

  btnConfirmApprove.addEventListener("click", function () {
    if (!activeConfirmHash || !ws) return;
    ws.send(JSON.stringify({
      type: "confirm",
      action_hash: activeConfirmHash,
      approve: true,
    }));
    hideConfirmModal();
  });

  btnConfirmDeny.addEventListener("click", function () {
    if (!activeConfirmHash || !ws) return;
    ws.send(JSON.stringify({
      type: "confirm",
      action_hash: activeConfirmHash,
      approve: false,
    }));
    hideConfirmModal();
  });

  // Finish Run & Update Stats
  function finishRun(evt) {
    if (runStartTime) {
      const elapsed = (Date.now() - runStartTime) / 1000;
      if (elapsed > 0 && tokenStats.completion > 0) {
        statSpeed.textContent = (tokenStats.completion / elapsed).toFixed(1) + " tok/s";
      }
    }
    resetRunState();
  }

  function resetRunState() {
    activeRunId = null;
    runStartTime = null;
    btnSend.classList.remove("btn-hidden");
    btnCancel.classList.add("btn-hidden");
    taskInput.disabled = false;
    taskInput.focus();
  }

  function addStepTrace(step, label) {
    const item = document.createElement("div");
    item.className = "trace-item";
    const left = document.createElement("span");
    left.textContent = "Step " + step;
    const right = document.createElement("span");
    right.textContent = sanitize(label);
    item.appendChild(left);
    item.appendChild(right);
    runStepTrace.appendChild(item);
  }

  // User Actions
  function sendTask() {
    const text = taskInput.value.trim();
    if (!text) return;

    if (text.startsWith("/")) {
      handleSlashCommand(text);
      taskInput.value = "";
      return;
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      appendChatMessage("ask", "Error: Not connected to daemon.");
      return;
    }

    appendChatMessage("user", text);
    taskInput.value = "";
    runStartTime = Date.now();

    btnSend.classList.add("btn-hidden");
    btnCancel.classList.remove("btn-hidden");
    runStepTrace.replaceChildren();

    ws.send(JSON.stringify({
      type: "run",
      task: text,
      session_id: activeSessionId,
    }));
  }

  function cancelRun() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "cancel",
        run_id: activeRunId,
      }));
    }
    appendChatMessage("ask", "Cancellation requested.");
    resetRunState();
  }

  btnSend.addEventListener("click", sendTask);
  btnCancel.addEventListener("click", cancelRun);

  taskInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTask();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelRun();
    }
  });

  // Slash Commands
  function handleSlashCommand(cmd) {
    const parts = cmd.split(" ");
    const command = parts[0].toLowerCase();

    if (command === "/help") {
      appendChatMessage("agent", "Available Commands:\n/help - Show this guide\n/model - List available model profiles\n/skills - View active skills\n/trace - Show active trace\n/clear - Clear chat messages");
    } else if (command === "/clear") {
      chatContainer.replaceChildren();
      appendChatMessage("agent", "Chat cleared.");
    } else if (command === "/model") {
      loadModels().then(function () {
        appendChatMessage("agent", "Profile changed or listed.");
      });
    } else if (command === "/skills") {
      btnTogglePanel.click();
    } else if (command === "/trace") {
      btnTogglePanel.click();
    } else {
      appendChatMessage("ask", "Unknown slash command: " + sanitize(command));
    }
  }

  // Sessions Management
  async function loadSessions() {
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) return;
      const sessions = await res.json();
      sessionList.replaceChildren();

      sessions.forEach(function (s) {
        const item = document.createElement("div");
        item.className = "session-item" + (s.session_id === activeSessionId ? " active" : "");

        const titleSpan = document.createElement("span");
        titleSpan.className = "session-title";
        titleSpan.textContent = sanitize(s.title || "Session " + s.session_id.slice(0, 8));

        const delBtn = document.createElement("button");
        delBtn.className = "session-del-btn";
        delBtn.textContent = "×";
        delBtn.addEventListener("click", async function (e) {
          e.stopPropagation();
          await fetch("/api/sessions/" + s.session_id, { method: "DELETE" });
          if (activeSessionId === s.session_id) activeSessionId = null;
          await loadSessions();
        });

        item.appendChild(titleSpan);
        item.appendChild(delBtn);

        item.addEventListener("click", function () {
          selectSession(s.session_id);
        });

        sessionList.appendChild(item);
      });

      if (!activeSessionId && sessions.length > 0) {
        selectSession(sessions[0].session_id);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
    }
  }

  async function selectSession(sessionId) {
    activeSessionId = sessionId;
    const items = sessionList.querySelectorAll(".session-item");
    items.forEach(function (it) { it.classList.remove("active"); });

    try {
      const res = await fetch("/api/sessions/" + sessionId);
      if (!res.ok) return;
      const data = await res.json();
      chatContainer.replaceChildren();

      (data.messages || []).forEach(function (m) {
        if (m.type === "user") {
          appendChatMessage("user", m.text);
        } else if (m.type === "agent") {
          if (m.stop_reason === "ask") {
            appendChatMessage("ask", "Agent asked: " + (m.question || m.summary));
          } else {
            appendChatMessage("agent", m.summary);
          }
        }
      });
    } catch (err) {
      console.error("Failed to load session history:", err);
    }
  }

  btnNewSession.addEventListener("click", async function () {
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Session" }),
      });
      if (res.ok) {
        const data = await res.json();
        await loadSessions();
        selectSession(data.session_id);
      }
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  });

  // Status & Hardware Info
  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      const daemon = data.daemon || {};
      const doctor = data.doctor || {};
      const specs = doctor.specs || {};

      modelIndicator.textContent = "Model: " + sanitize(daemon.active_profile || "None");
      hardwareHint.textContent = "HW: " + sanitize(specs.gpu_name || "CPU (" + (specs.ram_gb || 0) + "GB)");
    } catch (err) {
      console.error("Failed to load status:", err);
    }
  }

  // Model Profiles
  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const models = await res.json();
      profileSelect.replaceChildren();

      models.forEach(function (m) {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = sanitize(m.name + " (" + m.model + ")");
        if (m.active) opt.selected = true;
        profileSelect.appendChild(opt);
      });
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }

  profileSelect.addEventListener("change", async function () {
    const chosen = profileSelect.value;
    try {
      const res = await fetch("/api/models/use", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: chosen }),
      });
      if (!res.ok) {
        const errData = await res.json();
        alert(errData.error || "Cannot switch profile");
        await loadModels();
      } else {
        await loadStatus();
      }
    } catch (err) {
      console.error("Failed to set model profile:", err);
    }
  });

  // Skills List
  async function loadSkills() {
    try {
      const res = await fetch("/api/skills");
      if (!res.ok) return;
      const skills = await res.json();
      skillsList.replaceChildren();

      skills.forEach(function (s) {
        const item = document.createElement("div");
        item.className = "skill-item";

        const hdr = document.createElement("div");
        hdr.className = "skill-item-header";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = sanitize(s.name);

        const badge = document.createElement("span");
        badge.className = "risk-badge risk-" + sanitize(s.risk);
        badge.textContent = sanitize(s.risk);

        hdr.appendChild(nameSpan);
        hdr.appendChild(badge);

        const desc = document.createElement("div");
        desc.className = "skill-desc";
        desc.textContent = sanitize(s.summary || "");

        item.appendChild(hdr);
        item.appendChild(desc);
        skillsList.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load skills:", err);
    }
  }

  // Panel Toggle
  btnTogglePanel.addEventListener("click", function () {
    rightPanel.classList.toggle("panel-collapsed");
  });

  // Settings Modal
  btnSettings.addEventListener("click", async function () {
    settingsModal.classList.remove("modal-hidden");
    // Render masked profiles table
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const models = await res.json();
      profilesTableContainer.replaceChildren();

      const table = document.createElement("table");
      table.className = "settings-table";
      const thead = document.createElement("thead");
      const trH = document.createElement("tr");
      ["Profile", "Tier", "Model", "Endpoint"].forEach(function (h) {
        const th = document.createElement("th");
        th.textContent = h;
        trH.appendChild(th);
      });
      thead.appendChild(trH);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      models.forEach(function (m) {
        const tr = document.createElement("tr");
        [m.name, "Tier " + m.tier, m.model, m.endpoint].forEach(function (val) {
          const td = document.createElement("td");
          td.textContent = sanitize(val);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      profilesTableContainer.appendChild(table);
    } catch (err) {
      console.error("Settings load error:", err);
    }
  });

  btnCloseSettings.addEventListener("click", function () {
    settingsModal.classList.add("modal-hidden");
  });

  // Command Palette (Ctrl+K)
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      paletteModal.classList.toggle("modal-hidden");
      if (!paletteModal.classList.contains("modal-hidden")) {
        paletteInput.value = "";
        paletteInput.focus();
        renderPaletteCommands("");
      }
    } else if (e.key === "Escape" && !paletteModal.classList.contains("modal-hidden")) {
      paletteModal.classList.add("modal-hidden");
    }
  });

  const PALETTE_COMMANDS = [
    { cmd: "/help", desc: "Show harness help guide" },
    { cmd: "/model", desc: "View and switch model profiles" },
    { cmd: "/skills", desc: "Toggle registered skills panel" },
    { cmd: "/trace", desc: "Toggle step trace panel" },
    { cmd: "/clear", desc: "Clear active chat messages" },
  ];

  function renderPaletteCommands(filter) {
    paletteResults.replaceChildren();
    const filtered = PALETTE_COMMANDS.filter(function (c) {
      return c.cmd.includes(filter.toLowerCase()) || c.desc.toLowerCase().includes(filter.toLowerCase());
    });

    filtered.forEach(function (c) {
      const it = document.createElement("div");
      it.className = "palette-item";
      const cmdSpan = document.createElement("span");
      cmdSpan.textContent = c.cmd;
      const descSpan = document.createElement("span");
      descSpan.textContent = c.desc;
      it.appendChild(cmdSpan);
      it.appendChild(descSpan);

      it.addEventListener("click", function () {
        paletteModal.classList.add("modal-hidden");
        handleSlashCommand(c.cmd);
      });

      paletteResults.appendChild(it);
    });
  }

  paletteInput.addEventListener("input", function () {
    renderPaletteCommands(paletteInput.value);
  });

  // Start initialization
  checkAuthAndInit();
})();
