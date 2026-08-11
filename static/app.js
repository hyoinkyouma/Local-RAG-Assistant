// ===== STATE =====
let currentTab = "chat";
let isSending = false;
let polling = false;
let dlPolling = false;
let webSearchEnabled = true; // Web search on by default (master switch)
let thinkingEnabled = true; // Thinking enabled by default
let ragEnabled = true;
let chatHistory = [];
let currentChatId = null;

// ===== DOM REFS =====
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const tabChat = $("#tab-chat");
const tabSettings = $("#tab-settings");
const tabBtns = $$(".tab-btn");
const messagesEl = $("#messages");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const sendBtn = $("#send-btn");
const noModelBanner = $("#no-model-banner");
const dropZone = $("#drop-zone");
const fileInput = $("#file-input");
const fileList = $("#file-list");
const ingestBtn = $("#ingest-btn");
const clearBtn = $("#clear-btn");
const progressArea = $("#progress-area");
const progressBar = $("#progress-bar");
const progressMsg = $("#progress-message");
const progressPct = $("#progress-percent");
const webSearchToggle = $("#web-search-toggle");
const ragToggle = $("#rag-toggle");
const thinkingToggle = $("#thinking-toggle");
const modelList = $("#model-list");
const dlProgressArea = $("#dl-progress-area");
const dlProgressBar = $("#dl-progress-bar");
const dlProgressMsg = $("#dl-progress-message");
const dlProgressPct = $("#dl-progress-pct");
const gpuStatus = $("#gpu-status");
const sidebar = $("#sidebar");
const chatList = $("#chat-list");
const sidebarCollapseBtn = $("#sidebar-collapse-btn");
const sidebarExpandIcon = $("#sidebar-expand-icon");
const domainNameInput = $("#domain-name-input");
const createDomainBtn = $("#create-domain-btn");
const domainList = $("#domain-list");
const domainFilesSelector = $("#domain-files-selector");
const domainFilesList = $("#domain-files-list");
const ingestDomainSelect = $("#ingest-domain-select");
const chatDomainSelect = $("#chat-domain-select");
const domainFilterBar = $("#domain-filter-bar");

// ===== SIDEBAR: CHAT PERSISTENCE =====

async function saveCurrentChat() {
  if (chatHistory.length === 0) return null;
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);
  if (currentChatId) {
    await fetch(`/v1/chats/${currentChatId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatHistory }),
    });
    return currentChatId;
  } else {
    const resp = await fetch("/v1/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatHistory }),
    });
    const data = await resp.json();
    currentChatId = data.id;
    loadChatList();
    generateTitle(data.id);
    return data.id;
  }
}

async function loadChatList() {
  try {
    const resp = await fetch("/v1/chats");
    const chats = await resp.json();
    chatList.innerHTML = chats
      .map((c) => {
        const active = c.id === currentChatId ? "active" : "";
        const time = formatTime(c.updated_at);
        return `<div class="chat-list-item ${active}" data-id="${c.id}" onclick="switchToChat('${c.id}')">
        <span class="chat-title">${escapeHtml(c.title)}</span>
        <span class="chat-time">${time}</span>
        <button class="chat-del-btn" onclick="event.stopPropagation();deleteChat('${c.id}')" title="Delete">&times;</button>
      </div>`;
      })
      .join("");
  } catch (e) {
    console.error("Failed to load chats:", e);
  }
}

async function switchToChat(id) {
  if (id === currentChatId) return;
  await saveCurrentChat();
  try {
    const resp = await fetch(`/v1/chats/${id}`);
    const data = await resp.json();
    chatHistory = data.messages || [];
    currentChatId = data.id;
    messagesEl.innerHTML = "";
    for (const msg of chatHistory) {
      addMessage(msg.role, msg.content || "", msg.citations);
    }
    document
      .querySelectorAll(".chat-list-item.active")
      .forEach((el) => el.classList.remove("active"));
    const item = document.querySelector(`.chat-list-item[data-id="${id}"]`);
    if (item) item.classList.add("active");
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (e) {
    console.error("Failed to switch chat:", e);
  }
}

async function deleteChat(id) {
  if (!confirm("Delete this conversation?")) return;
  try {
    await fetch(`/v1/chats/${id}`, { method: "DELETE" });
    if (currentChatId === id) {
      currentChatId = null;
      messagesEl.innerHTML = "";
      chatHistory = [];
      addMessage(
        "assistant",
        "Hello! I can answer questions about your documents. Upload files in Settings to expand my knowledge. If no model is loaded, go to Settings to download and activate one.",
      );
    }
    loadChatList();
  } catch (e) {
    console.error("Failed to delete chat:", e);
  }
}

async function generateTitle(chatId) {
  if (!chatHistory.length) return;
  let text = chatHistory[0]?.content || "";
  if (chatHistory[1]) text += "\n" + (chatHistory[1]?.content || "");
  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "default",
        messages: [
          {
            role: "system",
            content:
              "Summarize this conversation in 5 words or less. Reply with ONLY the title.",
          },
          { role: "user", content: text.slice(0, 1000) },
        ],
        temperature: 0.3,
        max_tokens: 20,
        stream: false,
        web_search: false,
        enable_thinking: true,
      }),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    let raw = data.choices?.[0]?.message?.content || "";
    let title =
      raw
        .replace(/<think>[\s\S]*?<\/think>/gi, "")
        .replace(/<\/?\s*think\s*\/?>/gi, "")
        .trim() || "New Chat";
    title = title.replace(/^["'\s]+|["'\s]+$/g, "").replace(/[."']+$/, "");
    if (title.length > 60) title = title.slice(0, 60);
    await fetch(`/v1/chats/${chatId}/title`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title || "New Chat" }),
    });
    loadChatList();
  } catch (e) {
    // title generation is non-critical
  }
}

function formatTime(iso) {
  if (!iso) return "";
  const d = new Date(iso + "Z");
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 60) return "now";
  if (diff < 3600) return Math.round(diff / 60) + "m";
  if (diff < 86400) return Math.round(diff / 3600) + "h";
  return d.toLocaleDateString();
}

// ===== SIDEBAR COLLAPSE =====

sidebarCollapseBtn.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
  const isCollapsed = sidebar.classList.contains("collapsed");
  sidebarCollapseBtn.title = isCollapsed
    ? "Expand sidebar"
    : "Collapse sidebar";
  sidebarExpandIcon.classList.toggle("hidden", !isCollapsed);
});

// ===== TAB SWITCHING =====
tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  currentTab = tab;
  tabBtns.forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  tabChat.classList.toggle("hidden", tab !== "chat");
  tabChat.classList.toggle("flex", tab === "chat");
  tabSettings.classList.toggle("hidden", tab !== "settings");
  tabSettings.classList.toggle("flex", tab === "settings");
  if (tab === "settings") {
    refreshFileList();
    refreshModelList();
    refreshDomainUI();
  }
}

// ===== MODEL LIST =====
async function refreshModelList() {
  try {
    const resp = await fetch("/v1/models");
    const data = await resp.json();
    renderModelList(data.models, data.current_model);
  } catch (e) {
    console.error("Failed to fetch models:", e);
  }
}

function renderModelList(models, currentId) {
  modelList.innerHTML = models
    .map((m) => {
      let badge = "";
      let btn = "";
      if (m.active) {
        badge =
          '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">Active</span>';
      } else if (m.downloaded) {
        btn = `<button onclick="selectModel('${m.id}')" class="px-3 py-1 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700">Activate</button>`;
        badge =
          '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">Downloaded</span>';
      } else {
        btn = `<button onclick="downloadModel('${m.id}')" class="px-3 py-1 text-xs font-medium rounded-md bg-gray-200 text-gray-700 hover:bg-gray-300">Download</button>`;
        badge =
          '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-400">Not downloaded</span>';
      }
      const border = m.active ? "border-blue-500 active" : "border-gray-200";
      return `<div class="model-card rounded-xl border ${border} p-4 flex items-start justify-between gap-4">
      <div class="min-w-0">
        <div class="flex items-center gap-2 mb-1">
          <span class="font-medium text-sm text-gray-800">${m.name}</span>
          ${badge}
        </div>
        <p class="text-xs text-gray-500">${m.description}</p>
        <p class="text-xs text-gray-400 mt-1">${m.size_human} &middot; ${m.repo_id}</p>
      </div>
      <div class="shrink-0">${btn}</div>
    </div>`;
    })
    .join("");
}

async function downloadModel(key) {
  try {
    const resp = await fetch(`/v1/models/download/${key}`, { method: "POST" });
    const data = await resp.json();
    if (data.status === "started") {
      dlProgressArea.classList.remove("hidden");
      dlProgressBar.style.width = "0%";
      dlProgressPct.textContent = "0%";
      dlProgressMsg.textContent = "Starting download...";
      if (!dlPolling) startDlPolling();
    } else if (data.status === "already_downloaded") {
      refreshModelList();
    }
  } catch (e) {
    alert("Failed to start download: " + e.message);
  }
}

function startDlPolling() {
  dlPolling = true;
  const interval = setInterval(async () => {
    try {
      const resp = await fetch("/v1/models/download/progress");
      const prog = await resp.json();
      dlProgressBar.style.width = prog.progress + "%";
      dlProgressPct.textContent = prog.progress + "%";
      dlProgressMsg.textContent = prog.message || "Downloading...";

      if (prog.status === "completed") {
        clearInterval(interval);
        dlPolling = false;
        dlProgressMsg.textContent = "Download complete!";
        setTimeout(() => {
          dlProgressArea.classList.add("hidden");
        }, 3000);
        refreshModelList();
      } else if (prog.status === "error") {
        clearInterval(interval);
        dlPolling = false;
        dlProgressBar.style.width = "0%";
        dlProgressMsg.textContent = "Download failed. Check the server logs.";
      }
    } catch (e) {
      console.error("DL poll error:", e);
    }
  }, 1000);
}

async function newChat() {
  await saveCurrentChat();
  messagesEl.innerHTML = "";
  chatHistory = [];
  currentChatId = null;
  document
    .querySelectorAll(".chat-list-item.active")
    .forEach((el) => el.classList.remove("active"));
  addMessage(
    "assistant",
    "Hello! I can answer questions about your documents. Upload files in Settings to expand my knowledge. If no model is loaded, go to Settings to download and activate one.",
  );
}

async function selectModel(key) {
  try {
    const resp = await fetch(`/v1/models/select/${key}`, { method: "POST" });
    const data = await resp.json();
    if (data.status === "ok") {
      refreshModelList();
      checkHealth();
      newChat();
    }
  } catch (e) {
    alert("Failed to select model: " + e.message);
  }
}

// ===== HEALTH / FIRST-START =====
async function checkHealth() {
  try {
    const resp = await fetch("/health");
    const data = await resp.json();
    if (data.model_loaded) {
      noModelBanner.classList.add("hidden");
      chatInput.disabled = false;
      sendBtn.disabled = false;
    } else {
      noModelBanner.classList.remove("hidden");
      chatInput.disabled = true;
      sendBtn.disabled = true;
    }
    if (data.gpu_available) {
      gpuStatus.textContent =
        "Enabled \u2014 " + (data.gpu_name || "NVIDIA GPU");
      gpuStatus.className = "text-green-600";
    } else {
      gpuStatus.textContent = "Not available (CPU only)";
      gpuStatus.className = "text-gray-400";
    }
  } catch (e) {
    console.error("Health check failed:", e);
  }
}

// ===== MARKDOWN RENDERER =====
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function stripThinking(text) {
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  text = text.replace(/^[\s\S]*?<\/think>/i, "");
  text = text.replace(/<\/?\s*think\s*\/?>/gi, "");
  return text.trim();
}

function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  html = html.replace(
    /^(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)*)/gm,
    (_, head, sep, body) => {
      const headers = head
        .split("|")
        .filter((c) => c.trim())
        .map((c) => `<th>${c.trim()}</th>`)
        .join("");
      const rows = body
        .trim()
        .split("\n")
        .map((row) => {
          const cells = row
            .split("|")
            .filter((c) => c.trim())
            .map((c) => `<td>${c.trim()}</td>`)
            .join("");
          return `<tr>${cells}</tr>`;
        })
        .join("");
      return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    },
  );
  const paras = html.split(/\n\n+/);
  return paras
    .map((p) => {
      p = p.trim();
      if (!p) return "";
      if (/^<h[123]>/.test(p)) return p;
      if (/^<table>/.test(p)) return p;
      p = p.replace(/\n/g, "<br>");
      return `<p>${p}</p>`;
    })
    .join("");
}

// ===== CHAT =====
function splitThinking(text) {
  const cleaned = stripThinking(text);
  const thinkMatch = cleaned.match(
    /^(Thinking\s*Process|Thought|Reasoning|Analysis)\s*:?\s*\n/i,
  );
  if (!thinkMatch) return [null, cleaned];

  const lines = cleaned.split("\n");
  let splitIdx = lines.length;
  for (let i = lines.length - 1; i >= 0; i--) {
    const trimmed = lines[i].trim();
    if (
      /^(?:\d+\.\s|Final\s*:?\s|Check\s|Let's\s|Wait,\s|Actually,\s|Refining|Draft\s)/i.test(
        trimmed,
      ) &&
      trimmed.length < 120
    ) {
      splitIdx = i;
      break;
    }
  }
  const thinkText = lines.slice(0, splitIdx).join("\n").trim();
  const respText = lines.slice(splitIdx).join("\n").trim();
  return [thinkText || null, respText || cleaned];
}

function renderThinkingCollapsed(text) {
  const [thinkText, respText] = splitThinking(text);
  if (thinkText && respText !== thinkText) {
    return `<details class="thinking-details"><summary>Thinking Process</summary><div class="thinking-text">${renderMarkdown(thinkText)}</div></details>${renderMarkdown(respText)}`;
  }
  return renderMarkdown(respText);
}

function renderContent(text) {
  if (thinkingEnabled) return renderThinkingCollapsed(text);
  const [, respText] = splitThinking(text);
  return renderMarkdown(respText);
}

function openCitation(c) {
  if (!c.url) return;
  let url = c.url;
  if (!/^https?:\/\//i.test(url)) {
    url = new URL(url, window.location.href).href;
  }
  const api = window.pywebview && window.pywebview.api;
  if (api && (api.openExternal || api.open_external)) {
    const fn = api.openExternal || api.open_external;
    try { fn(url); return; } catch (e) {}
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

function buildCitationsElement(citations) {
  const citeWrapper = document.createElement("div");
  citeWrapper.className =
    "mt-2 pt-2 border-t border-gray-200 dark:border-gray-600 space-y-1";
  citations.forEach((c) => {
    const cite = document.createElement("div");
    cite.className = "text-xs";
    const body = document.createElement("div");
    body.className = "citation-body pl-4 text-gray-500";
    body.textContent = c.content;
    const header = document.createElement("div");
    header.className = "flex items-center gap-1";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "citation-toggle text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1";
    const isWeb = /^https?:\/\//i.test(c.url || "");
    const pagePart = c.page != null ? " (p." + (Number(c.page) + 1) + ")" : "";
    const icon = isWeb
      ? `<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>`
      : `<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19.5 14.25v-5.25a2.25 2.25 0 00-2.25-2.25H9.75a2.25 2.25 0 00-2.25 2.25v11.25A2.25 2.25 0 009.75 20.25h7.5A2.25 2.25 0 0021.5 18v-3.75z"/><path d="M15.75 2.25H6.75A2.25 2.25 0 004.5 4.5v11.25"/></svg>`;
    btn.innerHTML = icon + " " + c.source + pagePart;
    btn.title = isWeb
      ? "Open in browser"
      : (c.page != null ? "Open PDF at page " + (Number(c.page) + 1) : "Open in browser");
    btn.addEventListener("click", () => openCitation(c));
    header.appendChild(btn);
    const expandBtn = document.createElement("button");
    expandBtn.type = "button";
    expandBtn.className = "citation-toggle text-gray-400 hover:text-gray-600 ml-1";
    expandBtn.innerHTML = `<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg>`;
    expandBtn.title = "Show snippet";
    expandBtn.addEventListener("click", () => body.classList.toggle("open"));
    header.appendChild(expandBtn);
    cite.appendChild(header);
    cite.appendChild(body);
    citeWrapper.appendChild(cite);
  });
  return citeWrapper;
}

function addMessage(role, content, citations) {
  const div = document.createElement("div");
  div.className = `flex ${role === "user" ? "justify-end" : "justify-start"}`;
  const inner = document.createElement("div");
  inner.className = `max-w-[80%] md:max-w-[70%] px-4 py-2.5 msg-content ${role === "user" ? "msg-user" : "msg-assistant"}`;
  inner.innerHTML = renderContent(content);

  if (role === "assistant") {
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
    copyBtn.title = "Copy";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(content);
      copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>`;
      copyBtn.title = "Copied";
      setTimeout(() => {
        copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
        copyBtn.title = "Copy";
      }, 2000);
    });
    inner.appendChild(copyBtn);
  }

  div.appendChild(inner);

  if (role === "assistant" && citations && citations.length > 0) {
    const citeWrapper = buildCitationsElement(citations);
    inner.appendChild(citeWrapper);
  }

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTypingIndicator(label) {
  const div = document.createElement("div");
  div.id = "typing-indicator";
  div.className = "flex justify-start";
  div.innerHTML = `<div class="bg-gray-100 rounded-2xl px-4 py-3 typing-indicator flex items-center gap-2">
    <span class="text-xs text-gray-500">${escapeHtml(label || "Thinking")}&hellip;</span>
    <span class="typing-dots flex gap-1"><span></span><span></span><span></span></span>
  </div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function updateStreamStatus(container, message) {
  let statusEl = container.querySelector(".stream-status");
  if (!statusEl) {
    statusEl = document.createElement("div");
    statusEl.className = "stream-status";
    container.appendChild(statusEl);
  }
  statusEl.innerHTML = `<span class="typing-dots flex gap-1"><span></span><span></span><span></span></span><span class="text-xs text-gray-500">${escapeHtml(message)}</span>`;
}

function removeStreamStatus(container) {
  const statusEl = container.querySelector(".stream-status");
  if (statusEl) statusEl.remove();
}

ragToggle.addEventListener("click", () => {
  ragEnabled = !ragEnabled;
  ragToggle.classList.toggle("active", ragEnabled);
});

thinkingToggle.addEventListener("click", () => {
  thinkingEnabled = !thinkingEnabled;
  thinkingToggle.classList.toggle("active", thinkingEnabled); // Active when thinking is enabled
});

// Initialize thinking toggle to show active (thinking enabled by default)
thinkingToggle.classList.add("active");
// Initialize web search toggle to show active (web search on by default)
webSearchToggle.classList.add("active");

async function syncWebSearchToggle() {
  try {
    const resp = await fetch("/v1/settings");
    const data = await resp.json();
    webSearchEnabled = !!data.web_search_enabled;
    webSearchToggle.classList.toggle("active", webSearchEnabled);
  } catch (e) {
    console.error("Failed to load settings:", e);
  }
}

webSearchToggle.addEventListener("click", async () => {
  const next = !webSearchEnabled;
  webSearchToggle.classList.toggle("active", next);
  try {
    const resp = await fetch("/v1/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ web_search_enabled: next }),
    });
    if (resp.ok) {
      webSearchEnabled = next;
    } else {
      webSearchToggle.classList.toggle("active", webSearchEnabled);
      alert("Failed to update web search setting");
    }
  } catch (e) {
    webSearchToggle.classList.toggle("active", webSearchEnabled);
    alert("Failed to update web search setting: " + e.message);
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text || isSending) return;

  chatInput.value = "";
  sendBtn.disabled = true;
  isSending = true;

  const msgHtml = text;
  addMessage("user", msgHtml);
  addTypingIndicator("Thinking");

  try {
    const domFilter = chatDomainSelect.value;
    const history = chatHistory.map((m) => ({ role: m.role, content: m.content }));
    const body = {
      model: "default",
      messages: [...history, { role: "user", content: text }],
      temperature: 0.7,
      max_tokens: 8192,
      stream: thinkingEnabled,
      enable_thinking: thinkingEnabled,
    };
    if (!ragEnabled) body.disable_rag = true;
    if (domFilter) body.domains = [domFilter];
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const errData = await resp.json();
      throw new Error(errData.detail || resp.statusText);
    }

    // Create the assistant message div early for progressive fill
    removeTypingIndicator();
    const assistantDiv = document.createElement("div");
    assistantDiv.className = "flex justify-start";
    const assistantInner = document.createElement("div");
    assistantInner.className =
      "max-w-[80%] md:max-w-[70%] px-4 py-2.5 msg-content msg-assistant";
    assistantInner.id = "streaming-msg";
    assistantDiv.appendChild(assistantInner);
    messagesEl.appendChild(assistantDiv);

    // Build content progressively
    let rawContent = "";
    let citationsData = null;
    let contentStarted = false;
    updateStreamStatus(assistantInner, "Thinking");

    if (body.stream) {
      let buffer = "";
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") continue;

          try {
            const evt = JSON.parse(payload);

            if (evt.type === "status") {
              if (!contentStarted)
                updateStreamStatus(assistantInner, evt.message || evt.phase);
              continue;
            }
            if (evt.type === "citations") {
              citationsData = evt.citations;
              continue;
            }
            if (evt.type === "usage") continue;

            const choice = evt.choices?.[0];
            if (!choice) continue;

            const delta = choice.delta || {};
            if (delta.content) {
              if (!contentStarted) {
                contentStarted = true;
                removeStreamStatus(assistantInner);
              }
              rawContent += delta.content;
              const rendered = renderContent(rawContent);
              assistantInner.innerHTML = rendered;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch {}
        }
      }
    } else {
      const data = await resp.json();
      rawContent = data.choices?.[0]?.message?.content || "";
      citationsData = data.citations || null;
    }

    // Final render with citations
    const finalHtml = renderContent(rawContent);
    assistantInner.innerHTML = finalHtml;
    assistantInner.id = "";

    // Add copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
    copyBtn.title = "Copy";
    copyBtn.addEventListener("click", () => {
      const copyContent = thinkingEnabled
        ? rawContent
        : stripThinking(rawContent);
      navigator.clipboard.writeText(copyContent);
      copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>`;
      copyBtn.title = "Copied";
      setTimeout(() => {
        copyBtn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>`;
        copyBtn.title = "Copy";
      }, 2000);
    });
    assistantInner.appendChild(copyBtn);

    // Citations after the message
    if (citationsData && citationsData.length > 0) {
      assistantInner.appendChild(buildCitationsElement(citationsData));
    }

    // Save to history
    chatHistory.push({ role: "user", content: text });
    chatHistory.push({
      role: "assistant",
      content: thinkingEnabled ? rawContent : stripThinking(rawContent),
      citations: citationsData || [],
    });

    // Auto-save to backend
    await saveCurrentChat();

    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", "Error: " + err.message);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
  }
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

// ===== FILE UPLOAD =====
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("dragover"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = "";
});

async function uploadFiles(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  try {
    const resp = await fetch("/v1/files/upload", {
      method: "POST",
      body: form,
    });
    const data = await resp.json();
    if (data.status === "ok") refreshFileList();
  } catch (err) {
    alert("Upload failed: " + err.message);
  }
}

async function refreshFileList() {
  try {
    const resp = await fetch("/v1/files");
    const data = await resp.json();
    const uf = data.files || [];
    renderFileList(uf);
  } catch (err) {
    console.error("Failed to list files:", err);
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function renderFileList(files) {
  if (!files.length) {
    fileList.innerHTML =
      '<p class="text-sm text-gray-400 text-center py-4">No files uploaded yet</p>';
    ingestBtn.disabled = true;
    return;
  }
  ingestBtn.disabled = false;
  fileList.innerHTML = files
    .map(
      (f) =>
        `<div class="file-item flex items-center justify-between px-3 py-2 rounded-lg border border-gray-200">
      <div class="flex items-center gap-2 min-w-0">
        <svg class="w-5 h-5 text-gray-400 shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M19.5 14.25v-6.3a1.5 1.5 0 00-.44-1.06L14.6 3.44a1.5 1.5 0 00-1.06-.44H6.75A2.25 2.25 0 004.5 5.25v13.5A2.25 2.25 0 006.75 21h6.75"/><path d="M14.25 3.75v4.5a.75.75 0 00.75.75h4.5"/></svg>
        <span class="text-sm text-gray-700 truncate">${f.name}</span>
        <span class="text-xs text-gray-400 shrink-0">${formatSize(f.size)}</span>
      </div>
      <button onclick="deleteFile('${f.name}')" class="text-gray-400 hover:text-red-500 p-1 shrink-0">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>`,
    )
    .join("");
}

async function deleteFile(name) {
  try {
    await fetch("/v1/files/" + encodeURIComponent(name), { method: "DELETE" });
    refreshFileList();
  } catch (err) {
    console.error("Delete failed:", err);
  }
}

clearBtn.addEventListener("click", async () => {
  try {
    await fetch("/v1/files/clear", { method: "POST" });
    refreshFileList();
  } catch (err) {
    console.error("Clear failed:", err);
  }
});

// ===== DOMAINS =====

async function loadDomains() {
  try {
    const resp = await fetch("/v1/domains");
    const data = await resp.json();
    const domains = data.domains || [];
    return domains;
  } catch (e) {
    console.error("Failed to load domains:", e);
    return [];
  }
}

async function loadDomainFiles(domain) {
  try {
    const resp = await fetch(`/v1/domains/${encodeURIComponent(domain)}/files`);
    const data = await resp.json();
    return data.files || [];
  } catch (e) {
    console.error("Failed to load domain files:", e);
    return [];
  }
}

function populateDomainSelectors(domains) {
  const render = (sel, allLabel) => {
    const cur = sel.value;
    sel.innerHTML =
      `<option value="">${allLabel}</option>` +
      domains
        .map((d) => `<option value="${d.name}">${d.name}</option>`)
        .join("");
    if (cur) sel.value = cur;
  };
  render(ingestDomainSelect, "Select domain");
  render(domainFilesSelector, "Select domain");
  render(chatDomainSelect, "All domains");
  domainFilterBar.classList.toggle("hidden", domains.length <= 1);
}

async function refreshDomainUI() {
  const domains = await loadDomains();
  populateDomainSelectors(domains);
  //  domain list
  domainList.innerHTML = domains
    .map((d) => {
      const canDelete =
        d.name !== "General"
          ? `<button onclick="deleteDomain('${d.name}')" class="text-gray-400 hover:text-red-500 p-1 shrink-0"><svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg></button>`
          : "";
      return `<div class="domain-item flex items-center justify-between px-3 py-2 rounded-lg border border-gray-200">
      <span class="text-sm text-gray-700 font-medium">${d.name}</span>
      <span class="text-xs text-gray-400">${d.file_count} file(s)</span>
      ${canDelete}
    </div>`;
    })
    .join("");
  // Show files for selected domain in view selector
  const selDomain = domainFilesSelector.value;
  if (selDomain) {
    const files = await loadDomainFiles(selDomain);
    renderDomainFileList(selDomain, files);
  } else {
    domainFilesList.innerHTML =
      '<p class="text-sm text-gray-400 text-center py-4">Select a domain above to view its documents</p>';
  }
}

function renderDomainFileList(domain, files) {
  if (!files.length) {
    domainFilesList.innerHTML =
      '<p class="text-sm text-gray-400 text-center py-4">No documents in this domain</p>';
    return;
  }
  domainFilesList.innerHTML = files
    .map(
      (f) =>
        `<div class="file-item flex items-center justify-between px-3 py-2 rounded-lg border border-gray-200">
      <div class="flex items-center gap-2 min-w-0">
        <svg class="w-5 h-5 text-gray-400 shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M19.5 14.25v-6.3a1.5 1.5 0 00-.44-1.06L14.6 3.44a1.5 1.5 0 00-1.06-.44H6.75A2.25 2.25 0 004.5 5.25v13.5A2.25 2.25 0 006.75 21h6.75"/><path d="M14.25 3.75v4.5a.75.75 0 00.75.75h4.5"/></svg>
        <span class="text-sm text-gray-700 truncate">${f.name}</span>
        <span class="text-xs text-gray-400 shrink-0">${formatSize(f.size)}</span>
      </div>
      <button onclick="deleteDomainFile('${domain}','${f.name}')" class="text-gray-400 hover:text-red-500 p-1 shrink-0">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>`,
    )
    .join("");
}

async function deleteDomain(name) {
  if (!confirm(`Delete domain "${name}" and ALL its documents?`)) return;
  try {
    const resp = await fetch(`/v1/domains/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });
    if (!resp.ok) return alert("Failed to delete domain");
    refreshDomainUI();
    refreshFileList();
  } catch (e) {
    console.error("Delete domain failed:", e);
  }
}

async function deleteDomainFile(domain, filename) {
  if (!confirm(`Delete "${filename}" from domain "${domain}"?`)) return;
  try {
    const resp = await fetch(
      `/v1/domains/${encodeURIComponent(domain)}/files/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    );
    if (!resp.ok) return alert("Failed to delete file");
    const files = await loadDomainFiles(domain);
    renderDomainFileList(domain, files);
  } catch (e) {
    console.error("Delete domain file failed:", e);
  }
}

createDomainBtn.addEventListener("click", async () => {
  const name = domainNameInput.value.trim();
  if (!name) return;
  try {
    const resp = await fetch("/v1/domains", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      return alert(err.detail || "Failed to create domain");
    }
    domainNameInput.value = "";
    refreshDomainUI();
  } catch (e) {
    alert("Failed to create domain: " + e.message);
  }
});

domainNameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") createDomainBtn.click();
});

domainFilesSelector.addEventListener("change", async () => {
  const domain = domainFilesSelector.value;
  if (!domain) {
    domainFilesList.innerHTML =
      '<p class="text-sm text-gray-400 text-center py-4">Select a domain above to view its documents</p>';
    return;
  }
  const files = await loadDomainFiles(domain);
  renderDomainFileList(domain, files);
});

// ===== INGESTION =====
ingestBtn.addEventListener("click", async () => {
  if (ingestBtn.disabled) return;
  const domain = ingestDomainSelect.value;
  if (!domain) {
    alert("Please select a domain for ingestion");
    return;
  }
  try {
    const resp = await fetch("/v1/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain }),
    });
    const data = await resp.json();
    if (data.status === "started") {
      progressArea.classList.remove("hidden");
      progressBar.style.width = "0%";
      progressPct.textContent = "0%";
      progressMsg.textContent = "Starting ingestion...";
      ingestBtn.disabled = true;
      if (!polling) startPolling();
    }
  } catch (err) {
    alert("Failed to start ingestion: " + err.message);
  }
});

function startPolling() {
  polling = true;
  const interval = setInterval(async () => {
    try {
      const resp = await fetch("/v1/ingest/progress");
      const prog = await resp.json();
      const pct =
        prog.total > 0 ? Math.round((prog.current / prog.total) * 100) : 0;
      progressBar.style.width = pct + "%";
      progressPct.textContent = pct + "%";
      progressMsg.textContent = prog.message || "Processing...";
      if (
        prog.status === "completed" ||
        prog.status === "error" ||
        prog.status === "idle"
      ) {
        clearInterval(interval);
        polling = false;
        ingestBtn.disabled = false;
        if (prog.status === "completed") {
          progressMsg.textContent = "Done! " + (prog.message || "");
          refreshFileList();
          refreshDomainUI();
          setTimeout(() => {
            progressArea.classList.add("hidden");
          }, 3000);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1000);
}

// ===== DARK MODE =====
const darkToggle = $("#dark-toggle");
const html = document.documentElement;

function setTheme(dark) {
  if (dark) {
    html.classList.add("dark");
    localStorage.setItem("theme", "dark");
  } else {
    html.classList.remove("dark");
    localStorage.setItem("theme", "light");
  }
}

if (
  localStorage.getItem("theme") === "dark" ||
  (!localStorage.getItem("theme") &&
    window.matchMedia("(prefers-color-scheme: dark)").matches)
) {
  html.classList.add("dark");
}

darkToggle.addEventListener("click", () => {
  setTheme(!html.classList.contains("dark"));
});

// ===== INIT =====
async function init() {
  await checkHealth();
  syncWebSearchToggle();
  loadChatList();
  refreshDomainUI();
  addMessage(
    "assistant",
    "Hello! I can answer questions about your documents. Upload files in Settings to expand my knowledge. If no model is loaded, go to Settings to download and activate one.",
  );
}

init();
