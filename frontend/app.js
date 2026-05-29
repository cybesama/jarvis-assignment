/**
 * Jarvina frontend — WebSocket + AudioWorklet client.
 *
 * Audio pipeline (capture):
 *   Mic → AudioContext (48kHz) → AudioWorkletNode (resample + chunk)
 *   → WebSocket binary frames (Int16 PCM 16kHz, 100ms chunks)
 *
 * Audio pipeline (playback):
 *   WebSocket binary frames (WAV) → decode → AudioBuffer queue → AudioContext
 */

const WS_URL  = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
const PLAY_SR = 24000;   // TTS output sample rate

// ── DOM refs ──────────────────────────────────────────────────────────────────
const conversation  = document.getElementById("conversation");
const welcomeMsg    = document.getElementById("welcomeMsg");
const statusDot     = document.getElementById("statusDot");
const statusText    = document.getElementById("statusText");
const micBtn        = document.getElementById("micBtn");
const resetBtn      = document.getElementById("resetBtn");
const latencyBar    = document.getElementById("latencyBar");
const waveCanvas    = document.getElementById("waveCanvas");
const visualizer    = document.getElementById("visualizer");
const waveCtx       = waveCanvas.getContext("2d");

// ── State ─────────────────────────────────────────────────────────────────────
let ws           = null;
let audioCtx     = null;
let workletNode  = null;
let micStream    = null;
let micActive    = false;
let playQueue    = [];       // ArrayBuffer queue for TTS audio
let isPlaying    = false;
let nextPlayTime = 0;
let analyser     = null;
let currentAssistantBubble = null;

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("listening", "Listening");
    startMic();
  };

  ws.onclose = () => {
    setStatus("error", "Disconnected — retrying…");
    stopMic();
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => setStatus("error", "Connection error");

  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      enqueueAudio(evt.data);
    } else {
      handleJSON(JSON.parse(evt.data));
    }
  };
}

function handleJSON(msg) {
  switch (msg.type) {
    case "status":
      const labels = {
        listening:  "Listening",
        processing: "Thinking…",
        speaking:   "Speaking",
        idle:       "Idle",
      };
      setStatus(msg.state, labels[msg.state] || msg.state);
      if (msg.state === "processing") closeAssistantBubble();
      break;

    case "transcript":
      if (msg.is_final) {
        hideWelcome();
        appendMessage("user", msg.text);
      }
      break;

    case "response":
      appendAssistantToken(msg.text);
      break;

    case "latency":
      showLatency(msg.data);
      break;

    case "error":
      console.error("Server error:", msg.message);
      break;
  }
}

// ── Mic capture ───────────────────────────────────────────────────────────────
async function startMic() {
  if (micActive) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });

    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule("/static/audio-processor.js");

    const source = audioCtx.createMediaStreamSource(micStream);
    workletNode  = new AudioWorkletNode(audioCtx, "mic-processor");

    workletNode.port.onmessage = (e) => {
      if (e.data.type === "chunk" && ws?.readyState === WebSocket.OPEN) {
        ws.send(e.data.buffer);
      }
    };

    // Visualizer
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    source.connect(workletNode);
    workletNode.connect(audioCtx.destination);

    micActive = true;
    micBtn.classList.remove("muted");
    visualizer.classList.add("active");
    drawWave();
  } catch (err) {
    setStatus("error", "Mic access denied");
    console.error(err);
  }
}

function stopMic() {
  micStream?.getTracks().forEach(t => t.stop());
  workletNode?.disconnect();
  audioCtx?.close();
  micActive = false;
  micBtn.classList.add("muted");
  visualizer.classList.remove("active");
}

micBtn.addEventListener("click", async () => {
  if (micActive) {
    stopMic();
    setStatus("idle", "Mic off");
  } else {
    await startMic();
    setStatus("listening", "Listening");
  }
});

resetBtn.addEventListener("click", () => {
  ws?.send(JSON.stringify({ type: "reset" }));
  conversation.innerHTML = "";
  conversation.appendChild(welcomeMsg);
  welcomeMsg.style.display = "";
  currentAssistantBubble = null;
  latencyBar.style.display = "none";
});

// ── TTS playback ──────────────────────────────────────────────────────────────
async function enqueueAudio(wavBuffer) {
  playQueue.push(wavBuffer);
  if (!isPlaying) drainQueue();
}

async function drainQueue() {
  if (playQueue.length === 0) { isPlaying = false; return; }
  isPlaying = true;
  const buf = playQueue.shift();

  const playCtx = new AudioContext({ sampleRate: PLAY_SR });
  try {
    const decoded = await playCtx.decodeAudioData(buf);
    const source  = playCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(playCtx.destination);

    const now = Math.max(playCtx.currentTime, nextPlayTime);
    source.start(now);
    nextPlayTime = now + decoded.duration;

    source.onended = () => {
      playCtx.close();
      drainQueue();
    };
  } catch (e) {
    console.warn("Audio decode error:", e);
    playCtx.close();
    drainQueue();
  }
}

// ── Conversation UI ───────────────────────────────────────────────────────────
function hideWelcome() {
  welcomeMsg.style.display = "none";
}

function appendMessage(role, text) {
  const wrap   = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "👤" : "⚡";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  conversation.appendChild(wrap);
  conversation.scrollTop = conversation.scrollHeight;
  return bubble;
}

function appendAssistantToken(token) {
  if (!currentAssistantBubble) {
    currentAssistantBubble = appendMessage("assistant", "");
    currentAssistantBubble.classList.add("streaming");
  }
  currentAssistantBubble.textContent += token;
  conversation.scrollTop = conversation.scrollHeight;
}

function closeAssistantBubble() {
  if (currentAssistantBubble) {
    currentAssistantBubble.classList.remove("streaming");
    currentAssistantBubble = null;
  }
}

// ── Status indicator ──────────────────────────────────────────────────────────
function setStatus(state, label) {
  statusDot.className  = `status-dot ${state}`;
  statusText.textContent = label;
}

// ── Waveform visualizer ───────────────────────────────────────────────────────
function drawWave() {
  if (!analyser) return;
  requestAnimationFrame(drawWave);

  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);

  waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
  waveCtx.strokeStyle = "#5b8dee";
  waveCtx.lineWidth   = 1.5;
  waveCtx.beginPath();

  const step = waveCanvas.width / data.length;
  for (let i = 0; i < data.length; i++) {
    const y = (data[i] / 128.0) * (waveCanvas.height / 2);
    i === 0 ? waveCtx.moveTo(0, y) : waveCtx.lineTo(i * step, y);
  }
  waveCtx.stroke();
}

// ── Latency display ───────────────────────────────────────────────────────────
function showLatency(data) {
  latencyBar.style.display = "flex";
  document.getElementById("latASR").textContent   = `ASR ${data.asr_ms ?? "—"}ms`;
  document.getElementById("latRAG").textContent   = `RAG ${data.rag_ms ?? "—"}ms`;
  document.getElementById("latLLM").textContent   = `LLM ${data.llm_ms ?? "—"}ms`;
  document.getElementById("latTTS").textContent   = `TTS ${data.tts_first_ms ?? "—"}ms`;
  document.getElementById("latTotal").textContent = `Total ${data.total_ms ?? "—"}ms`;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
connectWS();
