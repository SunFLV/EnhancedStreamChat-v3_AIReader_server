"""
Lightweight proxy: simple TTS API → Intel GPT-SoVITS webui_api
No model loading. No modifications to the Intel project.

Requires: Intel project's webui_api.bat running on port 9876
"""

import io
import os
import time
import json
import logging
from pathlib import Path

import yaml
import numpy as np
import soundfile as sf
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tts-proxy")

# ── Config ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
with open(SCRIPT_DIR / "config_intel.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

BACKEND = config.get("backend", "http://127.0.0.1:9880")
POOL_SIZE = config.get("pool_size", 1)
BACKEND_BASE_PORT = config.get("backend_base_port", 9880)

# Backend pool management (voice-affinity + auto-restart on GPU OOM)
import subprocess
import threading
INTEL_ROOT = config.get("intel_root", r"D:\BSAIReader\GPT-SoVITS-For-Intel")
_backend_procs = {}  # port -> Popen
_pool_lock = threading.Lock()

def _port_in_use(port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except Exception:
        return False

def start_backend_instance(port):
    """Start one backend instance on the given port."""
    env = os.environ.copy()
    bin_dir = os.path.join(INTEL_ROOT, "runtime", "Library", "bin")
    ipex_bin = os.path.join(INTEL_ROOT, "runtime", "Lib", "site-packages",
                            "intel_extension_for_pytorch", "bin")
    env["PATH"] = f"{bin_dir};{ipex_bin};" + env.get("PATH", "")
    log_file = open(os.path.join(SCRIPT_DIR, f"backend_{port}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        [os.path.join(INTEL_ROOT, "runtime", "python.exe"),
         "api_v2.py", "-a", "0.0.0.0", "-p", str(port)],
        cwd=INTEL_ROOT, env=env,
        stdout=log_file, stderr=log_file,
    )
    _backend_procs[port] = proc
    logger.info("Backend %s starting (PID %s)", port, proc.pid)

def start_backend_pool():
    """Start all backend instances in the pool."""
    for i in range(POOL_SIZE):
        port = BACKEND_BASE_PORT + i
        if not _port_in_use(port):
            start_backend_instance(port)

def restart_backend(port):
    """Kill and restart one backend instance."""
    with _pool_lock:
        proc = _backend_procs.get(port)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not _port_in_use(port):
            start_backend_instance(port)


def _monitor_backends():
    """Daemon: periodically check each backend is alive, restart if dead.
    Fixes the case where a backend gets killed (OOM/external) and never
    comes back because the proxy only checks ports at startup."""
    import time as _t
    while True:
        _t.sleep(15)
        for i in range(POOL_SIZE):
            port = BACKEND_BASE_PORT + i
            if not _port_in_use(port):
                logger.warning("Backend %s is DEAD — restarting", port)
                try:
                    if not _port_in_use(port):
                        start_backend_instance(port)
                except Exception as e:
                    logger.error("Failed to restart backend %s: %s", port, e)

def backend_ready(port):
    """Wait until a backend instance responds."""
    url = f"http://127.0.0.1:{port}"
    for i in range(90):
        proc = _backend_procs.get(port)
        if proc and proc.poll() is not None:
            logger.error("Backend %s crashed (exit %s)", port, proc.poll())
            return False
        try:
            requests.post(f"{url}/tts", json={"text": "x", "text_lang": "zh"}, timeout=2)
            return True
        except requests.exceptions.ConnectionError:
            if i % 10 == 0:
                logger.info("Backend %s loading... (%ds)", port, i * 2)
            time.sleep(2)
        except Exception:
            return True
    return False

def voice_backend_port(voice):
    """Route voice to a backend instance (stable deterministic affinity)."""
    h = 0
    for ch in voice:
        h = (h * 31 + ord(ch)) & 0x7fffffff
    return BACKEND_BASE_PORT + (h % POOL_SIZE)
VOICES = {}
for v in config.get("voices", []):
    if v.get("name"):
        VOICES[v["name"]] = v
VOICES.setdefault("default", {"text_lang": "zh", "prompt_lang": "zh",
    "ref_audio_path": "", "prompt_text": ""})

# ── Speaker audio cache (name + voice + speed → WAV bytes) ────────
# Same viewer sends many messages; "名字说" is highly repeatable.
# Two layers: in-memory (fast) + disk (persistent across restarts).
_speaker_cache = {}
_speaker_cache_lock = threading.Lock()
MAX_SPEAKER_CACHE = 1000  # in-memory hot cache limit
SPEAKER_CACHE_DIR = SCRIPT_DIR / "speaker_cache"
SPEAKER_CACHE_DIR.mkdir(exist_ok=True)


def _speaker_cache_key(speaker, voice, speed):
    return f"{speaker}|{voice}|{speed}"


def _speaker_cache_path(speaker, voice, speed):
    """Disk path for a speaker's audio. Hash avoids illegal filename chars."""
    import hashlib
    key = _speaker_cache_key(speaker, voice, speed)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    # Voice + speed are safe-ish; speaker goes into hash
    return SPEAKER_CACHE_DIR / f"{voice}_{speed}_{h}.wav"


def _concat_wav(*wav_bytes_list):
    """Concatenate multiple WAV byte chunks into one WAV (same sample rate)."""
    chunks = []
    sr = None
    for wav_bytes in wav_bytes_list:
        data, s = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if sr is None:
            sr = s
        elif s != sr:
            # Resample is complex; just log and skip mismatch chunk
            logger.warning("Sample rate mismatch %s vs %s, skipping chunk", s, sr)
            continue
        chunks.append(data)
    if not chunks:
        return None, sr
    merged = np.concatenate(chunks)
    out = io.BytesIO()
    sf.write(out, merged, sr, format="WAV", subtype="PCM_16")
    out.seek(0)
    return out, sr


def _get_cached_speaker(speaker, voice, speed):
    key = _speaker_cache_key(speaker, voice, speed)
    # 1. In-memory cache
    with _speaker_cache_lock:
        if key in _speaker_cache:
            return _speaker_cache[key]
    # 2. Disk cache
    path = _speaker_cache_path(speaker, voice, speed)
    if path.exists():
        try:
            wav_bytes = path.read_bytes()
            with _speaker_cache_lock:
                _speaker_cache[key] = wav_bytes  # warm the in-memory cache
            return wav_bytes
        except Exception as e:
            logger.warning("Failed to read speaker cache %s: %s", path, e)
    return None


def _cache_speaker(speaker, voice, speed, wav_bytes):
    key = _speaker_cache_key(speaker, voice, speed)
    # 1. In-memory cache (with simple eviction)
    with _speaker_cache_lock:
        if len(_speaker_cache) >= MAX_SPEAKER_CACHE:
            _speaker_cache.pop(next(iter(_speaker_cache)))
        _speaker_cache[key] = wav_bytes
    # 2. Disk cache (persistent)
    path = _speaker_cache_path(speaker, voice, speed)
    try:
        path.write_bytes(wav_bytes)
    except Exception as e:
        logger.warning("Failed to write speaker cache %s: %s", path, e)

# Language mapping
LANG_MAP = {"zh": "中文", "ja": "日文", "en": "英文", "yue": "粤语", "ko": "韩文",
            "all_zh": "all_zh", "all_ja": "all_ja"}

# Global serial lock — one inference at a time across all backends.
# Parallel instances fragment GPU and hurt quality, so requests queue
# globally; a hot standby instance takes over only if the active one OOMs.
_tts_lock = threading.Lock()

# ── Long-text task management ────────────────────────────────────
# For audiobook-style synthesis: split text into sentences, synthesize
# each sequentially, report progress, then merge into one WAV.
import uuid

_tts_tasks = {}
_tts_tasks_lock = threading.Lock()
MAX_TASKS = 50
TASK_TTL = 43200  # seconds — tasks + WAV files auto-expire after 12 hours
TASK_CACHE_DIR = SCRIPT_DIR / "task_cache"
TASK_CACHE_DIR.mkdir(exist_ok=True)


def _save_task(task_id, task):
    """Persist a task to disk so it survives proxy crashes/restarts."""
    try:
        # WAV file
        wav = task.get("wav")
        if wav:
            (TASK_CACHE_DIR / f"{task_id}.wav").write_bytes(wav)
        # Metadata (excludes wav blob)
        meta = {k: v for k, v in task.items() if k != "wav"}
        (TASK_CACHE_DIR / f"{task_id}.json").write_text(
            json.dumps(meta), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to persist task %s: %s", task_id, e)


def _load_tasks_from_disk():
    """Load tasks from disk on startup. Incomplete tasks resume synthesis
    from the last completed sentence (crash recovery / resume)."""
    import time as _t
    now = _t.time()
    loaded = resumed = 0
    for jf in TASK_CACHE_DIR.glob("*.json"):
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
            task_id = jf.stem
            wav_path = TASK_CACHE_DIR / f"{task_id}.wav"
            meta["created"] = meta.get("created", now)
            if now - meta["created"] > TASK_TTL:
                _del_task_files(task_id)
                continue

            if meta.get("status") == "done" and wav_path.exists():
                meta["wav"] = wav_path.read_bytes()
                _tts_tasks[task_id] = meta
            elif meta.get("status") in ("running", "queued"):
                # Crash recovery: resume from where we left off
                meta["wav"] = None
                meta["status"] = "running"
                _tts_tasks[task_id] = meta
                threading.Thread(
                    target=_run_long_task,
                    args=(task_id, meta.get("text", ""), meta.get("voice", "default"),
                          meta.get("speed", 1.0)),
                    daemon=True,
                ).start()
                resumed += 1
            else:
                # failed/error with no fragments — mark error
                meta["wav"] = None
                meta["status"] = "error"
                meta["error"] = "Proxy restarted before task completed"
                _tts_tasks[task_id] = meta
            loaded += 1
        except Exception as e:
            logger.warning("Failed to load task %s from disk: %s", jf.name, e)
    if loaded:
        logger.info("Recovered %d task(s) from disk cache (%d resumed)", loaded, resumed)


def _del_task_files(task_id):
    (TASK_CACHE_DIR / f"{task_id}.json").unlink(missing_ok=True)
    (TASK_CACHE_DIR / f"{task_id}.wav").unlink(missing_ok=True)
    for f in TASK_CACHE_DIR.glob(f"{task_id}.*.wav"):
        f.unlink(missing_ok=True)  # sentence fragments


def _cleanup_tasks():
    """Periodically delete expired tasks (and trim to MAX_TASKS)."""
    import time as _t
    while True:
        _t.sleep(120)
        now = _t.time()
        with _tts_tasks_lock:
            expired = [k for k, v in _tts_tasks.items()
                       if now - v.get("created", now) > TASK_TTL]
            for k in expired:
                _del_task_files(k)
                del _tts_tasks[k]
            # also cap count
            while len(_tts_tasks) > MAX_TASKS:
                k = next(iter(_tts_tasks))
                _del_task_files(k)
                del _tts_tasks[k]


def _split_sentences(text):
    """Split long text into speakable sentences."""
    import re
    # Split on Chinese/English sentence terminators, keep them attached
    parts = re.split(r'(?<=[。！？!?；;\n])', text)
    sents = [p.strip() for p in parts if p.strip()]
    # Further split any sentence over 200 chars
    result = []
    for s in sents:
        while len(s) > 200:
            cut = s[:200].rfind('，')
            if cut < 50:
                cut = 200
            result.append(s[:cut])
            s = s[cut:]
        if s:
            result.append(s)
    return result


def _new_task(text, voice, speed):
    task_id = uuid.uuid4().hex[:12]
    key = uuid.uuid4().hex[:16]  # access token — required to fetch result
    import time as _t
    with _tts_tasks_lock:
        if len(_tts_tasks) >= MAX_TASKS:
            for k in list(_tts_tasks):
                if _tts_tasks[k]["status"] in ("done", "error"):
                    del _tts_tasks[k]
        _tts_tasks[task_id] = {
            "status": "queued", "total": 0, "done": 0,
            "error": None, "wav": None, "voice": voice, "speed": speed,
            "text": text, "key": key, "created": _t.time(),
        }
    return task_id, key


def _run_long_task(task_id, text, voice, speed):
    """Background worker: synthesize each sentence, merge, store result.
    Supports crash recovery — each completed sentence is persisted to disk
    (fragment WAV + progress), so a restart resumes from the last sentence."""
    try:
        sents = _split_sentences(text)
        total = len(sents)
        with _tts_tasks_lock:
            task = _tts_tasks[task_id]
            task["total"] = total
            task["status"] = "running"
            task["text"] = text
            start = task["done"]  # resume point (0 for fresh tasks)
        _save_task(task_id, dict(_tts_tasks[task_id]))

        # Load fragments already synthesized before a crash
        chunks = []
        for i in range(start):
            f = TASK_CACHE_DIR / f"{task_id}.{i}.wav"
            if f.exists():
                chunks.append(f.read_bytes())

        MAX_RETRIES = 3
        for i in range(start, total):
            ok = False
            for attempt in range(MAX_RETRIES):
                try:
                    wav = _synthesize(sents[i], voice, speed)
                    chunks.append(wav)
                    # Persist this sentence's audio immediately (crash-proof)
                    (TASK_CACHE_DIR / f"{task_id}.{i}.wav").write_bytes(wav)
                    ok = True
                    break
                except HTTPException as e:
                    logger.warning("Long-text: sentence %d attempt %d/%d failed: %s",
                                   i + 1, attempt + 1, MAX_RETRIES, e.detail)
                    time.sleep(1)
            if not ok:
                logger.warning("Long-text: sentence %d SKIPPED after %d attempts", i + 1, MAX_RETRIES)
            with _tts_tasks_lock:
                _tts_tasks[task_id]["done"] = i + 1
                _save_task(task_id, dict(_tts_tasks[task_id]))  # progress per sentence

        merged, _ = _concat_wav(*chunks) if chunks else (None, None)
        with _tts_tasks_lock:
            task = _tts_tasks[task_id]
            if merged:
                task["wav"] = merged.getvalue()
            task["status"] = "done" if merged else "error"
            if not merged:
                task["error"] = "No sentences synthesized"
            _save_task(task_id, dict(task))
        # Clean up sentence fragments after successful merge
        for i in range(total):
            (TASK_CACHE_DIR / f"{task_id}.{i}.wav").unlink(missing_ok=True)
        logger.info("Long-text task %s done: %d/%d sentences", task_id, task["done"], task["total"])
    except Exception as e:
        logger.error("Long-text task %s failed: %s", task_id, e)
        with _tts_tasks_lock:
            _tts_tasks[task_id]["status"] = "error"
            _tts_tasks[task_id]["error"] = str(e)
            _save_task(task_id, dict(_tts_tasks[task_id]))


# ── FastAPI ──────────────────────────────────────────────────────
app = FastAPI(title="AI TTS Proxy", version="3.0.0")


# ── Web UI ───────────────────────────────────────────────────────
WEB_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 语音朗读</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
  min-height: 100vh; display:flex; align-items:center; justify-content:center;
  color:#eee;
}
.card {
  background: rgba(255,255,255,0.06); backdrop-filter: blur(12px);
  border:1px solid rgba(255,255,255,0.12); border-radius:20px;
  padding:32px; width:100%; max-width:560px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
h1 { text-align:center; font-size:24px; margin-bottom:24px;
     background: linear-gradient(90deg,#a78bfa,#f472b6); -webkit-background-clip:text;
     -webkit-text-fill-color:transparent; }
label { display:block; font-size:13px; color:#aaa; margin:14px 0 6px; }
input[type=text], textarea, select {
  width:100%; padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.15);
  background: rgba(0,0,0,0.3); color:#eee; font-size:15px; outline:none;
}
textarea { resize:vertical; min-height:90px; }
input:focus, textarea:focus, select:focus { border-color:#a78bfa; }
.row { display:flex; gap:14px; }
.row > div { flex:1; }
select option { background:#1a1a2e; }
input[type=range] { width:100%; accent-color:#a78bfa; }
.speed-val { color:#a78bfa; font-weight:bold; }
button {
  width:100%; margin-top:20px; padding:13px; border:none; border-radius:12px;
  background: linear-gradient(90deg,#7c3aed,#ec4899); color:#fff; font-size:16px;
  font-weight:600; cursor:pointer; transition: opacity .2s;
}
button:hover { opacity:.85; } button:disabled { opacity:.4; cursor:not-allowed; }
#status { text-align:center; margin-top:12px; font-size:13px; min-height:18px; color:#a78bfa; }
#player { width:100%; margin-top:16px; display:none; }
audio { width:100%; }
.voices-note { text-align:center; font-size:12px; color:#666; margin-top:8px; }
.progress-wrap { margin-top:12px; display:none; }
.progress-bar {
  height:8px; border-radius:4px; background:rgba(255,255,255,0.1); overflow:hidden;
}
.progress-fill {
  height:100%; width:0%; border-radius:4px;
  background: linear-gradient(90deg,#7c3aed,#ec4899); transition:width .3s;
}
.progress-text { text-align:center; font-size:12px; color:#aaa; margin-top:6px; }
.btn-sec {
  margin-top:10px; padding:10px; border-radius:12px; border:1px solid rgba(255,255,255,0.2);
  background:rgba(124,58,237,0.2); color:#d8b4fe; font-size:14px; font-weight:600;
  cursor:pointer; transition:opacity .2s;
}
.btn-sec:hover { opacity:.85; } .btn-sec:disabled { opacity:.4; cursor:not-allowed; }
.tabs { display:flex; gap:8px; margin-bottom:18px; }
.tab-btn {
  flex:1; padding:10px; border:none; border-radius:10px; font-size:14px; font-weight:600;
  background:rgba(255,255,255,0.06); color:#aaa; cursor:pointer; transition:all .2s;
}
.tab-btn.active {
  background:linear-gradient(90deg,#7c3aed,#ec4899); color:#fff;
}
.tab-pane { display:none; }
.tab-pane.active { display:block; }
</style>
</head>
<body>
<div class="card">
  <h1>🎙 AI 语音朗读</h1>
  <div class="tabs">
    <button class="tab-btn active" id="tabShort" onclick="switchTab('short')">弹幕朗读</button>
    <button class="tab-btn" id="tabLong" onclick="switchTab('long')">📖 长文合成</button>
  </div>

  <!-- 弹幕朗读 tab -->
  <div class="tab-pane active" id="pane-short">
    <label>弹幕文本</label>
    <textarea id="text" placeholder="输入要朗读的文本，支持中英混合…">欢迎来到直播间</textarea>
    <div class="row">
      <div>
        <label>声音</label>
        <select id="voice"></select>
      </div>
      <div>
        <label>语速 <span class="speed-val" id="speedVal">1.0x</span></label>
        <input type="range" id="speed" min="0.5" max="2.0" step="0.1" value="1.0">
      </div>
    </div>
    <label>说话人（可选，会拼接"xxx说"）</label>
    <input type="text" id="speaker" placeholder="留空则只读文本内容">
    <button id="go">合成并播放</button>
  </div>

  <!-- 长文合成 tab -->
  <div class="tab-pane" id="pane-long">
    <label>小说文本（支持整章粘贴，自动分段）</label>
    <textarea id="longText" placeholder="粘贴整本/整章文本…（支持长文，自动分段）"></textarea>
    <div class="row">
      <div>
        <label>声音</label>
        <select id="voiceLong"></select>
      </div>
      <div>
        <label>语速 <span class="speed-val" id="speedValLong">1.0x</span></label>
        <input type="range" id="speedLong" min="0.5" max="2.0" step="0.1" value="1.0">
      </div>
    </div>
    <button id="longBtn">开始合成长文</button>
    <button class="btn-sec" id="showRecover" style="margin-top:8px;">🔁 我有任务ID/密钥，恢复进度</button>
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-text" id="progressText"></div>
    </div>
    <div class="task-info" id="taskInfo" style="display:none; margin-top:12px; padding:10px; border-radius:10px; background:rgba(0,0,0,0.25); font-size:12px; line-height:1.8; word-break:break-all;">
      <div>📋 任务ID: <span id="taskIdText"></span></div>
      <div>🔑 密钥: <span id="taskKeyText" style="color:#a78bfa;"></span></div>
      <div style="color:#888;">⚠️ 断网/刷新后可凭任务ID和密钥恢复（12小时内有效）</div>
    </div>
    <div class="recover-box" id="recoverBox" style="display:none; margin-top:12px; padding:10px; border-radius:10px; background:rgba(0,0,0,0.25);">
      <label>恢复进度</label>
      <input type="text" id="recoverId" placeholder="任务ID">
      <input type="text" id="recoverKey" placeholder="密钥" style="margin-top:6px;">
      <button class="btn-sec" id="recoverBtn" style="margin-top:6px;">🔄 恢复进度</button>
    </div>
  </div>

  <div id="status"></div>
  <div id="player"><audio id="audio" controls></audio></div>
  <div class="voices-note" id="note"></div>
</div>

<script>
const $ = id => document.getElementById(id);
const voiceSel = $('voice'), status = $('status'), player = $('player'), audio = $('audio');
const voiceLong = $('voiceLong');

function switchTab(name) {
  $('tabShort').classList.toggle('active', name === 'short');
  $('tabLong').classList.toggle('active', name === 'long');
  $('pane-short').classList.toggle('active', name === 'short');
  $('pane-long').classList.toggle('active', name === 'long');
}

async function loadVoices() {
  try {
    const r = await fetch('/api/voices');
    const data = await r.json();
    const opts = data.voices.map(v =>
      `<option value="${v.name}">${v.name} — ${v.description || ''}</option>`).join('');
    voiceSel.innerHTML = opts;
    voiceLong.innerHTML = opts;
    $('note').textContent = `${data.voices.length} 个声音可用`;
  } catch(e) {
    status.textContent = '加载声音列表失败: ' + e;
  }
}

$('speed').oninput = e => $('speedVal').textContent = (+e.target.value).toFixed(1) + 'x';
$('speedLong').oninput = e => $('speedValLong').textContent = (+e.target.value).toFixed(1) + 'x';

const progressWrap = $('progressWrap'), progressFill = $('progressFill'), progressText = $('progressText');
let pollTimer = null;

function showProgress(pct, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width = pct + '%';
  progressText.textContent = label;
}

let taskKey = null;

async function pollTask(taskId) {
  clearTimeout(pollTimer);
  try {
    const r = await fetch(`/api/tts/progress/${taskId}?key=${taskKey}`);
    const p = await r.json();
    const pct = p.total ? Math.round(p.done / p.total * 100) : 0;
    showProgress(pct, `正在合成 ${p.done}/${p.total} 句（${pct}%）…`);

    if (p.status === 'done') {
      showProgress(100, '✅ 长文合成完成');
      audio.src = `/api/tts/result/${taskId}?key=${taskKey}`;
      player.style.display = 'block';
      audio.play();
      longBtn.disabled = false; go.disabled = false;
      return;
    }
    if (p.status === 'error') {
      showProgress(0, '❌ 合成失败: ' + (p.error || '未知错误'));
      longBtn.disabled = false; go.disabled = false;
      return;
    }
    pollTimer = setTimeout(() => pollTask(taskId), 800);
  } catch(e) {
    showProgress(0, '❌ 轮询失败: ' + e);
    longBtn.disabled = false; go.disabled = false;
  }
}

$('go').onclick = async () => {
  const text = $('text').value.trim();
  if (!text) { status.textContent = '请输入文本'; return; }
  const btn = $('go'); btn.disabled = true;
  status.textContent = '合成中，请稍候…';
  try {
    const r = await fetch('/api/tts', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        speaker: $('speaker').value.trim(),
        text, voice: voiceSel.value,
        speed: parseFloat($('speed').value)
      })
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err.slice(0, 150));
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    audio.src = url;
    player.style.display = 'block';
    status.textContent = '✅ 合成完成（' + (r.headers.get('X-TTS-Time') || '?') + 's）';
    audio.play();
  } catch(e) {
    status.textContent = '❌ ' + e;
  } finally {
    btn.disabled = false;
  }
};

let currentTaskId = null;

function showTaskInfo(taskId, key) {
  currentTaskId = taskId;
  $('taskIdText').textContent = taskId;
  $('taskKeyText').textContent = key;
  $('taskInfo').style.display = 'block';
}

async function recoverTask() {
  const id = $('recoverId').value.trim();
  const k = $('recoverKey').value.trim();
  if (!id || !k) {
    status.textContent = '请输入任务ID和密钥';
    return;
  }
  taskKey = k;
  showTaskInfo(id, k);
  $('recoverBox').style.display = 'none';
  longBtn.disabled = true; go.disabled = true;
  showProgress(1, '正在查询任务进度…');
  pollTask(id);
}

$('recoverBtn').onclick = recoverTask;

$('showRecover').onclick = () => {
  const box = $('recoverBox');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  if (box.style.display === 'block') {
    $('recoverId').value = currentTaskId || '';
    $('recoverKey').value = taskKey || '';
  }
};

$('longBtn').onclick = async () => {
  const text = $('longText').value.trim();
  if (!text) { status.textContent = '请输入文本'; return; }
  if (text.length < 50) {
    status.textContent = '长文模式建议输入 50 字以上（适合整章/整段文本）';
  }
  longBtn.disabled = true; go.disabled = true;
  showProgress(1, '正在提交长文任务…');
  try {
    const r = await fetch('/api/tts/long', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        text, voice: voiceLong.value,
        speed: parseFloat($('speedLong').value)
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || '提交失败');
    taskKey = data.key;
    showTaskInfo(data.task_id, data.key);
    pollTask(data.task_id);
  } catch(e) {
    showProgress(0, '❌ 提交失败: ' + e);
    longBtn.disabled = false; go.disabled = false;
  }
};

loadVoices();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return WEB_HTML


@app.get("/api/health")
async def health():
    try:
        r = requests.get(f"{BACKEND}/", timeout=5)
        return {"status": "ok", "backend": BACKEND, "voices": len(VOICES)}
    except Exception:
        return {"status": "degraded", "backend": "unreachable", "voices": len(VOICES)}


@app.get("/api/voices")
async def voices():
    return {"voices": [{"name": k, "description": v.get("description", "")}
                       for k, v in VOICES.items()]}


class TTSRequest(BaseModel):
    speaker: str = ""  # 发送者名字，可为空（测试/系统消息）
    text: str = Field(..., min_length=1, max_length=200)
    voice: str = "default"
    speed: float = Field(1.0, ge=0.5, le=2.0)


def _candidate_ports(voice):
    """Voice-affinity port first, then the rest of the pool (for failover)."""
    affinity = voice_backend_port(voice)
    others = [BACKEND_BASE_PORT + i for i in range(POOL_SIZE)
              if BACKEND_BASE_PORT + i != affinity]
    return [affinity] + others


def _synthesize(text, voice, speed):
    """Synthesize one text chunk via backend, with OOM failover.
    Tries voice-affinity backend first; on OOM/unreachable, falls back
    to the next healthy instance while restarting the failed one in background."""
    v = VOICES.get(voice, VOICES["default"])
    payload = {
        "text": text,
        "text_lang": v.get("text_lang", "zh"),
        "ref_audio_path": v.get("ref_audio_path", ""),
        "prompt_text": v.get("prompt_text", ""),
        "prompt_lang": v.get("prompt_lang", "zh"),
        "speed_factor": speed,
        "text_split_method": "cut5",
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "seed": -1,
        "batch_size": 1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "streaming_mode": False,
    }

    # Serialize globally: one inference at a time.
    # If the active instance OOMs, fall through to the hot standby.
    with _tts_lock:
        for port in _candidate_ports(voice):
            backend_url = f"http://127.0.0.1:{port}"
            try:
                r = requests.post(f"{backend_url}/tts", json=payload, timeout=300)
            except requests.exceptions.ConnectionError:
                logger.warning("Backend %s unreachable — failover to next", port)
                continue
            if r.status_code == 200:
                return r.content
            err = r.text
            if "OUT_OF_DEVICE_MEMORY" in err or "OUT_OF_RESOURCES" in err:
                logger.warning("Backend %s GPU OOM — failover to standby, restarting in background", port)
                threading.Thread(target=restart_backend, args=(port,), daemon=True).start()
                continue
            raise HTTPException(502, f"Backend error {r.status_code}: {err[:200]}")

    raise HTTPException(503, "All backends unavailable. Please retry.")


@app.post("/api/tts")
def tts(req: TTSRequest):
    # NOTE: sync def (not async) — FastAPI runs sync endpoints in a thread pool,
    # so blocking requests.post here does NOT block other concurrent requests.
    # async def would serialize all requests on the event loop.
    voice = req.voice
    t0 = time.time()

    try:
        # Speaker audio (cached) — "名字说" is highly repeatable
        speaker_wav = None
        if req.speaker:
            speaker_wav = _get_cached_speaker(req.speaker, voice, req.speed)
            if speaker_wav is None:
                speaker_text = f"{req.speaker}说"
                speaker_wav = _synthesize(speaker_text, voice, req.speed)
                _cache_speaker(req.speaker, voice, req.speed, speaker_wav)

        # Text audio (always generated, with failover handled inside)
        text_wav = _synthesize(req.text, voice, req.speed)

        # Merge: speaker + text
        if speaker_wav:
            merged, sr = _concat_wav(speaker_wav, text_wav)
            elapsed = time.time() - t0
            logger.info(f"TTS: voice={voice} speaker={req.speaker} len={len(req.text)} {elapsed:.1f}s")
            return StreamingResponse(merged, media_type="audio/wav",
                headers={"X-TTS-Time": f"{elapsed:.1f}"})
        else:
            elapsed = time.time() - t0
            size_kb = len(text_wav) / 1024
            logger.info(f"TTS: voice={voice} len={len(req.text)} {elapsed:.1f}s {size_kb:.0f}KB")
            return StreamingResponse(io.BytesIO(text_wav), media_type="audio/wav",
                headers={"X-TTS-Time": f"{elapsed:.1f}"})
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, f"Backend {voice_backend_port(voice)} not running.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, str(e))


class LongTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)  # 100k chars ≈ full novel
    voice: str = "default"
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.post("/api/tts/long")
def tts_long(req: LongTTSRequest):
    """Start async long-text synthesis. Returns task_id + access key."""
    task_id, key = _new_task(req.text, req.voice, req.speed)
    threading.Thread(target=_run_long_task,
                     args=(task_id, req.text, req.voice, req.speed),
                     daemon=True).start()
    return {"task_id": task_id, "key": key}


@app.get("/api/tts/progress/{task_id}")
def tts_progress(task_id: str, key: str = None):
    with _tts_tasks_lock:
        t = _tts_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found or expired")
    if t["key"] != key:
        raise HTTPException(403, "Invalid key")
    return {"status": t["status"], "done": t["done"], "total": t["total"],
            "error": t["error"]}


@app.get("/api/tts/result/{task_id}")
def tts_result(task_id: str, key: str = None):
    with _tts_tasks_lock:
        t = _tts_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found or expired")
    if t["key"] != key:
        raise HTTPException(403, "Invalid key")
    if t["status"] == "running":
        raise HTTPException(409, "Still synthesizing")
    if t["status"] == "error":
        raise HTTPException(500, t["error"] or "Synthesis failed")
    if not t["wav"]:
        raise HTTPException(500, "No audio produced")
    return StreamingResponse(io.BytesIO(t["wav"]), media_type="audio/wav")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, threading
    sc = config.get("server", {})
    host = sc.get("host", "0.0.0.0")
    listeners = sc.get("listeners", [{"port": sc.get("port", 8000)}])

    # Load long-text tasks from disk (survive proxy restart)
    _load_tasks_from_disk()

    # Daemons: task cleanup + backend keep-alive
    threading.Thread(target=_cleanup_tasks, daemon=True).start()
    threading.Thread(target=_monitor_backends, daemon=True).start()

    # Start backend pool
    logger.info("Starting %s backend instance(s)...", POOL_SIZE)
    start_backend_pool()
    for i in range(POOL_SIZE):
        port = BACKEND_BASE_PORT + i
        if backend_ready(port):
            logger.info("Backend %s ready.", port)
        else:
            logger.warning("Backend %s not ready yet.", port)

    def serve(lst):
        """Serve one listener. host '::' = IPv6, '0.0.0.0' = IPv4."""
        import socket as _socket
        host = lst.get("host", "0.0.0.0")
        port = lst["port"]
        ssl_cert = lst.get("ssl_cert", "")
        ssl_key = lst.get("ssl_key", "")

        # Create a bound socket first so IPv4/IPv6 can share the same port
        if host == "::":
            sock = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            sock.setsockopt(_socket.IPPROTO_IPV6, _socket.IPV6_V6ONLY, 1)  # IPv6 only
        else:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(2048)

        kw = {}
        if ssl_cert and ssl_key and os.path.exists(str(ssl_cert)) and os.path.exists(str(ssl_key)):
            kw = {"ssl_keyfile": str(ssl_key), "ssl_certfile": str(ssl_cert)}
            logger.info(f"https://[{host}]:{port} → {BACKEND}")
        else:
            logger.info(f"http://[{host}]:{port} → {BACKEND}")

        # Use Server + pre-bound socket (uvicorn.run doesn't accept sock)
        server = uvicorn.Server(uvicorn.Config(app, **kw))
        server.run(sockets=[sock])

    for i, lst in enumerate(listeners):
        t = threading.Thread(target=serve, args=(lst,), daemon=True)
        t.start()
        logger.info(f"Listener {i+1}: {lst.get('host','0.0.0.0')}:{lst['port']}")

    # Keep main thread alive
    import time
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down")
