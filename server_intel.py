"""
Lightweight proxy: simple TTS API → Intel GPT-SoVITS api_v2 backends.

The proxy spawns api_v2.py instances itself and manages them LAZILY:
a backend's model (~90 s to load, one full copy per process) is started only
on the first synthesis that needs it and stopped again when idle, so boot
loads nothing and two backends are never resident at once.
"""

import io
import os
import time
import json
import base64
import logging
from pathlib import Path

import yaml
import numpy as np
import soundfile as sf
import requests
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tts-proxy")

# ── Config ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
# 独立附加包模式：解压到后端根目录后，后端就是 SCRIPT_DIR 的父目录。
# BSAI_BACKEND_DIR 可显式覆盖；不设时按「父目录有 runtime/python.exe + api_v2.py」
# 自动判断。这样同一份代码既能放在 server/ 下用，也能整包丢进后端根目录。
def _detect_backend_dir(base):
    for cand in (os.environ.get("BSAI_BACKEND_DIR"), str(base.parent), str(base)):
        if not cand:
            continue
        p = Path(cand)
        if (p / "runtime" / "python.exe").exists() and (p / "api_v2.py").exists():
            return str(p)
    return None

BACKEND_DIR = _detect_backend_dir(SCRIPT_DIR)
# BSAI_CONFIG 可指向另一份配置：便于开一个「不影响线上」的测试实例
# （不同监听端口 + 不同 backend_base_port，共用同一套代码）。
CONFIG_PATH = Path(os.environ.get("BSAI_CONFIG") or (SCRIPT_DIR / "config_intel.yaml"))
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config_text = f.read()
# 配置里的 {BACKEND_DIR} / {ADDON_DIR} 占位符在加载时展开 —— 换机器只需解压，
# 不用改一整份绝对路径。BACKEND_DIR 未探测到时留空串，配置可自行写绝对路径。
_config_text = (_config_text
                .replace("{BACKEND_DIR}", (BACKEND_DIR or "").replace("\\", "/"))
                .replace("{ADDON_DIR}", str(SCRIPT_DIR).replace("\\", "/")))
config = yaml.safe_load(_config_text)

BACKEND = config.get("backend", "http://127.0.0.1:9880")
POOL_SIZE = config.get("pool_size", 1)
BACKEND_BASE_PORT = config.get("backend_base_port", 9880)

# ── Backend pool: lazy, single-resident model loading ───────────────
# api_v2.py loads its model at process start and binds its HTTP port only
# AFTER the ~90 s load, so "process responding" == "model resident" and
# "process stopped" == "model fully released". To save XPU/RAM we therefore:
#   * start NO backend at boot — a backend is started (and its model loaded)
#     only on the first synthesis that actually needs it (lazy),
#   * stop a hot backend again once it sits idle (pool_idle_ttl),
#   * serialize model loads (global warm lock) so two backends never load at
#     the same instant, and load a cold one only after releasing idle peers.
# State: cold (no proc) / warming (proc up, model loading) / hot (responding).
import subprocess
import threading
import time
INTEL_ROOT = (config.get("intel_root") or BACKEND_DIR
              or r"D:\BSAIReader\GPT-SoVITS-For-Intel")
INTEL_ROOT = os.path.normpath(INTEL_ROOT)

# ── Model profiles: one backend config per selectable version ─────
# The frontend sends a `version`; each profile gets its own generated
# tts_infer YAML, so `api_v2.py -c <profile>.yaml` loads the matching
# SoVITS/GPT pair. Ports are dedicated per profile so a hot v4 process
# can never serve a v5turbo request.
TTS_BASE_CFG = config.get("tts", {}) or {}
DEFAULT_MODEL = config.get("default_model", "v5turbo")
MODEL_PROFILES = config.get("models", {}) or {}
if not MODEL_PROFILES:
    MODEL_PROFILES = {
        DEFAULT_MODEL: {
            "version": TTS_BASE_CFG.get("version", "v5turbo"),
            "label": DEFAULT_MODEL,
            "t2s_weights_path": TTS_BASE_CFG.get("t2s_weights_path"),
            "vits_weights_path": TTS_BASE_CFG.get("vits_weights_path"),
            "sample_steps": 4 if "v5" in str(TTS_BASE_CFG.get("version", "")) else 32,
            "is_half": TTS_BASE_CFG.get("is_half", True),
        }
    }
MODEL_KEYS = list(MODEL_PROFILES.keys())
if DEFAULT_MODEL not in MODEL_PROFILES:
    DEFAULT_MODEL = MODEL_KEYS[0]


def _probe_backend_capabilities():
    """这台机器上的后端**实际**支持哪些版本 —— 不信配置，信后端自己。

    两个来源，取交集：
      1. Patcher 应用补丁后写的 `BSAI_PATCH.json`（machine-readable 能力清单，
         未来补丁只装 v4 就会写 v4，服务端据此不再对外宣称 v5turbo）；
      2. 后端 `GPT_SoVITS/configs/tts_infer.yaml` 里真实存在的版本段
         （v1/v2/v3/v4/v5turbo/v5dev），旧版后端没有 v5 段就不会被点亮。

    两个来源都没有时返回 None（= 不限制，按配置来），保持向后兼容。
    """
    try:
        root = Path(INTEL_ROOT)
    except Exception:
        return None
    declared = None
    patch_json = root / "BSAI_PATCH.json"
    if patch_json.exists():
        try:
            j = json.loads(patch_json.read_text(encoding="utf-8"))
            v = j.get("models") or j.get("versions")
            if isinstance(v, list) and v:
                declared = [str(x) for x in v]
        except Exception as e:
            logger.warning("BSAI_PATCH.json unreadable: %s", e)

    segments = None
    infer_yaml = root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if infer_yaml.exists():
        try:
            y = yaml.safe_load(infer_yaml.read_text(encoding="utf-8")) or {}
            known = {"v1", "v2", "v3", "v4", "v5turbo", "v5dev"}
            segments = [k for k in y.keys() if str(k) in known]
        except Exception as e:
            logger.warning("tts_infer.yaml unreadable: %s", e)

    if declared and segments:
        return [v for v in declared if v in segments] or None
    return declared or segments or None


BACKEND_CAPS = _probe_backend_capabilities()
if BACKEND_CAPS:
    _supported = set(BACKEND_CAPS)
    _kept = {k: v for k, v in MODEL_PROFILES.items()
             if str(v.get("version") or k) in _supported}
    if _kept:
        MODEL_PROFILES = _kept
        MODEL_KEYS = list(MODEL_PROFILES.keys())
        if DEFAULT_MODEL not in MODEL_PROFILES:
            DEFAULT_MODEL = MODEL_KEYS[0]
    else:
        logger.warning("后端声明的版本 %s 与本机配置 %s 无交集 —— 按配置继续，"
                       "但合成很可能失败，请检查补丁安装", BACKEND_CAPS, list(MODEL_KEYS))
# ── 并行推理开关：决定实例池策略 ────────────────────────────────
# parallel_infer=true (默认): 强制单实例 —— 无论请求什么版本，都只用默认模型
#   的 1 个实例，靠 api_v2 的 /tts_batch 把多句拼成一次前向一起解码。
#   其他模型 profile 直接忽略，绝不会因为"模型不同"而启动第二个实例。
#   显存省（单份 ~2.2 GiB）。
# parallel_infer=false: 保持原状 —— 每个模型 ports_per_model 个实例并发，
#   实例内改用串行推理（payload parallel_infer=False, batch_size=1）。
PARALLEL_INFER = bool(config.get("parallel_infer", True))
BATCH_SIZE = int(config.get("parallel_batch_size", 4)) if PARALLEL_INFER else 1
# 批内长度容忍度：AR 是所有序列同步走一步，批里只要有一句特别长，短句就得
# padding 陪跑到最后（通道空转）。只把长度在 [1/ratio, ratio] 之间的句子凑
# 一批，离群的句子自己一批。ratio<=1 表示不约束（对比测试用）。
# 实测（16 句自然文本，A770）：不约束 127.1s；ratio=2 143.1s（批被切碎，更慢）；
# ratio=3 118.0s。2 倍太紧，3 倍既能保住批大小、又能把「4 字 vs 67 字」这种
# 极端组合拆开，所以默认 3.0。
BATCH_LEN_RATIO = float(config.get("parallel_batch_len_ratio", 3.0) or 3.0)
if BATCH_LEN_RATIO > 1.0:
    BATCH_LEN_RATIO_LO = 1.0 / BATCH_LEN_RATIO
    BATCH_LEN_RATIO_HI = BATCH_LEN_RATIO
else:
    BATCH_LEN_RATIO_LO, BATCH_LEN_RATIO_HI = 0.0, float("inf")
# dynamic_batch=true：后端槽位一空就补下一句（真正的「接活」）。此时批内长度
# 不必相近（短的做完立刻换新的），服务端一次提交 queued 句，由后端自己控并发。
DYNAMIC_BATCH = bool(config.get("dynamic_batch", True)) and PARALLEL_INFER
QUEUE_SIZE = int(config.get("parallel_queue_size", 32)) if DYNAMIC_BATCH else BATCH_SIZE
# Force-single-instance: collapse all profiles to the default one so no
# version the frontend sends can ever select a second model/port. This is the
# root fix for the OOM seen when a different version spun up a 2nd process.
if PARALLEL_INFER and DEFAULT_MODEL in MODEL_PROFILES:
    MODEL_PROFILES = {DEFAULT_MODEL: MODEL_PROFILES[DEFAULT_MODEL]}
    MODEL_KEYS = [DEFAULT_MODEL]
MODEL_CONFIG_DIR = SCRIPT_DIR / "model_configs"
MODEL_CONFIG_DIR.mkdir(exist_ok=True)
# 单实例模式只有 1 个端口；关闭并行时恢复 ports_per_model 个端口做多实例并发。
PORTS_PER_MODEL = (1 if PARALLEL_INFER
                   else int(config.get("ports_per_model", config.get("pool_size", 1)) or 1))
MODEL_PORT_BASE = {key: BACKEND_BASE_PORT + i * PORTS_PER_MODEL
                   for i, key in enumerate(MODEL_KEYS)}
PORT_MODEL = {port: key for key, base in MODEL_PORT_BASE.items()
              for port in range(base, base + PORTS_PER_MODEL)}
ALL_BACKEND_PORTS = sorted(PORT_MODEL)
_backend_model = {}    # port -> model_key actually loaded

# Port-lease pool: one lease per backend instance, caps concurrency at the
# pool size and guarantees a port is never POSTed concurrently.
import queue as _queue
_port_leases = _queue.Queue()
for _p in ALL_BACKEND_PORTS:
    _port_leases.put(_p)


def _acquire_port(voice=None, model_key=None):
    """Take a free backend port (blocks until one is available).

    With parallel_infer=true there is one port per model, so all requests for
    the same model share it and concurrency comes from batch_infer inside that
    single process. With parallel_infer=false there are several ports per model
    and requests spread across them. Either way a lease is held until
    _release_port() after synthesis, so a port is never POSTed twice at once."""
    return _port_leases.get()


def _release_port(port):
    _port_leases.put(port)


def _resolve_model_key(model_key=None):
    key = str(model_key or DEFAULT_MODEL)
    return key if key in MODEL_PROFILES else DEFAULT_MODEL


def _write_model_config(model_key):
    """Generate the per-version tts_infer YAML used by api_v2.py -c."""
    key = _resolve_model_key(model_key)
    profile = MODEL_PROFILES[key]
    version = profile.get("version", key)
    custom = {
        "bert_base_path": profile.get("bert_base_path", TTS_BASE_CFG.get("bert_base_path")),
        "cnhuhbert_base_path": profile.get("cnhuhbert_base_path", TTS_BASE_CFG.get("cnhuhbert_base_path")),
        "device": str(profile.get("device", TTS_BASE_CFG.get("device", "auto"))),
        "is_half": bool(profile.get("is_half", TTS_BASE_CFG.get("is_half", True))),
        "t2s_weights_path": profile.get("t2s_weights_path", TTS_BASE_CFG.get("t2s_weights_path")),
        "version": version,
        "vits_weights_path": profile.get("vits_weights_path", TTS_BASE_CFG.get("vits_weights_path")),
    }
    doc = {"version": version, "custom": custom, version: dict(custom)}
    path = MODEL_CONFIG_DIR / f"{key}.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


for _model_key in MODEL_KEYS:
    _write_model_config(_model_key)

POOL_IDLE_TTL = config.get("pool_idle_ttl", 600)       # idle seconds → unload backend
POOL_EVICT_GRACE = config.get("pool_evict_grace", 120)  # 换用另一端口时，对方至少空闲这么多秒才先卸载
_backend_procs = {}    # port -> Popen
_backend_state = {}    # port -> 'cold' | 'warming' | 'hot'
_backend_last_use = {}  # port -> epoch of last synthesis touch
# RLock (reentrant), NOT Lock: _ensure_backend_hot calls _touch() (which takes
# _pool_lock) while already inside `with _pool_lock:` on the hot fast-path /
# warm re-check. A plain Lock self-deadlocks there — seen in prod as "backend
# loads, first sentence hangs forever, every later request queues behind it".
_pool_lock = threading.RLock()
_warm_lock = threading.RLock()

# ── Backend lifecycle events (web UI / SSE) ─────────────────────
_backend_since = {}        # port -> epoch when its process (re)started
_backend_events = []       # bounded ring of {type,port,state,msg,at}
_backend_events_max = 300
_backend_subs = []         # (asyncio-loop, asyncio.Queue) per connected SSE client
_backend_event_lock = threading.Lock()


def _backend_event(port, state, msg):
    """Record a backend lifecycle event and push it to every SSE client.
    Callable from any thread (model loads happen in worker threads)."""
    ev = {"type": "backend", "port": port, "state": state, "msg": msg,
          "at": time.strftime("%H:%M:%S")}
    with _backend_event_lock:
        _backend_events.append(ev)
        over = len(_backend_events) - _backend_events_max
        if over > 0:
            del _backend_events[:over]
        subs = list(_backend_subs)
    for loop, q in subs:
        try:
            loop.call_soon_threadsafe(q.put_nowait, ev)
        except Exception:
            pass


def _backend_snapshot():
    """Current per-port backend state, for /api/health & the status UI."""
    now = time.time()
    out = []
    with _pool_lock:
        for port in ALL_BACKEND_PORTS:
            proc = _backend_procs.get(port)
            st = _backend_state.get(port, "cold")
            if proc is None or proc.poll() is not None:
                st = "cold"  # process gone → report reality, not stale state
            since = _backend_since.get(port)
            out.append({
                "port": port,
                "model": _backend_model.get(port) or PORT_MODEL.get(port),
                "state": st,
                "pid": proc.pid if (proc is not None and proc.poll() is None) else None,
                "elapsed": round(now - since, 1) if since is not None and st in ("warming", "hot") else None,
            })
    return out


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


def start_backend_instance(port, model_key=None):
    """Start one backend instance on the given port (process begins loading
    its model; the port binds only once the model is loaded, ~90 s)."""
    key = _resolve_model_key(model_key or PORT_MODEL.get(port))
    cfg_path = _write_model_config(key)
    env = os.environ.copy()
    bin_dir = os.path.join(INTEL_ROOT, "runtime", "Library", "bin")
    ipex_bin = os.path.join(INTEL_ROOT, "runtime", "Lib", "site-packages",
                            "intel_extension_for_pytorch", "bin")
    env["PATH"] = f"{bin_dir};{ipex_bin};" + env.get("PATH", "")
    log_file = open(os.path.join(SCRIPT_DIR, f"backend_{port}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        [os.path.join(INTEL_ROOT, "runtime", "python.exe"),
         "api_v2.py", "-a", "0.0.0.0", "-p", str(port), "-c", str(cfg_path)],
        cwd=INTEL_ROOT, env=env,
        stdout=log_file, stderr=log_file,
    )
    with _pool_lock:
        _backend_procs[port] = proc
        _backend_state[port] = "warming"
        _backend_model[port] = key
    _backend_since[port] = time.time()
    _backend_event(port, "warming", f"后端 {port}（{key}）进程已启动，正在加载模型…")
    logger.info("Backend %s starting (%s, PID %s, config %s)", port, key, proc.pid, cfg_path)


def _stop_backend(port):
    """Force-kill a backend process — releases its model from XPU/RAM.
    Used on timeout/OOM recovery and explicit restarts."""
    with _pool_lock:
        proc = _backend_procs.pop(port, None)
        _backend_state[port] = "cold"
        _backend_model.pop(port, None)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    logger.info("Backend %s stopped — model released.", port)
    _backend_since.pop(port, None)
    _backend_event(port, "cold", f"后端 {port} 已停止，模型已释放")


def _evict_if_idle(port, min_idle):
    """Stop `port` only if it is still hot AND has been idle > min_idle.
    Re-checks under the lock so a peer that just became active is spared."""
    with _pool_lock:
        if _backend_state.get(port) != "hot":
            return
        if time.time() - _backend_last_use.get(port, 0) <= min_idle:
            return
        proc = _backend_procs.pop(port, None)
        _backend_state[port] = "cold"
        _backend_model.pop(port, None)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    logger.info("Backend %s idle > %ss — model released.", port, min_idle)
    _backend_since.pop(port, None)
    _backend_event(port, "cold", f"后端 {port} 空闲超时，已卸载模型（释放显存）")


def backend_ready(port):
    """Block until the backend on `port` is serving (model loaded).

    api_v2 binds its port only after loading the model, so a responding port
    means the model is resident. ~4 min load budget. Returns True/False."""
    for i in range(120):
        proc = _backend_procs.get(port)
        if proc and proc.poll() is not None:
            logger.error("Backend %s crashed (exit %s)", port, proc.poll())
            _backend_event(port, "cold", f"后端 {port} 加载中崩溃（exit {proc.poll()}）")
            return False
        if _port_in_use(port):
            _backend_event(port, "hot", f"后端 {port} 模型加载完成，已就绪")
            return True
        if i % 5 == 0:
            logger.info("Backend %s loading model... (%ds)", port, i * 2)
            _backend_event(port, "warming", f"后端 {port} 正在加载模型 ({i * 2}s)")
        time.sleep(2)
    logger.error("Backend %s not ready after %ds", port, 240)
    _backend_event(port, "cold", f"后端 {port} 加载超时（240s），请查日志")
    return False


def _touch(port):
    with _pool_lock:
        _backend_last_use[port] = time.time()


def _ensure_backend_hot(port, model_key=None):
    """Idempotently bring `port` to 'hot' (model loaded & responding), starting
    the process lazily if needed. Call right before a synthesis uses the port.

    Model loads are serialized (single global warm lock), and before a cold
    backend loads, hot peers idle > POOL_EVICT_GRACE are stopped so only the
    actively-needed model stays resident. Returns True when ready; a cold start
    blocks ~90 s (the price of not keeping models warm)."""
    key = _resolve_model_key(model_key or PORT_MODEL.get(port))
    # A port is dedicated to one profile, but a config edit/hot reload can
    # leave the wrong model on it. Restart rather than serve the wrong voice.
    if _port_in_use(port) and _backend_model.get(port) not in (None, key):
        logger.warning("Backend %s holds %s, need %s — restarting",
                       port, _backend_model.get(port), key)
        _stop_backend(port)
    with _pool_lock:
        if (_backend_state.get(port) == "hot" and _port_in_use(port)
                and _backend_model.get(port) in (None, key)):
            _touch(port)
            return True
    # Single-model pool: no peer eviction. Every port holds the same v5Turbo
    # model, so hot peers are never stopped to make room. Loads stay serialized
    # under _warm_lock so concurrent cold starts never spike XPU memory.
    with _warm_lock:
        with _pool_lock:
            if (_backend_state.get(port) == "hot" and _port_in_use(port)
                    and _backend_model.get(port) in (None, key)):
                _touch(port)
                return True
            proc = _backend_procs.get(port)
            need_start = (not _port_in_use(port)
                          and (proc is None or proc.poll() is not None))
            if not need_start:
                _backend_state[port] = "warming"  # a proc is up, model still loading
        if need_start:
            start_backend_instance(port, key)
        ok = backend_ready(port)
        with _pool_lock:
            if ok:
                _backend_state[port] = "hot"
                _backend_model[port] = key
                _backend_last_use[port] = time.time()
                logger.info("Backend %s ready (%s, model loaded).", port, key)
            else:
                _backend_state[port] = "cold"
                _backend_model.pop(port, None)
        return ok


def _monitor_backends():
    """Daemon: unload idle backends to release RAM/XPU memory, and mark dead
    ones cold for lazy reload. A single loaded model holds ~7 GB RAM, so
    POOL_IDLE_TTL must stay > 0 on this machine or the box runs out of memory.
    Idle eviction is skipped only when POOL_IDLE_TTL == 0 (not recommended)."""
    while True:
        time.sleep(15)
        now = time.time()
        to_evict = []
        with _pool_lock:
            for port in ALL_BACKEND_PORTS:
                proc = _backend_procs.get(port)
                st = _backend_state.get(port, "cold")
                if proc is None:
                    continue
                alive = proc.poll() is None
                if POOL_IDLE_TTL > 0 and st == "hot" and alive \
                        and now - _backend_last_use.get(port, 0) > POOL_IDLE_TTL:
                    to_evict.append(port)
                elif not alive:
                    # died (was hot/warming) — model already gone; reload on demand
                    logger.warning("Backend %s process DEAD (was %s) — will reload on next use",
                                   port, st)
                    _backend_state[port] = "cold"
                    _backend_procs.pop(port, None)
                    _backend_model.pop(port, None)
                    _backend_since.pop(port, None)
                    _backend_event(port, "cold", f"后端 {port} 进程异常退出（was {st}），将在下次使用时重载")
        for port in to_evict:
            _evict_if_idle(port, POOL_IDLE_TTL)

def voice_backend_port(voice, model_key=None):
    """Route voice to a backend instance of the selected model profile."""
    key = _resolve_model_key(model_key)
    base = MODEL_PORT_BASE[key]
    h = 0
    for ch in voice:
        h = (h * 31 + ord(ch)) & 0x7fffffff
    return base + (h % PORTS_PER_MODEL)
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


def _speaker_cache_key(speaker, voice, speed, model_key=None):
    return f"{_resolve_model_key(model_key)}|{speaker}|{voice}|{speed}"


def _speaker_cache_path(speaker, voice, speed, model_key=None):
    """Disk path for a speaker's audio. Hash avoids illegal filename chars."""
    import hashlib
    key = _speaker_cache_key(speaker, voice, speed, model_key)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    # Voice + speed are safe-ish; speaker goes into hash
    return SPEAKER_CACHE_DIR / f"{_resolve_model_key(model_key)}_{voice}_{speed}_{h}.wav"


def _speech_span(data, sr, threshold=0.005, window_ms=100, pad_ms=50):
    """(start, end) sample indices of the audible span in 1-D float audio, or
    None when the clip is too short / entirely silent / would trim to nothing.
    Used by _trim_silence()."""
    import numpy as _np
    n = len(data)
    win = max(1, int(sr * window_ms / 1000.0))
    n_windows = n // win
    if n_windows < 2:
        return None
    seg = data[: n_windows * win].reshape(n_windows, win)
    rms = _np.sqrt((seg ** 2).mean(axis=1))
    above = _np.where(rms > threshold)[0]
    if len(above) == 0:
        return None            # all silent — caller keeps the audio unchanged
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, int(above[0]) * win - pad)
    end = min(n, int(above[-1] + 1) * win + pad)
    if end <= start:
        return None
    return start, end


def _trim_silence(wav_bytes, threshold=0.005, window_ms=100, pad_ms=50):
    """Trim leading/trailing silence from a WAV using windowed RMS.

    Used ONLY when splicing a danmaku reply together (speaker clip + text
    clip), where the vocoder's ~0.5 s lead-in would otherwise put an audible
    pause between "xxx说" and the message. Chapter audio is NOT trimmed: the
    stored WAV stays a plain concatenation of the sentence clips, which is what
    keeps the per-sentence times in the split file exact. A clip that is silent
    throughout is not handled here — _is_silent_wav() rejects it earlier."""
    try:
        import io as _io
        data, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        span = _speech_span(data, sr, threshold, window_ms, pad_ms)
        if span is None:
            return wav_bytes
        start, end = span
        out = _io.BytesIO()
        sf.write(out, data[start:end], sr, format="WAV", subtype="PCM_16")
        out.seek(0)
        return out.getvalue()
    except Exception as e:
        logger.warning("Trim silence failed: %s", e)
        return wav_bytes


def _wav_seconds(wav_bytes):
    """Duration of a WAV in seconds (0.0 if unreadable)."""
    try:
        info = sf.info(io.BytesIO(wav_bytes))
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return 0.0


def _head_trim_seconds(wav_bytes, threshold=0.005, window_ms=100, pad_ms=50):
    """How much _trim_silence() would cut off the FRONT, in seconds.

    No longer used for chapter audio — the merged WAV is not trimmed, so the
    split file's times need no rebasing. Kept as the measurement helper for
    anyone comparing a danmaku splice against the raw clip."""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        span = _speech_span(data, sr, threshold, window_ms, pad_ms)
        return span[0] / float(sr) if span else 0.0
    except Exception:
        return 0.0


def _sentence_starts(durations):
    """Where each sentence begins inside the chapter audio, in seconds.

    The chapter is the plain concatenation of the sentence clips (the merged
    audio is never trimmed), so these are just running sums:
        starts[k] = sum(durations[:k])

    No end array is stored, because a sentence's end is simply the next
    sentence's start — and the last one ends wherever the file does. That also
    means a sentence dropped for silence (duration 0.0) collapses to a
    zero-length slot at the boundary without disturbing any other time."""
    starts, acc = [], 0.0
    for d in durations:
        starts.append(acc)
        acc += d
    return starts


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


def _get_cached_speaker(speaker, voice, speed, model_key=None):
    key = _speaker_cache_key(speaker, voice, speed, model_key)
    # 1. In-memory cache
    with _speaker_cache_lock:
        if key in _speaker_cache:
            return _speaker_cache[key]
    # 2. Disk cache
    path = _speaker_cache_path(speaker, voice, speed, model_key)
    if path.exists():
        try:
            wav_bytes = path.read_bytes()
            with _speaker_cache_lock:
                _speaker_cache[key] = wav_bytes  # warm the in-memory cache
            return wav_bytes
        except Exception as e:
            logger.warning("Failed to read speaker cache %s: %s", path, e)
    return None


def _cache_speaker(speaker, voice, speed, wav_bytes, model_key=None):
    key = _speaker_cache_key(speaker, voice, speed, model_key)
    # 1. In-memory cache (with simple eviction)
    with _speaker_cache_lock:
        if len(_speaker_cache) >= MAX_SPEAKER_CACHE:
            _speaker_cache.pop(next(iter(_speaker_cache)))
        _speaker_cache[key] = wav_bytes
    # 2. Disk cache (persistent)
    path = _speaker_cache_path(speaker, voice, speed, model_key)
    try:
        path.write_bytes(wav_bytes)
    except Exception as e:
        logger.warning("Failed to write speaker cache %s: %s", path, e)

# Language mapping
LANG_MAP = {"zh": "中文", "ja": "日文", "en": "英文", "yue": "粤语", "ko": "韩文",
            "all_zh": "all_zh", "all_ja": "all_ja"}

# ── 4 路并发：端口租约池 ────────────────────────────────────────
# 所有端口现在跑同一个 v5Turbo 模型（单 profile、4 ports_per_model）。
# 旧的全局 _tts_lock 会把所有请求串行化；现在每个请求从租约池取一个端口，
# 独占到合成结束再归还 —— 最多 4 路真正并行，且同一端口不会被并发_POST。
# 租约池在 ALL_BACKEND_PORTS 确定后初始化（见下方）。
_port_lease_lock = threading.Lock()
# 静音计数器的并发保护（旧代码依赖 _tts_lock，已移除）
_silent_lock = threading.Lock()

# ── Long-text task management ────────────────────────────────────
# For audiobook-style synthesis: split text into sentences, synthesize
# each sequentially, report progress, then merge into one WAV.
import uuid

_tts_tasks = {}
_tts_tasks_lock = threading.Lock()
MAX_TASKS = 50
TASK_TTL = 172800  # seconds (48h = 2 days) — tasks + WAV files auto-expire
TASK_CACHE_DIR = Path(os.environ.get("BSAI_TASK_CACHE") or (SCRIPT_DIR / "task_cache"))
TASK_CACHE_DIR.mkdir(exist_ok=True)


def _save_task(task_id, task):
    """Persist a task to disk so it survives proxy crashes/restarts."""
    try:
        # WAV file
        wav = task.get("wav")
        if wav:
            (TASK_CACHE_DIR / f"{task_id}.wav").write_bytes(wav)
        # Metadata (excludes the wav blob and worker-local bookkeeping such as
        # the partial-audio cache, whose bytes are not JSON-serializable — one
        # stray key here would abort the whole per-sentence progress write).
        meta = {k: v for k, v in task.items()
                if k not in ("wav", "key") and not k.startswith("_")}
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
            elif meta.get("status") in ("running", "queued") or (
                    meta.get("status") == "error" and meta.get("done", 0) > 0):
                # Crash recovery: resume from where we left off. "error" tasks
                # with any progress were aborted mid-run by a transient failure
                # (e.g. backend read timeout), not rejected by the user — resume
                # them the same as "running".
                # 关键：done == total 也必须恢复 —— 所有片段都合成好了、但服务在
                # 「合并整章 + 保存 wav」之前重启时，任务会带着 done==total 落成
                # error（旧条件 0<done<total 把它永远卡死在 "Proxy restarted"）。
                # _run_long_task 在 start==total 时会跳过合成、直接从磁盘读片段
                # 合并，正是这里需要的收尾路径。
                meta["wav"] = None
                meta["status"] = "running"
                _tts_tasks[task_id] = meta
                threading.Thread(
                    target=_run_long_task,
                    args=(task_id, meta.get("text", ""), meta.get("voice", "default"),
                          meta.get("speed", 1.0), meta.get("version", DEFAULT_MODEL)),
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


def _pick_lang(text, default):
    """Pick the GPT-SoVITS text_lang for one cleaned sentence.

    The backend's `zh` frontend splits embedded English out via LangSegmenter
    and voices it in English (TextPreprocessor.get_phones_and_bert), so a
    Chinese sentence that merely carries an English word or name —
    “那个人，叫Vein。” — MUST stay on the default (zh): the Chinese is spoken
    and “Vein” is read as English. Routing such a line to `en` makes the en
    frontend DROP all the CJK and voice only the latin bits.

    Only pick a non-default script when the line is essentially ONE script:
    - pure latin (no CJK / kana at all) → en  (e.g. “My pleasure，sweetie。” —
      here the zh path would go near-silent), or
    - kana-dominant → ja.
    Anything that still has a real CJK component stays zh, whose embedded
    latin fragments are read correctly anyway. A trailing ASCII marker such as
    “-END-” adds latin letters but no CJK — it must NOT be able to flip a
    Chinese sentence to en (cjk > 0 keeps it zh)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return default
    latin = sum(1 for c in letters if 'a' <= c.lower() <= 'z')
    cjk = sum(1 for c in letters if '一' <= c <= '鿿')
    kana = sum(1 for c in letters if '぀' <= c <= 'ヿ')
    n = len(letters)
    if kana and kana >= n * 0.5 and kana > cjk:
        return "ja"
    if cjk == 0 and kana == 0 and latin >= 2:
        return "en"
    return default


def _speakable_len(s):
    """Length of a sentence ignoring leading/trailing punctuation — the
    amount that actually reaches the phoneme extractor."""
    return len(s.strip('。！？!?；;,.…～~、 \t　"“”‘’'))


def _core_speak_len(s):
    """Speakable length in CJK-equivalent DURATION units — the amount of
    audio GPT-SoVITS will produce, i.e. AFTER _clean_text() strips curly
    quotes / em-dashes / ellipsis and drops non-synthesizable chars. One CJK
    character is spoken in ~0.3 s, one latin letter/digit in ~0.1 s, so every
    3 latin chars count as 1 unit here: '"Of cause."' measures 7 raw chars
    but only ~0.6 s of speech — exactly the ultra-short zone whose T2S decodes
    immediate-EOS → near-silence — while 7 CJK chars (~2 s) are safe. A raw
    char count let sub-second English lines slip past the shorts-merge and
    burn dozens of doomed retries (task 942a20b9ba6c, sentence 118). Shortness
    tests must use THIS. Emoji are ignored (they expand to Chinese during
    cleaning, but a bare emoji string is far below the merge threshold anyway)."""
    cjk = 0
    latin = 0
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:      # kana
            cjk += 1
        elif 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF:    # CJK ideographs
            cjk += 1
        elif 0xAC00 <= o <= 0xD7AF:                              # hangul
            cjk += 1
        elif 0x21 <= o <= 0x7E:                                  # ASCII
            if ch.isalnum():
                latin += 1
        elif 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A \
                or 0xFF41 <= o <= 0xFF5A:                        # full-width alnum
            latin += 1
    return cjk + (latin + 2) // 3   # ceil(latin/3): ~0.1 s per latin char


def _split_sentences(text):
    """Split long text into speakable sentences.
    - Normalizes CRLF/CR to LF (Windows pastes)
    - Splits on sentence terminators AND newlines (paragraph breaks)
    - Merges isolated short fragments (closing quotes like 」”）) back
      into a neighbouring sentence, so `“xxx。”` isn't torn apart into
      `“xxx。` + `”` (the bare quote would be cleaned to empty & skipped).
    - Merges any sentence whose POST-CLEAN core is tiny (≤ SHORT_CORE) into a
      neighbour — GPT-SoVITS's T2S decodes immediate-EOS → near-silent audio
      for very short texts, so they must never reach the backend alone.
      Shortness is judged on _core_speak_len() (what _clean_text() actually
      leaves), NOT the raw length: a quote line like “萧——未——影！” measures 7
      raw chars but is a 3-char utterance after the em-dashes/quotes are
      stripped — with a raw-length test it slips through alone and comes out
      silent. A short that trails a longer sentence joins it; a leading run of
      shorts (a paragraph opening “嗯。”) carries forward onto the next
      non-short sentence instead of standing alone.
    """
    import re
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # A trailing book-end marker (“-END-”, “——END——”, “ END”) must NOT be left
    # inside a Chinese sentence — its latin letters used to flip the whole line
    # to English in _pick_lang (“那个人，叫Vein。-END-” read only “Vein”). So
    # strip it from the prose, and instead let it be spoken as a clean English
    # “End.” appended as its own fragment (pure-latin → en frontend). Only
    # matches a clearly-marked token: uppercase END or any dash/underscore-
    # delimited end — prose ending in lowercase “…the end” (or with a trailing
    # period) is left untouched.
    _end_read = False
    _em = re.search(r'[\s\-—＿_~]*end[\s\-—＿_~]*$', text, re.IGNORECASE)
    if _em:
        _region = _em.group(0)
        _core = _region.strip(' \t\r\n-—＿_~')
        if _core == 'END' or any(ch in _region for ch in '-—＿_'):
            text = text[:_em.start()]
            _end_read = True

    # Split keeping the terminator attached to its sentence
    parts = re.split(r'(?<=[。！？!?；;\n])', text)
    sents = [p.strip() for p in parts if p.strip()]

    SHORT_CORE = 6  # post-clean speakable core ≤ this must not stand alone

    def _too_short(s):
        return _core_speak_len(s) <= SHORT_CORE

    # Pass 1 — a too-short fragment joins its (long) predecessor. Consecutive
    # shorts keep folding back into that same predecessor, so “…。好。嗯。”
    # collapses into one speakable sentence instead of a silent 1-char tail.
    merged = []
    for s in sents:
        if merged and _too_short(s):
            merged[-1] += s
        else:
            merged.append(s)

    # Pass 2 — whatever is still too short is a LEADING run (nothing long
    # behind it). Glue consecutive shorts and attach them to the FRONT of the
    # following non-short sentence; a short-only tail joins the last sentence.
    result = []
    pending = []
    for s in merged:
        if pending and not _too_short(s):
            pending.append(s)
            s = "".join(pending)
            pending = []
        if _too_short(s):
            pending.append(s)
        else:
            result.append(s)
    if pending:
        tail = "".join(pending)
        if result:
            result[-1] += tail
        else:
            result.append(tail)
    merged = result

    # Further split any sentence over 200 chars
    result = []
    for s in merged:
        while len(s) > 200:
            cut = s[:200].rfind('，')
            if cut < 50:
                cut = 200
            result.append(s[:cut])
            s = s[cut:]
        if s:
            result.append(s)
    # Voiced “End.” for a stripped book-end marker (see above).
    if _end_read:
        result.append("End.")
    return result


# Common danmaku emoji → Chinese word, so they're read aloud instead of dropped.
EMOJI_MAP = {
    "🎉": "撒花", "🎊": "庆祝", "🎆": "庆祝", "🎇": "庆祝",
    "😀": "开心", "😄": "开心", "😁": "开心", "😆": "开心",
    "😂": "笑哭", "🤣": "笑哭", "😅": "尴尬", "😊": "微笑",
    "😍": "喜欢", "🥰": "喜欢", "😘": "飞吻", "😎": "得意",
    "👍": "点赞", "👎": "踩", "👏": "鼓掌", "🙏": "拜托",
    "🔥": "好棒", "💀": "无语", "😭": "大哭", "😢": "难过",
    "😡": "生气", "🤬": "愤怒", "😱": "惊吓", "🤯": "震惊",
    "🤔": "思考", "😳": "害羞", "😴": "睡觉", "😋": "好吃",
    "🥺": "可怜", "😇": "天使", "😈": "恶魔", "🤗": "拥抱",
    "🤪": "搞怪", "🤑": "发财", "🤩": "惊喜", "🥳": "庆祝",
    "❤": "爱心", "💕": "爱心", "💯": "满分", "✨": "闪亮",
    "🌟": "星星", "⭐": "星星", "🍀": "好运", "👌": "可以",
    "✌": "胜利", "🤝": "握手", "👋": "再见", "🫡": "敬礼",
    "🤤": "流口水", "🥵": "热", "🥶": "冷", "😤": "不服",
    "😮‍💨": "叹气", "🫠": "融化", "🫶": "比心",
}

import re

# Longest-first alternation so multi-codepoint entries (e.g. 😮‍💨) match
# whole instead of their parts.
_CURATED_EMOJI_RE = re.compile(
    "|".join(map(re.escape, sorted(EMOJI_MAP, key=len, reverse=True))))

try:
    import emoji as _emoji_mod
except Exception:
    _emoji_mod = None


def _emoji_to_cn(text):
    """Convert emojis to spoken Chinese so GPT-SoVITS reads them aloud.

    Layer 1: curated danmaku words (EMOJI_MAP) — 👍🏻 → 点赞, 🔥 → 好棒.
    Layer 2: the `emoji` package's Unicode CLDR Simplified-Chinese short
    names for everything else — 🇨🇳 → 中国, 👨‍👩‍👧 → 家庭男人女人女孩.
    Falls back to layer 1 alone if the package is missing.
    """
    # Strip invisible chars that would fragment multi-codepoint emoji:
    # variation selectors (FE00–FE0F, incl. emoji presentation), ZWSP, ZWNJ,
    # LRM/RLM, word joiner, BOM. NOT ZWJ (U+200D) — demojize needs it to
    # join family/couple sequences whole.
    text = re.sub(r'[﻿︀-️​‌‎‏⁠]', '', text)
    # Skin-tone modifiers strip out, so 👍🏻 falls back to base 👍 — reads as
    # "点赞" instead of "点赞_较浅肤色" mid-sentence.
    text = re.sub(r'[\U0001f3fb-\U0001f3ff]', '', text)
    # Layer 1: curated danmaku words (one longest-first pass)
    text = _CURATED_EMOJI_RE.sub(lambda m: EMOJI_MAP[m.group(0)], text)
    # Layer 2: CLDR zh short names for whatever remains
    if _emoji_mod is not None:
        try:
            text = _emoji_mod.demojize(text, language='zh')
            text = text.replace(':', '').replace('_', '')
        except Exception:
            pass  # keep layer-1 result
    return text


# ── 符号 → 读音 ────────────────────────────────────────────────────
# GPT-SoVITS 的文本前端会在归一化阶段**静默丢弃**这些符号（实测：后端日志里
# “A+B=3” 出来是 “A Bthree”，+ 和 = 直接没了；℃ 属于非白名单字符，更早一步
# 就被 _clean_text 丢掉）。所以必须在送进后端之前换成读音，否则整段意思会缺字。
_SYMBOL_SPEECH = {
    # 温度 / 角度
    "℃": "摄氏度", "℉": "华氏度", "°": "度",
    # 数学 / 比较
    "+": "加", "＋": "加", "=": "等于", "＝": "等于",
    "×": "乘", "✕": "乘", "✖": "乘", "÷": "除以",
    "±": "正负", "≈": "约等于", "≠": "不等于",
    "≥": "大于等于", "≤": "小于等于",
    "＜": "小于", "＞": "大于",
    # 连接符
    "&": "和", "＆": "和", "→": "到", "←": "到",
    # 标记符：语音前端会直接丢掉，但正文里它常常带语义（#标签 / @某人）
    "#": "井号", "＃": "井号",
    "@": "艾特", "＠": "艾特",
    # 其他
    "‰": "千分之", "№": "编号",
}
# 装饰性重复符号（章节分隔线 ===== / ---- / **** / ####）不读，先压成空格
_DECOR_RUN = re.compile(r"([=\-*#~_+·])\1{2,}")


def _symbols_to_cn(text):
    """把“念不出来”的符号换成中文读音；装饰性符号直接去掉。

    `%` 不在这里处理：语音前端自己会把 “50%” 读成“百分之五十”，改动反而更糟。
    `~` 只在数字之间读“到”（3~5 分钟），其他位置（颜文字）保持原样。"""
    if not text:
        return ""
    text = _DECOR_RUN.sub(" ", text)
    for sym, word in _SYMBOL_SPEECH.items():
        if sym in text:
            text = text.replace(sym, word)
    text = re.sub(r"(?<=\d)\s*[~～〜]\s*(?=\d)", "到", text)
    return text


def _clean_text(text):
    """Convert emojis to Chinese words, then KEEP ONLY characters that
    GPT-SoVITS can synthesize (CJK, kana, hangul, ASCII, full-width punct).
    Anything else is dropped — this prevents GBK codec errors like
    \"'gbk' codec can't encode character '\\u23ae'\" on the Windows backend."""
    import re
    if not text:
        return ""
    # 符号 → 读音（必须在白名单过滤之前：℃ 这类字符不在白名单里）
    text = _symbols_to_cn(text)
    # Emojis → spoken Chinese (curated danmaku words, then CLDR zh short names)
    text = _emoji_to_cn(text)

    def _keep(ch):
        o = ord(ch)
        if ch == ' ':
            return True
        if 0x20 <= o <= 0x7E:      # ASCII printable (letters/digits/punct)
            return True
        if 0x3000 <= o <= 0x303F:  # CJK punctuation 。、，？！ etc
            return True
        if 0x3040 <= o <= 0x30FF:  # hiragana + katakana
            return True
        if 0x31F0 <= o <= 0x31FF:  # katakana phonetic extensions
            return True
        if 0x3400 <= o <= 0x9FFF:  # CJK unified ideographs
            return True
        if 0xF900 <= o <= 0xFAFF:  # CJK compatibility ideographs
            return True
        if 0xAC00 <= o <= 0xD7AF:  # hangul
            return True
        if 0xFF00 <= o <= 0xFFEF:  # full-width forms（，。！？ etc）
            return True
        return False

    text = ''.join(c for c in text if _keep(c))
    # A leading sentence mark ("。短文本") hands the GPT-SoVITS frontend a
    # degenerate phone sequence: T2S decodes immediate-EOS → near-silence,
    # identically for every seed (measured: 6/6 inferences EOS at 12 tokens).
    # Leading punctuation is never speakable — drop it before the text ships.
    text = text.lstrip('。．｡，,、；;：:！!？?…～~·.')
    # A sentence that cleans down to only punctuation — e.g. an emoji tag
    # line "🖤💙🔁❌！" → "！" — has nothing speakable. Return empty so the
    # caller skips it instead of sending a bare "！" to the backend
    # (which can't synthesize it and just yields silence).
    if not any(c.isalnum() for c in text):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _new_task(text, voice, speed, model_key=None):
    # The task id IS the credential. No endpoint ever lists tasks and
    # task_cache/ is not mounted anywhere, so a chapter is reachable only by
    # knowing its random id. The separate `key` added nothing: it was handed
    # out everywhere the id was (deep links, the downloaded split file), so it
    # never was the "something extra" it looked like. Old links and old
    # archives keep working — an unknown ?key=... is simply ignored.
    task_id = uuid.uuid4().hex[:12]
    import time as _t
    with _tts_tasks_lock:
        if len(_tts_tasks) >= MAX_TASKS:
            for k in list(_tts_tasks):
                if _tts_tasks[k]["status"] in ("done", "error"):
                    del _tts_tasks[k]
        _tts_tasks[task_id] = {
            "status": "queued", "total": 0, "done": 0,
            "error": None, "wav": None, "voice": voice, "speed": speed,
            "version": _resolve_model_key(model_key),
            "text": text, "created": _t.time(),
        }
    return task_id


def _run_long_task(task_id, text, voice, speed, model_key=None, engine=None):
    """Background worker: synthesize each sentence, merge, store result.
    Supports crash recovery — each completed sentence is persisted to disk
    (fragment WAV + progress), so a restart resumes from the last sentence."""
    key = _resolve_model_key(model_key)
    try:
        # Reuse the exact split stored on the task when resuming, so a restart
        # after a split-logic change doesn't shift fragment indices and tear
        # the audio mid-sentence. Fresh tasks split below.
        sents = None
        with _tts_tasks_lock:
            saved = _tts_tasks[task_id].get("sents")
            if saved:
                sents = list(saved)
        if sents is None:
            sents = _split_sentences(text)
        total = len(sents)
        with _tts_tasks_lock:
            task = _tts_tasks[task_id]
            task["total"] = total
            task["status"] = "running"
            task["text"] = text
            task["sents"] = sents  # persist so later resumes reuse this split
            start = task["done"]  # resume point (0 for fresh tasks)
        _save_task(task_id, dict(_tts_tasks[task_id]))

        # Backends load lazily inside _synthesize() on the first sentence.
        # The old voice-affinity preload is gone: with a 4-port lease pool any
        # free port serves any voice, and preloading one port here would pin a
        # lease-less backend / block a concurrent request during a cold start.
        # Skip entirely when a 点读包 has all fragments already (nothing to do).

        # Load fragments synthesized before a crash, measuring each one's
        # duration on the way in: 点读 converts these into seek offsets inside
        # the merged chapter audio, so the clips themselves need not be kept.
        # One slot per sentence (0.0 for a skipped one) keeps the list aligned
        # with the split file.
        # 片段按「句子下标」存，不按完成顺序 append：调度随时可能按长度重排
        # 批次（甚至将来改成动态接活），只要最后按下标组装，音频顺序就永远
        # 与 sents 一致 —— 点读的 starts 是按 durations 累加的，顺序错一格全乱。
        chunks = {}
        durations = [0.0] * total   # one slot per sentence, 0.0 = skipped
        for i in range(start):
            f = TASK_CACHE_DIR / f"{task_id}.{i}.wav"
            # Re-zero before measuring: a persisted duration whose fragment is
            # gone (skipped sentence, or a fragment lost with a wiped cache)
            # must collapse to a zero-length slot, or every later start would
            # be off by the phantom duration.
            durations[i] = 0.0
            if f.exists():
                raw = f.read_bytes()
                chunks[i] = raw
                durations[i] = _wav_seconds(raw)

        MAX_RETRIES = 10  # 每句最多重试次数（原3次对偶发超时/静音不够）

        def _synth_one(text, i):
            """单句合成（带重试）。批处理失败时的回退路径。"""
            for attempt in range(MAX_RETRIES):
                try:
                    wav = _synthesize(text, voice, speed, pad_short=False, model_key=key,
                                      engine=engine)
                    if wav:
                        return wav
                except HTTPException as e:
                    logger.warning("Long-text: sentence %d attempt %d/%d failed: %s",
                                   i + 1, attempt + 1, MAX_RETRIES, e.detail)
                    time.sleep(1)
                except requests.exceptions.RequestException as e:
                    # A request error must NOT abort the whole task — it's a
                    # transient backend issue; retry the sentence (and let the
                    # failover inside _synthesize deal with the backend).
                    logger.warning("Long-text: sentence %d attempt %d/%d request error: %s",
                                   i + 1, attempt + 1, MAX_RETRIES, e)
                    time.sleep(1)
            logger.warning("Long-text: sentence %d SKIPPED after %d attempts", i + 1, MAX_RETRIES)
            return None

        def _commit(i, wav):
            if not wav:
                return
            chunks[i] = wav
            durations[i] = _wav_seconds(wav)
            # Persist this sentence's audio immediately (crash-proof)
            (TASK_CACHE_DIR / f"{task_id}.{i}.wav").write_bytes(wav)

        # ── 逐句 → 分批 ────────────────────────────────────────────
        # 后端是单进程、请求串行的，所以「一句一请求」无论 batch_size 设多少都
        # 跑不出并行。真正能并行的是：把几句拼成一次请求，让模型在同一批前向
        # 里一起解码，后端再把每句的音频单独返回，点读时间点不受影响。
        # 注意：一次请求只能有一个 text_lang，所以只把同语言的相邻句并成一批。
        cleaned_all = [_clean_text(s) for s in sents]
        v_default_lang = VOICES.get(voice, VOICES["default"]).get("text_lang", "zh")
        langs_all = [_pick_lang(t, v_default_lang) if t else None for t in cleaned_all]

        i = start
        while i < total:
            cleaned = cleaned_all[i]
            if not cleaned:
                logger.warning("Long-text: sentence %d empty after cleanup — skipping", i + 1)
                with _tts_tasks_lock:
                    _tts_tasks[task_id]["done"] = i + 1
                i += 1
                continue

            group = [(i, cleaned)]
            if DYNAMIC_BATCH:
                # 动态接活：批内长度随便混（短句做完立刻补下一句），只要求同语言、
                # 不超过一次请求的队列长度；并发槽位由后端 BATCH_SIZE 控制。
                nxt = i + 1
                while (len(group) < QUEUE_SIZE and nxt < total
                       and cleaned_all[nxt] and langs_all[nxt] == langs_all[i]):
                    group.append((nxt, cleaned_all[nxt]))
                    nxt += 1
            else:
                base_len = len(cleaned)
                nxt = i + 1
                while (len(group) < BATCH_SIZE and nxt < total
                       and cleaned_all[nxt] and langs_all[nxt] == langs_all[i]):
                    # 批内长度要接近。AR 是「所有序列同步走一步」，短句先 EOS 之后
                    # 只能 padding 陪着最长那句走完 —— 4 字的一句要等 67 字的一句跑
                    # 十几秒，这个通道就算白占了。所以只用长度在 1/ratio~ratio 之间
                    # 的句子凑批；离群的句子自己一批（它本来就快/本来就慢）。
                    n = len(cleaned_all[nxt])
                    if not (base_len * BATCH_LEN_RATIO_LO <= n <= BATCH_LEN_RATIO_HI):
                        break
                    group.append((nxt, cleaned_all[nxt]))
                    nxt += 1

            committed = set()

            def _on_item(seq, wav):
                """后端每算好一句就回调：立刻落盘 + 推进「连续完成前缀」。"""
                gi = group[seq][0]
                _commit(gi, wav)
                committed.add(gi)
                d = group[0][0]
                while d in committed:
                    d += 1
                with _tts_tasks_lock:
                    _tts_tasks[task_id]["done"] = d
                    # 每 4 句落一次盘，避免几百句的小文件反复写
                    if len(committed) % 4 == 0:
                        _save_task(task_id, dict(_tts_tasks[task_id]))

            if len(group) > 1:
                got = (_synthesize_batch_stream([t for _, t in group], voice, speed, key, _on_item,
                                                 engine=engine)
                       if DYNAMIC_BATCH else None)
                if got is None:
                    # 流式不可用（老后端/报错）→ 退回一次性批量，再不行逐句
                    wavs = _synthesize_batch([t for _, t in group], voice, speed,
                                             model_key=key, engine=engine)
                    if wavs is None:
                        wavs = [_synth_one(t, gi) for gi, t in group]
                    elif any(w is None for w in wavs):
                        wavs = [w if w else _synth_one(t, gi) for (gi, t), w in zip(group, wavs)]
                    for (gi, _t), wav in zip(group, wavs):
                        _commit(gi, wav)
                        committed.add(gi)
            else:
                _commit(group[0][0], _synth_one(group[0][1], group[0][0]))
                committed.add(group[0][0])

            # 流式里被判静音、或整批失败漏掉的句子：逐句补
            for gi, gt in group:
                if gi not in committed:
                    wav = _synth_one(gt, gi)
                    if wav:
                        _commit(gi, wav)

            with _tts_tasks_lock:
                _tts_tasks[task_id]["done"] = group[-1][0] + 1
                _save_task(task_id, dict(_tts_tasks[task_id]))  # 每批存一次进度
            i = group[-1][0] + 1

        # 按下标组装：批怎么排、谁先跑完都不影响这里的顺序
        ordered = [chunks[k] for k in sorted(chunks)]
        merged, _ = _concat_wav(*ordered) if ordered else (None, None)
        with _tts_tasks_lock:
            task = _tts_tasks[task_id]
            if merged:
                # Ships the plain concatenation — no head/tail trimming. A clip
                # that was silent throughout never got this far (_synthesize
                # rejects it), and leaving the rest untouched is what makes the
                # per-sentence times exact running sums rather than estimates
                # that need rebasing against a trimmed timeline.
                audio = merged.getvalue()
                seconds = _wav_seconds(audio)
                task["wav"] = audio
                task["audio_seconds"] = round(seconds, 3)
                task["durations"] = [round(d, 3) for d in durations]
                # Only starts are needed to navigate: sentence k runs to
                # starts[k+1], and the last sentence plays out the file. Slice
                # the WAV at starts[k]..starts[k+1] and you have sentence k —
                # nothing to derive, nothing to rebase.
                task["starts"] = [round(s, 3) for s in _sentence_starts(durations)]
            task["status"] = "done" if merged else "error"
            if not merged:
                task["error"] = "No sentences synthesized"
            task.pop("_partial_cache", None)   # final wav supersedes the prefix
            _save_task(task_id, dict(task))
        # Now that the merged WAV + per-sentence offsets are persisted, the
        # per-sentence clips are redundant: 点读 seeks inside the merged file.
        # Dropping them keeps a 100+ sentence chapter from costing ~100 MB.
        for i in range(total):
            (TASK_CACHE_DIR / f"{task_id}.{i}.wav").unlink(missing_ok=True)
        logger.info("Long-text task %s done: %d/%d sentences", task_id, task["done"], task["total"])
    except Exception as e:
        logger.error("Long-text task %s failed: %s", task_id, e, exc_info=True)
        with _tts_tasks_lock:
            _tts_tasks[task_id]["status"] = "error"
            _tts_tasks[task_id]["error"] = str(e)
            _save_task(task_id, dict(_tts_tasks[task_id]))


# ── FastAPI ──────────────────────────────────────────────────────
app = FastAPI(title="AI TTS Proxy", version="3.0.0")

# 跨源：前端「服务器选择」会直接去目标机器的 /api/* 读它支持的模型/音色。
# 没有这层头，浏览器会拦掉跨域读，前端只能看到「加载失败」。
# 只放开 /api/*（网页本身还是同源），并且允许带凭证以外的普通 GET/POST。
from fastapi.middleware.cors import CORSMiddleware as _CORSMiddleware
app.add_middleware(
    _CORSMiddleware,
    allow_origins=["*"],              # TTS 服务是自用/内网，允许任意来源读
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


# ── 反向转发：前端只连本机，其它服务器由本机代理 ──────────────────
# 浏览器直连远端会撞两堵墙：跨域（CORS）和混合内容（https 页面里发 http 请求）。
# 所以改成「前端永远只发同源请求，带一个 _srv=<下标> 说明要哪台」，由本服务
# 把请求转发过去、把响应原样流回来 —— 音频、Range 拖动、SSE 进度都不受影响。
import httpx as _httpx
from urllib.parse import urlencode as _urlencode

_forward_client = None


def _forward_tls_verify():
    return bool(config.get("forward_tls_verify", False))


def _server_key(request):
    """请求里选的服务器：_srv 查询参数（音频/EventSource 也带得上）或请求头。"""
    return request.query_params.get("_srv") or request.headers.get("x-bsai-server")


def _server_by_key(raw):
    """_srv（下标或服务器名）→ config 里的那台。指向本机时返回 None。"""
    if raw in (None, "", "0", "local"):
        return None
    servers = [s for s in (config.get("servers") or [])
               if isinstance(s, dict) and s.get("name")]
    if str(raw).isdigit():
        i = int(raw)
        return servers[i] if 0 <= i < len(servers) else None
    return next((s for s in servers if s.get("name") == raw), None)


def _selected_remote(request):
    """要转发到哪台。返回 (根地址, 名字)；本机/空 url 返回 None。"""
    picked = _server_by_key(_server_key(request))
    url = str((picked or {}).get("url") or "").rstrip("/")
    return (url, picked.get("name")) if url else None


def _engine_config(raw):
    """把某台服务器解析成「合成引擎」。

    服务器条目里写 engine: 就走引擎模式 —— 长文任务、进度、点读仍然全部留在
    本机（任务缓存、时间轴、断点续传都不变），只是每一句的合成请求发给远端池，
    所以远端不装 bsai 代理也能吃到它的算力。

        - name: "9070XT"
          url: "http://other:9881"       # 能力接口（/api/voices）也从这里取
          engine:
            api_key: "..."               # 远端池的 X-API-Key（没开鉴权就省略）
            default_voice: "sunflv_fire" # 本机音色在远端没有对应时的兜底
            voice_map:                   # 本机音色名 → 远端音色名
              v4_default: sunflv_fire
    """
    srv = _server_by_key(raw)
    if not srv:
        return None
    eng = srv.get("engine")
    if eng is True:
        eng = {}
    if not isinstance(eng, dict) or eng.get("enabled") is False:
        return None
    url = str(eng.get("url") or srv.get("url") or "").rstrip("/")
    if not url:
        return None
    return {
        "name": srv.get("name") or "",
        "url": url,
        "api_key": str(eng.get("api_key") or ""),
        "voice_map": dict(eng.get("voice_map") or {}),
        "default_voice": str(eng.get("default_voice") or ""),
        "voices": list(eng.get("voices") or []),
    }


def _engine_headers(eng):
    h = {"Content-Type": "application/json"}
    if eng.get("api_key"):
        h["X-API-Key"] = eng["api_key"]
    return h


def _engine_voice(eng, voice):
    """本机音色名 → 远端音色名：显式映射优先，其次同名，最后兜底默认。"""
    vm = eng.get("voice_map") or {}
    if voice in vm:
        return vm[voice]
    if voice and voice in (eng.get("voices") or []):
        return voice
    return eng.get("default_voice") or ""


def _engine_synth(eng, text, voice, speed):
    """单句走远端池的 POST /（直接回 audio/wav）。

    远端的参考音不一定适配所有文本（实测「火海句」那种情绪参考音念短句会
    念成气声），所以映射到的音色出静音时，自动退到 default_voice 再试一次。
    """
    cands = []
    for v in (_engine_voice(eng, voice), eng.get("default_voice"), *(eng.get("voices") or [])):
        if v and v not in cands:
            cands.append(v)
    cands = cands[:2] or [""]
    last_err = ""
    for v in cands:
        payload = {"text": text, "speed_factor": speed,
                   "sample_steps": 4, "media_type": "wav"}
        if v:
            payload["voice"] = v
        try:
            r = requests.post(eng["url"] + "/", headers=_engine_headers(eng),
                              json=payload, timeout=900)
        except requests.exceptions.RequestException as e:
            raise HTTPException(502, f"无法连接合成引擎「{eng['name']}」: {e}")
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}: {r.text[:160]}"
            continue
        if r.content and not _is_silent_wav(r.content):
            return r.content
        last_err = f"音色 {v or '(默认)'} 返回静音"
        logger.warning("Engine %s silent with voice=%s for %r", eng["name"], v, text[:20])
    raise HTTPException(502, f"引擎「{eng['name']}」合成失败：{last_err}")


def _engine_synth_batch(eng, texts, voice, speed):
    """一批句子走远端池的 /tts_batch，按序返回逐句 WAV；不可用时 None。"""
    payload = {"texts": list(texts), "speed_factor": speed, "sample_steps": 4,
               "text_split_method": "cut0", "batch_size": len(texts),
               "media_type": "wav"}
    v = _engine_voice(eng, voice)
    if v:
        payload["voice"] = v
    try:
        r = requests.post(eng["url"] + "/tts_batch", headers=_engine_headers(eng),
                          json=payload, timeout=1800)
        if r.status_code != 200:
            logger.warning("Engine %s batch HTTP %s: %s", eng["name"],
                           r.status_code, r.text[:160])
            return None
        data = r.json()
        wavs = [base64.b64decode(w) for w in (data.get("wavs") or [])]
    except Exception as e:
        logger.warning("Engine %s batch failed: %s", eng["name"], e)
        return None
    if len(wavs) != len(texts):
        logger.warning("Engine %s returned %d wavs for %d texts", eng["name"],
                       len(wavs), len(texts))
        return None
    out = []
    for t, w in zip(texts, wavs):
        if w and not _is_silent_wav(w):
            out.append(w)
            continue
        # 批里有句子抽到静音：单独再合成一次（带音色回退），避免整批退回本机
        try:
            out.append(_engine_synth(eng, t, voice, speed))
        except HTTPException as e:
            logger.warning("Engine %s single retry failed: %s", eng["name"], e.detail)
            out.append(None)
    return out


# 这些路径永远由本机处理：服务器列表、健康检查、网页本身
_NO_FORWARD = ("/api/servers", "/api/health")
# 配了 engine 的机器（只借它的算力）：只有这几条只读元数据转发过去，
# 任务/进度/音频都留本机 —— 点读时间轴与断点续传仍然由本机维护
_ENGINE_META = ("/api/voices", "/api/models", "/api/capabilities")


@app.middleware("http")
async def _forward_to_selected_server(request: Request, call_next):
    global _forward_client
    path = request.url.path
    if not path.startswith("/api/") or path in _NO_FORWARD:
        return await call_next(request)

    target = _selected_remote(request)
    if not target:
        return await call_next(request)
    if _engine_config(_server_key(request)) and path not in _ENGINE_META:
        return await call_next(request)       # 引擎模式：只转发能力查询
    base, name = target

    # 组装转发 URL：去掉 _srv（否则远端会再解释一次），其余查询参数原样带上
    q = [(k, v) for k, v in request.query_params.multi_items() if k != "_srv"]
    url = base + path + (("?" + _urlencode(q)) if q else "")

    body = await request.body()
    # 只挑必要的头：Accept-Encoding 固定 identity，免得 gzip 后 Content-Length 对不上
    fwd_headers = {"accept-encoding": "identity"}
    for k in ("content-type", "accept", "range", "if-range",
              "if-none-match", "if-modified-since"):
        v = request.headers.get(k)
        if v:
            fwd_headers[k] = v

    if _forward_client is None:
        _forward_client = _httpx.AsyncClient(
            verify=_forward_tls_verify(),
            timeout=_httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            follow_redirects=False,
        )

    try:
        # 一定要 send(stream=True)：整章音频可能几百 MB，不能先读进内存
        up_req = _forward_client.build_request(
            request.method, url, headers=fwd_headers,
            content=body if body else None,
        )
        upstream = await _forward_client.send(up_req, stream=True)
    except _httpx.HTTPError as e:
        logger.warning("Forward to %s (%s) failed: %s", name, url, e)
        return JSONResponse(status_code=502, content={
            "message": f"无法连接服务器「{name}」",
            "detail": f"{type(e).__name__}: {e}",
            "url": url,
        })

    # 只透传语义相关的头；Content-Length 保留（identity 编码下仍然准确）
    keep = ("content-type", "content-length", "content-range", "accept-ranges",
            "cache-control", "last-modified", "etag", "content-disposition")
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in keep}

    async def _stream():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(_stream(), status_code=upstream.status_code,
                             headers=resp_headers)


# ── Web UI ───────────────────────────────────────────────────────
WEB_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 有声书</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; }
body {
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
  color:#eee; overflow:hidden;
}
/* Full-height shell: tabs on top, the active pane taking the rest. */
.shell { height:100vh; display:flex; flex-direction:column; }
.topbar {
  flex:none; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  padding:10px 18px; background:rgba(0,0,0,0.28);
  border-bottom:1px solid rgba(255,255,255,0.1);
}
.brand { font-size:15px; font-weight:600; white-space:nowrap;
         background: linear-gradient(90deg,#a78bfa,#f472b6); -webkit-background-clip:text;
         -webkit-text-fill-color:transparent; }
.topbar .spacer { flex:1 1 auto; }
.tabs { display:flex; gap:6px; }
.tab-btn {
  width:auto; margin:0; padding:7px 20px; font-size:13.5px; font-weight:600;
  border-radius:9px; border:1px solid rgba(255,255,255,0.12);
  background:rgba(255,255,255,0.06); color:#aaa; cursor:pointer; transition:all .2s;
}
.tab-btn.active { background:linear-gradient(90deg,#7c3aed,#ec4899); color:#fff;
                  border-color:transparent; }
.pane { display:none; flex:1; min-height:0; }
.pane.active { display:block; overflow-y:auto; }
.card {
  background: rgba(255,255,255,0.06); backdrop-filter: blur(12px);
  border:1px solid rgba(255,255,255,0.12); border-radius:20px;
  padding:28px 30px; width:100%; max-width:560px; margin:24px auto;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
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
/* ── 点读：独立标签页，铺满屏幕，水平翻页 ───────────────────────
   The pane is a full-height flex column: toolbar / page stage / footer.
   The stage is the measurable box the layout is paginated against, and it
   holds ONE real page plus, briefly, a ghost during a turn. Turns slide
   horizontally; nothing scrolls. */
/* The reader tab fills the window, so that is where the background image goes:
   proportionally scaled (`cover`) around its centre, never stretched, and
   clipped rather than tiled. Putting it on the reading band alone would leave
   the toolbar and footer on the page gradient instead of the image. */
#paneRead.active { display:flex; flex-direction:column; overflow:hidden; }
#paneRead { background-color:rgba(0,0,0,0.22);
            background-size:cover; background-position:center; background-repeat:no-repeat; }
.rd-toolbar {
  flex:none; display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:9px 18px; border-bottom:1px solid rgba(255,255,255,0.08);
}
.rd-toolbar .spacer { flex:1 1 auto; }
.rd-auto { margin:0; display:flex; align-items:center; gap:6px; color:#aaa;
           font-size:12.5px; cursor:pointer; }
.icon-btn { width:auto; margin:0; padding:6px 12px; font-size:12.5px; border-radius:9px;
            background:rgba(255,255,255,0.08); color:#ccc;
            border:1px solid rgba(255,255,255,0.15); cursor:pointer;
            text-decoration:none; }
.icon-btn:hover { opacity:.85; }
.icon-btn.disabled { opacity:.35; cursor:not-allowed; pointer-events:none; }
.rd-settings { flex:none; padding:12px 18px; background:rgba(0,0,0,0.3);
               border-bottom:1px solid rgba(255,255,255,0.1); font-size:12.5px; }
.rd-settings label { margin:0 0 5px; font-size:12px; }
.rd-row { margin-bottom:12px; }
.rd-row:last-child { margin-bottom:0; }
.rd-colors { display:flex; gap:7px; flex-wrap:wrap; margin:5px 0 7px; align-items:center; }
.rd-swatch { width:24px; height:24px; border-radius:50%; cursor:pointer; padding:0; margin:0;
             border:2px solid rgba(255,255,255,0.3); }
.rd-swatch.sel { border-color:#a78bfa; box-shadow:0 0 0 2px rgba(167,139,250,0.45); }

.rd-stage { position:relative; flex:1; min-height:0; overflow:hidden;
            font-size:18px; line-height:2.0; color:#eee; }
/* All padding lives on the page, never on the stage: an inset:0 page is sized
   to the containing block's PADDING box, so stage padding would push the text
   area past the visible region and clip the last lines. */
.rd-page { position:absolute; inset:0; padding:22px 6vw 16px; overflow:hidden;
           transition:transform .3s ease; }
.rd-sent { cursor:pointer; border-radius:4px; text-underline-offset:5px;
           text-decoration-thickness:2px; transition:background .15s; }
.rd-sent:hover { background:rgba(124,58,237,0.2); }
.rd-sent.cur { text-decoration:underline; }
.rd-sent.flat { cursor:default; }
.rd-sent.flat:hover { background:none; }
/* 还没合成到的句子：划掉并禁用（不属于当前目标音频） */
.rd-sent.pending { text-decoration:line-through; opacity:.45; cursor:default; }
.rd-sent.pending:hover { background:none; }

.rd-footer { flex:none; display:flex; align-items:center; gap:14px; padding:10px 18px 2px; }
.rd-btn { flex:0 0 auto; width:auto; margin:0; padding:8px 20px; font-size:13.5px;
          border-radius:10px; background:rgba(124,58,237,0.2); color:#d8b4fe;
          border:1px solid rgba(255,255,255,0.18); }
.rd-btn:hover { opacity:.85; } .rd-btn:disabled { opacity:.35; cursor:not-allowed; }
.rd-pos { font-size:12.5px; color:#999; min-width:70px; text-align:center; white-space:nowrap; }
.rd-footer .rd-prog { flex:1 1 auto; }
.rd-prog { height:6px; border-radius:3px; background:rgba(255,255,255,0.1);
           overflow:hidden; }
.rd-prog-fill { height:100%; width:0%; border-radius:3px;
                background:linear-gradient(90deg,#7c3aed,#ec4899); transition:width .25s; }
.rd-hint { flex:none; font-size:11.5px; color:#777; padding:6px 18px 8px; line-height:1.6; }
#paneRead #player { flex:none; padding:0 18px 14px; }
.reader-hint { font-size:11.5px; color:#777; margin-top:8px; line-height:1.6; }
.import-box, .recover-box { margin-top:12px; padding:12px; border-radius:12px;
                            background:rgba(0,0,0,0.25); }
input[type=file] { width:100%; font-size:12.5px; color:#aaa; margin-top:4px; }
input[type=color] { width:38px; height:26px; padding:0; border:none; background:none;
                    cursor:pointer; vertical-align:middle; }
.bk-label { color:#888; white-space:nowrap; font-size:12.5px; }
/* 服务器选择器（下拉项由后端 /api/servers 提供，配置驱动） */
.srv-wrap { display:inline-flex; align-items:center; gap:5px; white-space:nowrap;
            font-size:12.5px; color:#aaa; }
#serverPick { max-width:150px; padding:4px 8px; font-size:12.5px; cursor:pointer; }
#bkChips { display:inline-flex; gap:8px; flex-wrap:wrap; }
.bk-chip {
  display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
  border-radius:999px; background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.14); white-space:nowrap; font-size:12.5px;
}
.bk-dot { width:8px; height:8px; border-radius:50%; background:#666; flex:none; }
.bk-chip.hot { border-color:rgba(74,222,128,0.55); color:#bbf7d0; }
.bk-chip.hot .bk-dot { background:#4ade80; box-shadow:0 0 6px #4ade80; }
.bk-chip.warming { border-color:rgba(250,204,21,0.55); color:#fde68a; }
.bk-chip.warming .bk-dot { background:#facc15; box-shadow:0 0 6px #facc15; animation:bkblink 1s infinite; }
.bk-chip.cold { color:#999; }
.bk-msg { color:#a78bfa; font-size:12px; }
@keyframes bkblink { 50% { opacity:.35; } }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <span class="brand">📖 AI 有声书</span>
    <div class="tabs">
      <button class="tab-btn active" id="tabSynth">合成</button>
      <button class="tab-btn" id="tabRead">点读</button>
    </div>
    <span class="spacer"></span>
    <label class="srv-wrap" title="切换 TTS 服务器（列表来自服务端配置）">
      🖥 <select id="serverPick"></select>
    </label>
    <!-- 后端引擎实时状态（SSE 推送：启动 / 加载 N s / 就绪 / 空闲卸载） -->
    <span class="bk-label">⚙</span>
    <span id="bkChips"></span>
    <span class="bk-msg" id="bkMsg"></span>
  </div>

  <!-- 合成 -->
  <div class="pane active" id="paneSynth">
    <div class="card">
      <label>小说文本（支持整章粘贴，自动分段）</label>
      <textarea id="longText" placeholder="粘贴整本/整章文本…（支持长文，自动分段）"></textarea>
      <div class="row">
        <div>
          <label>声音</label>
          <select id="voiceLong"></select>
        </div>
        <div>
          <label>模型版本</label>
          <select id="versionLong"></select>
        </div>
        <div>
          <label>语速 <span class="speed-val" id="speedValLong">1.0x</span></label>
          <input type="range" id="speedLong" min="0.5" max="2.0" step="0.1" value="1.0">
        </div>
      </div>
      <button id="longBtn">开始合成</button>
      <button class="btn-sec" id="showRecover" style="margin-top:8px;">🔁 我有任务ID，恢复进度</button>
      <button class="btn-sec" id="sendToRead" style="margin-top:8px; display:none;">📖 一键发送到点读（合成中可用）</button>
      <button class="btn-sec" id="showImport" style="margin-top:8px;">⬆ 上传音频与分割文件</button>
      <div class="progress-wrap" id="progressWrap">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-text" id="progressText"></div>
      </div>
      <div class="task-info" id="taskInfo" style="display:none; margin-top:12px; padding:10px; border-radius:10px; background:rgba(0,0,0,0.25); font-size:12px; line-height:1.8; word-break:break-all;">
        <div>📋 任务ID: <span id="taskIdText"></span></div>
        <div style="color:#888;">⚠️ 断网/刷新后可凭任务ID恢复（2天内有效）</div>
      </div>
      <div class="recover-box" id="recoverBox" style="display:none;">
        <label>恢复进度</label>
        <input type="text" id="recoverId" placeholder="任务ID">
        <button class="btn-sec" id="recoverBtn" style="margin-top:6px;">🔄 恢复进度</button>
      </div>
      <div class="import-box" id="importBox" style="display:none;">
        <label>整章音频（.wav）</label>
        <input type="file" id="importAudio" accept=".wav,audio/wav">
        <label>分割文件（.json）</label>
        <input type="file" id="importSplit" accept=".json,application/json">
        <button class="btn-sec" id="importBtn" style="margin-top:8px;">⬆ 上传并继续点读</button>
        <div class="reader-hint">
          就是服务端原本保存的那两个文件（任务ID.wav + 任务ID.json），下载后可直接传回。<br>
          · <b>两个都传</b> → 复用音频，秒恢复，不重新合成；<br>
          · <b>只传分割文件</b> → 按同一套分句重新合成，句子边界完全一致。
        </div>
      </div>
      <div id="status"></div>
      <div class="voices-note" id="note"></div>
    </div>
  </div>

  <!-- 点读：独立标签页，铺满屏幕，水平翻页 -->
  <div class="pane" id="paneRead">
    <div class="rd-toolbar">
      <label class="rd-auto"><input type="checkbox" id="autoNext" checked> 自动连读</label>
      <span class="spacer"></span>
      <a class="icon-btn" id="dlFull">⬇ 音频</a>
      <a class="icon-btn" id="dlSplit">⬇ 分割文件</a>
      <button class="icon-btn" id="rdSettingsBtn">⚙ 外观</button>
    </div>

    <div class="rd-settings" id="rdSettings" style="display:none;">
      <div class="rd-row">
        <label>字号 <span id="rdFontVal">18px</span></label>
        <input type="range" id="rdFontSize" min="14" max="30" step="1" value="18">
      </div>
      <div class="rd-row">
        <label>文字颜色</label>
        <div class="rd-colors" id="rdColors"></div>
      </div>
      <div class="rd-row">
        <label>背景图（自行上传，可选）</label>
        <input type="file" id="rdBgFile" accept="image/*">
        <button class="btn-sec" id="rdBgClear" style="margin-top:6px;">清除背景图</button>
      </div>
      <div class="rd-row">
        <label>字体（自行上传 .ttf / .otf / .woff2，可选）</label>
        <input type="file" id="rdFontFile" accept=".ttf,.otf,.woff,.woff2,font/*">
        <button class="btn-sec" id="rdFontClear" style="margin-top:6px;">恢复默认字体</button>
      </div>
    </div>

    <div class="rd-stage" id="rdStage">
      <div class="rd-page" id="rdPage"></div>
    </div>

    <div class="rd-footer">
      <button class="rd-btn" id="rdPrev">◀ 上一页</button>
      <div class="rd-prog"><div class="rd-prog-fill" id="rdProgFill"></div></div>
      <span class="rd-pos" id="rdPos">1 / 1</span>
      <button class="rd-btn" id="rdNext">下一页 ▶</button>
    </div>
    <div class="rd-hint" id="readerHint"></div>
    <div id="player"><audio id="audio" controls></audio></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const status = $('status'), player = $('player'), audio = $('audio');
const voiceLong = $('voiceLong');
const versionLong = $('versionLong');
const longBtn = $('longBtn');

// ── 服务器选择 ───────────────────────────────────────────────
// 下拉项来自当前服务的 /api/servers（由 config_intel.yaml 的 servers: 驱动）。
// 请求**永远发本机**（同源相对路径），用 _srv=<下标> 告诉服务端转发到哪台 ——
// 这样跨域（CORS）和 https 页面里发 http 请求（混合内容）都不会发生。
let srvBase = '';        // 仅用于界面显示，不再拼进请求地址
let srvIdx = 0;
let serverList = [];
// 所有 API/音频/事件流统一走这里：改一次服务器，全站请求跟着走。
function api(path) {
  const s = serverList[srvIdx];
  if (!s || !s.url) return path;                 // 本机：原样
  return path + (path.includes('?') ? '&' : '?') + '_srv=' + srvIdx;
}

async function loadServers() {
  const sel = $('serverPick');
  try {
    // 服务器列表永远从「当前页面所在的那台」拿，不然切成远端后会拿到远端的列表
    const r = await fetch('/api/servers', { cache: 'no-store' });
    const data = await r.json();
    serverList = data.servers || [];
    // 每台后面标出探测结果：能不能直接切过去用，一眼可见
    const badge = s => {
      if (!s.kind || s.kind === 'local') return '';
      if (s.engine)             return ` ⚡ 合成引擎${s.detail ? ' · ' + s.detail : ''}`;
      if (s.kind === 'proxy')   return ` ✓ ${s.detail || ''}`;
      if (s.kind === 'backend') return ` ⚠ 仅合成接口${s.detail ? ' · ' + s.detail : ''}`;
      if (s.kind === 'pool')    return ` ⚠ 仅池 API${s.detail ? ' · ' + s.detail : ''}`;
      if (s.kind === 'offline') return ` ✕ 不可达`;
      return ` · ${s.detail || s.kind}`;
    };
    sel.innerHTML = serverList.map((s, i) =>
      `<option value="${i}">${s.name}${s.desc ? ' — ' + s.desc : ''}${badge(s)}</option>`).join('');
    // 记住上次选择；配置改过导致下标失效时退回第一台
    const saved = localStorage.getItem('bsai.server');
    const idx = saved !== null && saved < serverList.length ? Number(saved) : 0;
    sel.value = String(idx);
    applyServer(idx, true);          // 首次：只设 base，不动播放器
  } catch(e) {
    sel.innerHTML = '<option value="0">本机</option>';
    applyServer(0, true);
  }
  sel.onchange = () => applyServer(Number(sel.value), false);
}

function applyServer(idx, initial) {
  const s = serverList[idx] || { url: '' };
  // 请求都走本机转发，所以这里提示的是「选中这台之后哪些功能用不了」
  if (!initial && s.url) {
    if (s.kind === 'offline') {
      status.textContent = `⚠ ${s.name} 探测不到（${s.detail || '连接失败'}），`
        + `请求会返回 502。检查地址/服务是否启动。`;
    } else if (s.engine) {
      status.textContent = `⚡ 已切到 ${s.name}：长文/点读仍在本机，合成由它完成`
        + `（音色会映射成它自己的参考音）。`;
    } else if (s.kind === 'backend' || s.kind === 'pool') {
      status.textContent = `ℹ 已切到 ${s.name}：合成由它完成，但它没有长文任务与点读接口`
        + `（要完整功能需在那台部署 bsai 附加包）。`;
    }
  }
  srvIdx = idx;
  srvBase = String(s.url || '').replace(/\/+$/, '');   // 只用于显示
  localStorage.setItem('bsai.server', String(idx));
  if (!initial) {
    // 用户主动切换 = 换了一个后端：停播、重拉音色、重建事件流
    try { audio.pause(); } catch(_) {}
    player.style.display = 'none';
    if (typeof initBackendStatus === 'function') initBackendStatus();
  }
  loadVoices();
}

async function loadVoices() {
  // 音色和模型版本都是「每台服务器各自」的：请求照样发本机，由本机转发到
  // 选中的那台（带 _srv）。拿不到就把选择器锁掉并说明原因，
  // 免得用户以为还能用本地那套模型。
  const picked = serverList[srvIdx];
  const where = (picked && picked.name) || '本机';
  voiceLong.innerHTML = '<option>加载中…</option>';
  versionLong.innerHTML = '<option>加载中…</option>';
  try {
    const r = await fetch(api('/api/voices'), { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const opts = (data.voices || []).map(v =>
      `<option value="${v.name}">${v.name} — ${v.description || ''}</option>`).join('');
    voiceLong.innerHTML = opts || '<option>（无）</option>';
    const vers = (data.versions || []).map(v =>
      `<option value="${v.name}">${v.label || v.name}</option>`).join('');
    versionLong.innerHTML = vers || '<option>（无）</option>';
    if (data.default_version) versionLong.value = data.default_version;
    versionLong.disabled = !vers;
    var cap = data.backend_capabilities;
    $('note').textContent = `${data.voices.length} 个声音可用 · ${(data.versions || []).length} 个模型版本`
      + (cap ? ` · ${where} 后端支持: ${cap.join(' / ')}` : ` · ${where}`);
    status.textContent = '';
  } catch(e) {
    // 跨源被拦 / 服务器没起 / 域名不通，都落到这里：明确告诉用户是"这台连不上"
    voiceLong.innerHTML = '<option>（本服务器不可用）</option>';
    versionLong.innerHTML = '<option>（本服务器不可用）</option>';
    voiceLong.disabled = versionLong.disabled = true;
    $('note').textContent = `服务器「${where}」不可用：${e.message || e}`;
    status.textContent = `⚠ 无法读取 ${where} 的音色与模型列表（检查地址/证书/服务是否启动）`;
  }
}

$('speedLong').oninput = e => $('speedValLong').textContent = (+e.target.value).toFixed(1) + 'x';

const progressWrap = $('progressWrap'), progressFill = $('progressFill'), progressText = $('progressText');
let pollTimer = null;
let pollTaskId = null;      // 正在轮询的任务：同一个任务只跑一条轮询链
let pollFails = 0;          // 连续失败次数（服务器重启/断网时重试，不立刻放弃）
const POLL_MS = 800;        // 合成中的轮询间隔
const POLL_RETRY_MS = 2000; // 请求失败后的重试间隔
const POLL_MAX_FAILS = 5;   // 连续失败这么多次才放弃

function showProgress(pct, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width = pct + '%';
  progressText.textContent = label;
}

// 结束轮询链。只清理「自己那条」：换了任务以后，旧链的收尾不能把新链掐掉。
function endPoll(taskId) {
  if (pollTaskId !== taskId) return;
  clearTimeout(pollTimer);
  pollTimer = null;
  pollTaskId = null;
  longBtn.disabled = false;
}

// 轮询入口（提交任务 / 恢复任务 / 打开点读时调用）：同一个任务已经在轮询就
// 直接返回，让调用方复用现有那条链，而不是叠加第二条。
// 真正的循环在 pollTick 里 —— 入口的守卫**不能**挂在 tick 上，否则
// `setTimeout(() => pollTask(id))` 回到入口时会被自己挡住：轮询只跑一次，
// 进度条和点读页就永远停在第一帧（合成完成后也不更新）。
async function pollTask(taskId) {
  if (pollTaskId === taskId) return;      // 已经在轮询这个任务
  clearTimeout(pollTimer);
  pollTaskId = taskId;
  pollFails = 0;
  return pollTick(taskId);
}

async function pollTick(taskId) {
  if (pollTaskId !== taskId) return;      // 已换成别的任务 / 已收尾
  let p;
  try {
    const r = await fetch(api(`/api/tts/progress/${taskId}`), { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    p = await r.json();
  } catch(e) {
    if (pollTaskId !== taskId) return;
    if (++pollFails >= POLL_MAX_FAILS) {
      endPoll(taskId);
      showProgress(0, '❌ 轮询失败: ' + e + '（可凭任务ID恢复进度）');
      return;
    }
    showProgress(0, `⚠️ 与服务器断开（${e}），${POLL_RETRY_MS / 1000}s 后重试 `
      + `${pollFails}/${POLL_MAX_FAILS}…`);
    pollTimer = setTimeout(() => pollTick(taskId), POLL_RETRY_MS);
    return;
  }
  if (pollTaskId !== taskId) return;      // await 期间被换掉了
  pollFails = 0;
  // 进度接口顺带给了后端池快照：状态条跟着走，不必等下一次兜底轮询
  if (p.pool) { bkPool = p.pool; bkRender(); }

  const pct = p.total ? Math.round(p.done / p.total * 100) : 0;

  if (p.status === 'done') {
    showProgress(100, '✅ 合成完成 — 可在下方点读');
    $('sendToRead').textContent = '📖 打开点读';
    endPoll(taskId);
    // 点读里就地升级成整章音频（不重排、不跳页），否则装回合成结果
    if (partialMode) { upgradeToFinal(); return; }
    // No auto-play: a chapter can run for hours. The player is pointed at
    // the result (Range-enabled, so seeking works) and 点读 drives it.
    audio.src = api(`/api/tts/result/${taskId}`);
    audio.volume = 1.0;
    player.style.display = 'block';
    await loadReader(taskId);
    return;
  }
  if (p.status === 'error') {
    showProgress(0, '❌ 合成失败: ' + (p.error || '未知错误'));
    endPoll(taskId);
    if (partialMode) {
      $('readerHint').textContent = `⚠️ 合成中断（${p.error || '未知错误'}）：`
        + `已就绪的 ${readyCount()}/${p.total} 句仍可点读`;
    }
    return;
  }
  // 排队中（还没拿到后端）与合成中分开显示，避免「0/0 句（0%）」这种死进度
  showProgress(pct, p.status === 'queued' || !p.total
    ? '⏳ 排队中：等待空闲后端…'
    : `正在合成 ${p.done}/${p.total} 句（${pct}%）…`);
  // 合成中：partial 点读跟着进度换更长的前缀音频
  if (partialMode && splitDoc && p.done !== readyCount()) await refreshPartial();
  if (pollTaskId !== taskId) return;      // refreshPartial 期间任务被换掉
  pollTimer = setTimeout(() => pollTick(taskId), POLL_MS);
}

let currentTaskId = null;
let readerUrl = null;          // object URL of the chapter audio, once loaded

function showTaskInfo(taskId) {
  currentTaskId = taskId;
  $('taskIdText').textContent = taskId;
  $('taskInfo').style.display = 'block';
  // 任务一提交就能进点读：合成中用已完成句子的前缀音频
  $('sendToRead').style.display = 'block';
}

async function recoverTask() {
  const id = $('recoverId').value.trim();
  if (!id) {
    status.textContent = '请输入任务ID';
    return;
  }
  showTaskInfo(id);
  $('recoverBox').style.display = 'none';
  longBtn.disabled = true;
  showProgress(1, '正在查询任务进度…');
  pollTask(id);
}

$('recoverBtn').onclick = recoverTask;

$('showRecover').onclick = () => {
  const box = $('recoverBox');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  if (box.style.display === 'block') {
    $('recoverId').value = currentTaskId || '';
  }
};

$('longBtn').onclick = async () => {
  const text = $('longText').value.trim();
  if (!text) { status.textContent = '请输入文本'; return; }
  if (text.length < 50) {
    status.textContent = '长文模式建议输入 50 字以上（适合整章/整段文本）';
  }
  longBtn.disabled = true;
  showProgress(1, '正在提交任务…');
  try {
    const r = await fetch(api('/api/tts/long'), {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        text, voice: voiceLong.value, version: versionLong.value,
        speed: parseFloat($('speedLong').value)
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || '提交失败');
    showTaskInfo(data.task_id);
    pollTask(data.task_id);
  } catch(e) {
    showProgress(0, '❌ 提交失败: ' + e);
    longBtn.disabled = false;
  }
};

// ── 翻页式点读 ──────────────────────────────────────────────────
// Sentences are laid out into pages by measuring against the page box, the
// page turns on its own to follow playback, and the current sentence is shown
// underlined. Playback stops at the end of a sentence when 自动连读 is off —
// the chapter is ONE audio file, so "the end of a sentence" has to be enforced
// by watching the clock; without that, playback runs to the end of the chapter.
let splitDoc = null, curSent = -1;
let pages = [];        // page -> array of sentence indices
let pageIdx = 0;
let partialMode = false;          // 点读正用「合成到一半」的前缀音频当目标音频
let pendingAutoContinue = false;  // 读到已就绪末尾，等新片段出现后自动续读

// The seek points, one per sentence: `starts` in the split file we write,
// `offsets` in files downloaded before that rename.
function seekPoints() {
  if (!splitDoc) return [];
  const s = splitDoc.starts || splitDoc.offsets || [];
  return s.length === (splitDoc.sents || []).length ? s : [];
}

// Where sentence i ends: the next sentence's start, or null for the last one
// (it plays out the file).
function sentenceEnd(i) {
  const s = seekPoints();
  return (i + 1 < s.length) ? s[i + 1] : null;
}

// How long clicking sentence i plays for: up to the next sentence's start, and
// for the last one, to the end of the audio file.
function spanSeconds(i) {
  const s = seekPoints();
  if (!s.length) return 0;
  const end = sentenceEnd(i);
  return Math.max(0, (end === null ? (splitDoc.audio_seconds || s[i]) : end) - s[i]);
}

function fmtTime(s) {
  // Round to one decimal BEFORE splitting into m:ss — padding a value that
  // rounds up past the boundary would render "0:010.0" for 9.96s (and "0:60.0"
  // for 59.96s) instead of "0:10.0" / "1:00.0".
  const t = Math.round(s * 10) / 10;
  const m = Math.floor(t / 60), r = t - m * 60;
  return m + ':' + (r < 10 ? '0' : '') + r.toFixed(1);
}

function setDownloads(taskId) {
  $('dlFull').href = api(`/api/tts/result/${taskId}`);
  $('dlSplit').href = api(`/api/tts/split/${taskId}`);
  // Named after the server's own files, so a downloaded pair re-uploads as-is.
  $('dlFull').download = `${taskId}.wav`;
  $('dlSplit').download = `${taskId}.json`;
}

function sentenceCount() {
  return splitDoc ? (splitDoc.sents || []).length : 0;
}

function pageOfSentence(i) {
  for (let p = 0; p < pages.length; p++) if (pages[p].includes(i)) return p;
  return -1;
}

// ── 合成中提前点读：用已完成句子的「前缀音频」当目标音频 ──────────────
// 服务端 /api/tts/partial 给的是已完成片段的拼接（与最终整章音频逐样本一致），
// 所以合成推进 / 完成时只需换 URL 并保留 currentTime，时间点一个都不用改。
function partialAudioUrl(taskId, done) {
  return api(`/api/tts/partial/${taskId}/audio?v=${done || 0}`);
}

// 当前该绑定的音频：partial 任务用前缀，否则用最终整章音频。
function audioUrl() {
  return partialMode
    ? partialAudioUrl(currentTaskId, splitDoc ? (splitDoc.ready || 0) : 0)
    : api(`/api/tts/result/${currentTaskId}`);
}

// 现在可以点的句子数：partial 任务里只有已合成部分是目标音频的一部分。
function readyCount() {
  const n = sentenceCount();
  if (!splitDoc || !splitDoc.partial) return n;
  const r = splitDoc.ready != null ? splitDoc.ready : (splitDoc.done || 0);
  return Math.max(0, Math.min(r, n));
}

// 下载 = 存档（完整音频 + 描述完整音频的分割文件），只合成了一半时先禁用。
function setDownloadsEnabled(on) {
  ['dlFull', 'dlSplit'].forEach(id => {
    const a = $(id);
    a.classList.toggle('disabled', !on);
    a.setAttribute('aria-disabled', on ? 'false' : 'true');
  });
}

// 已读到当前合成进度的末尾：暂停挂起，等下一句合成出来再续读。
function waitForMore() {
  pendingAutoContinue = true;
  audio.pause();
  $('readerHint').textContent = `⏳ 已读到当前进度（已就绪 ${readyCount()}/`
    + `${sentenceCount()} 句），合成到下一句会自动继续…`;
}

// 合成推进了：换更长的前缀音频。前缀与最终音频逐样本一致，已记录的 starts
// 仍指向同一句，所以只要保住播放位置即可。
async function refreshPartial() {
  if (!partialMode || !currentTaskId) return;
  const at = audio.currentTime, playing = !audio.paused;
  let doc;
  try {
    const r = await fetch(api(`/api/tts/partial/${currentTaskId}`));
    if (!r.ok) return;
    doc = await r.json();
  } catch(_) { return; }             // 下一轮轮询再试
  const wasReady = readyCount();
  splitDoc = doc;
  if (!doc.partial) return;          // 已完成 —— 交给轮询的 done 分支升级
  if (readyCount() === wasReady) return;
  audio.src = audioUrl();
  audio.onloadedmetadata = () => {
    audio.currentTime = at;
    if (pendingAutoContinue) {       // 正卡在边界上：接着读新就绪的那一句
      pendingAutoContinue = false;
      playSentence(wasReady);
    } else if (playing) {
      audio.play().catch(() => {});
    }
  };
  renderPage();                      // 划掉的句子随 ready 减少
  updateHint();
}

// 合成完成：换成最终整章音频。前缀与最终音频逐样本一致，所以当前位置、当前句
// 与页码原样保留，只把提示和下载按钮打开。
function upgradeToFinal() {
  const at = audio.currentTime, playing = !audio.paused;
  partialMode = false;
  pendingAutoContinue = false;
  if (splitDoc) {
    splitDoc.partial = false;
    splitDoc.ready = sentenceCount();
  }
  setDownloadsEnabled(true);
  audio.src = api(`/api/tts/result/${currentTaskId}`);
  audio.onloadedmetadata = () => {
    audio.currentTime = at;
    if (playing) audio.play().catch(() => {});
  };
  renderPage();
  updateHint();
}

// 「一键发送到点读」：任务刚提交、还在合成时也能进 —— 先读已完成的部分，
// 合成追上来后自动换更长的前缀，完成时自动换最终音频。
async function openReaderNow() {
  if (!currentTaskId) {
    status.textContent = '还没有任务：先开始合成，或恢复一个任务';
    return;
  }
  const btn = $('sendToRead');
  btn.disabled = true;
  status.textContent = '正在打开点读…';
  const ok = await loadReader(currentTaskId);
  if (ok) status.textContent = '';
  btn.disabled = false;
}
$('sendToRead').onclick = openReaderNow;

// Lay the sentences out into pages by measuring against the stage: append one
// at a time to a hidden probe and break as soon as it overflows. The probe is
// a sibling of the pages so it inherits the same font and padding, and it is
// invisible so the visible page never flickers through candidate layouts.
function paginate() {
  const stage = $('rdStage');
  const avail = stage.clientHeight;
  // Fall back rather than produce an empty reader: a zero width would make the
  // probe wrap nothing and silently leave every page blank.
  const wide = stage.clientWidth || document.documentElement.clientWidth || 1024;
  if (!avail) return;              // hidden pane measures as zero — caller must show it first
  const probe = document.createElement('div');
  probe.className = 'rd-page';
  // The .rd-page class is absolutely positioned with `inset:0`, so override the
  // WHOLE geometry. Leaving `right:0` inherited next to `left:-99999px` would
  // stretch the probe to ~100000px wide: nothing would ever wrap, the measured
  // line count would be far too low, and every real page would overflow and
  // clip its tail. The width must equal the real page's box (the stage's
  // padding box, since the page sits at inset:0) or the breaks won't match.
  probe.style.cssText = 'position:absolute;left:-99999px;top:0;right:auto;'
    + 'bottom:auto;visibility:hidden;overflow:visible;';
  probe.style.width = wide + 'px';
  probe.style.height = avail + 'px';
  stage.appendChild(probe);

  const n = sentenceCount();
  pages = [];
  let cur = [];
  for (let i = 0; i < n; i++) {
    const el = document.createElement('span');
    el.className = 'rd-sent';
    el.textContent = (splitDoc.sents[i] || '') + ' ';
    probe.appendChild(el);
    if (probe.scrollHeight > avail && cur.length) {
      probe.removeChild(el);
      pages.push(cur);
      cur = [];
      probe.innerHTML = '';
      probe.appendChild(el);
    }
    cur.push(i);
  }
  if (cur.length) pages.push(cur);
  if (!pages.length) pages = [[]];
  probe.remove();
  if (pageIdx > pages.length - 1) pageIdx = pages.length - 1;
}

// Fill a page element with the given sentences, marking the one being read.
function fill(el, list) {
  const flat = seekPoints().length !== sentenceCount();
  const ready = readyCount();
  el.innerHTML = '';
  list.forEach(i => {
    const sp = document.createElement('span');
    const pending = !!splitDoc && splitDoc.partial && i >= ready;
    sp.className = 'rd-sent' + (flat ? ' flat' : '') + (pending ? ' pending' : '')
      + (i === curSent ? ' cur' : '');
    sp.textContent = (splitDoc.sents[i] || '') + ' ';
    if (!flat && !pending) sp.onclick = e => { e.stopPropagation(); playSentence(i); };
    el.appendChild(sp);
  });
}

// Put the transform back to rest without animating the jump.
function restTransform(el) {
  el.style.transition = 'none';
  el.style.transform = 'translateX(0)';
  void el.offsetWidth;
  el.style.transition = '';
}

function updateNav() {
  $('rdPrev').disabled = pageIdx <= 0;
  $('rdNext').disabled = pageIdx >= pages.length - 1;
  $('rdPos').textContent = `${pageIdx + 1} / ${pages.length}`;
  updateProgress();
}

// Instant render — used on load, on re-pagination, and for jumps.
function renderPage() {
  const box = $('rdPage');
  fill(box, pages[pageIdx] || []);
  restTransform(box);
  updateNav();
}

// Horizontal page turn: the incoming page slides in from the side while the
// current one leaves in the other direction — no vertical scrolling anywhere.
function slideTo(p, dir) {
  const box = $('rdPage');
  const stage = $('rdStage');
  if (p === pageIdx || p < 0 || p > pages.length - 1) return;
  // A rapid second click can land mid-slide; drop the previous ghost instead of
  // stacking pages on the stage.
  [...stage.children].forEach(n => { if (n !== box) n.remove(); });
  const ghost = document.createElement('div');
  ghost.className = 'rd-page';
  fill(ghost, pages[p] || []);
  ghost.style.transition = 'none';
  ghost.style.transform = `translateX(${dir > 0 ? 100 : -100}%)`;
  stage.appendChild(ghost);
  void ghost.offsetWidth;                       // commit the off-screen start
  ghost.style.transition = '';
  ghost.style.transform = 'translateX(0)';
  box.style.transform = `translateX(${dir > 0 ? -100 : 100}%)`;
  pageIdx = p;
  updateNav();
  let done = false;
  const settle = () => {
    if (done) return;
    done = true;
    fill(box, pages[pageIdx] || []);
    restTransform(box);
    ghost.remove();
  };
  ghost.addEventListener('transitionend', settle, { once: true });
  setTimeout(settle, 400);                      // fallback: no transitionend (hidden tab)
}

// Programmatic jump: instant, no animation (used for far jumps and by callers).
function gotoPage(p) {
  pageIdx = Math.max(0, Math.min(pages.length - 1, p));
  renderPage();
}

// User-facing turn: slides horizontally.
function turnPage(delta) {
  slideTo(pageIdx + delta, delta);
}

function stopReading() {
  curSent = -1;
  markCurrent();
  updateProgress();
}

// ── 标签页 ──────────────────────────────────────────────────────
function switchTab(name) {
  const read = name === 'read';
  $('tabSynth').classList.toggle('active', !read);
  $('tabRead').classList.toggle('active', read);
  $('paneSynth').classList.toggle('active', !read);
  $('paneRead').classList.toggle('active', read);
  if (read) {
    // The stage only has a height once its pane is displayed (reading
    // clientHeight below forces the layout), so lay out now, not before.
    if (!splitDoc) {
      $('readerHint').textContent = '还没有任务：先在「合成」里合成，或上传之前的音频与分割文件。';
      return;
    }
    paginate();
    renderPage();
  }
}

$('tabSynth').onclick = () => switchTab('synth');
$('tabRead').onclick = () => switchTab('read');

// Show the reader tab and lay it out — pagination needs a visible stage.
function showReaderTab() {
  switchTab('read');
}

// 点读的提示行：partial 任务显示「已就绪 k/n」并说明划线句。
function updateHint() {
  if (!splitDoc) {
    $('readerHint').textContent = '还没有任务：先在「合成」里合成，或上传之前的音频与分割文件。';
    return;
  }
  const n = sentenceCount();
  const seekable = seekPoints().length === n && n > 0;
  if (splitDoc.partial) {
    $('readerHint').textContent = seekable
      ? `⏳ 合成中：已就绪 ${readyCount()}/${n} 句 · 划线句还没合成（读完会自动继续）`
      : `⏳ 合成中：已就绪 ${readyCount()}/${n} 句 · 已完成部分可播放`;
  } else {
    $('readerHint').textContent = seekable
      ? `${n} 句 · 整章 ${fmtTime(splitDoc.audio_seconds || 0)} · 点击句子朗读，`
        + `← → 翻页（关闭自动连读则每句播完即停）`
      : `⚠️ 该任务的分割文件不含每句起点，无法按句跳转；整章音频仍可用上面的播放器收听`;
  }
}

// 装载一份点读文档：合成中的 partial 文档，或已完成 / 导入的最终文档。
// 绑定音频源、切到点读页、排版。
async function showDoc(doc) {
  splitDoc = doc;
  partialMode = !!doc.partial;
  pendingAutoContinue = false;
  pageIdx = 0;
  player.style.display = 'block';
  setDownloadsEnabled(!partialMode);
  audio.src = audioUrl();
  // Show the reader before paginating — a hidden stage measures as zero height,
  // which would put every sentence on its own page. The stored font has to be
  // in place too: a different face re-flows the text and moves the breaks.
  showReaderTab();
  await ensureFontReady();
  applyAppearance();
  stopReading();
  paginate();
  renderPage();
  updateHint();
  // 深链可能落在还在合成的任务上：继续盯着它，音频会自己升级成整章。
  if (partialMode) pollTask(currentTaskId);
}

async function loadReader(taskId) {
  currentTaskId = taskId;
  setDownloads(taskId);
  splitDoc = null;
  try {
    // 合成中的任务也能读：这个端点给已就绪句子的前缀音频与时间点，完成后
    // 与 /api/tts/split 返回同一份最终文档。
    const r = await fetch(api(`/api/tts/partial/${taskId}`));
    if (!r.ok) throw new Error(await r.text());
    await showDoc(await r.json());
    return true;
  } catch(e) {
    $('readerHint').textContent = '❌ 读取分割文件失败: ' + e;
    status.textContent = '❌ 读取点读数据失败: ' + e;
    return false;
  }
}

// The bar tracks the sentence number, not the clock: the sentence being read,
// or — when idle — the first sentence of the page you are looking at.
function updateProgress() {
  const n = sentenceCount();
  if (!n) { $('rdProgFill').style.width = '0%'; return; }
  const anchor = curSent >= 0 ? curSent : ((pages[pageIdx] || [0])[0] || 0);
  $('rdProgFill').style.width = (((anchor + 1) / n) * 100).toFixed(2) + '%';
}

function markCurrent() {
  const box = $('rdPage');
  for (const el of box.children) el.classList.remove('cur');
  if (curSent < 0) return;
  const cur = pages[pageIdx] || [];
  const at = cur.indexOf(curSent);
  if (at >= 0 && box.children[at]) box.children[at].classList.add('cur');
}

function playSentence(i) {
  if (!splitDoc) return;
  const n = sentenceCount(), offs = seekPoints();
  if (i < 0 || i >= n) { if (i >= n) { audio.pause(); stopReading(); } return; }
  if (offs.length !== n) return;
  if (i >= readyCount()) {          // 还没合成到这一句（划线句不可点）
    $('readerHint').textContent = `⏳ 第 ${i + 1} 句还没合成完（已就绪 `
      + `${readyCount()}/${n} 句），合成到了会自动继续`;
    return;
  }
  curSent = i;
  // Follow the reading: turn to the page holding this sentence — sliding when
  // it is the next/previous page, jumping instantly for a far-away sentence.
  const p = pageOfSentence(i);
  if (p >= 0 && p !== pageIdx) {
    if (Math.abs(p - pageIdx) === 1) slideTo(p, p > pageIdx ? 1 : -1);
    else { pageIdx = p; renderPage(); }
  } else {
    markCurrent();
  }
  updateProgress();
  player.style.display = 'block';
  const seekAndPlay = () => {
    const target = offs[i] || 0;
    // 自动连读跨过句边界时，播放位置本来就在 target 附近 —— timeupdate 有
    // 250ms 粒度，通常还超前一点。此时再赋一次 currentTime 会让浏览器重新走
    // 一次 seek（Range 请求 + 解码器重置），听感就是"每句开头卡一下"，而且
    // 倒回去的那几个字会重复。位置已经对齐就直接续播，不碰 currentTime。
    const drift = audio.currentTime - target;
    if (!(drift >= -0.05 && drift <= 0.35)) audio.currentTime = target;
    audio.play().catch(() => {
      $('readerHint').textContent = '⚠️ 自动播放被浏览器阻止，请点击下方的播放器播放';
    });
  };
  // The <audio> element already points at this task's result after a synthesis
  // or recovery; only (re)bind when the source is not this task's audio.
  const want = audioUrl();
  if (!audio.src || !audio.src.split('?')[0].endsWith(want.split('?')[0])) {
    audio.src = want;
    audio.onloadedmetadata = seekAndPlay;
  } else if (audio.readyState >= 1) {
    seekAndPlay();
  } else {
    audio.onloadedmetadata = seekAndPlay;
  }
}

// Stop at the end of the current sentence. 自动连读 off means exactly that —
// otherwise pressing play on one sentence would read the rest of the chapter.
audio.ontimeupdate = () => {
  if (curSent < 0 || audio.paused) return;
  const end = sentenceEnd(curSent);
  if (end === null) return;                      // last sentence: play it out
  if (audio.currentTime < end - 0.02) return;
  if (!$('autoNext').checked) { audio.pause(); stopReading(); return; }
  // 合成中读到「已就绪末尾」：下一句还没有音频，挂起等它合成出来
  if (splitDoc && splitDoc.partial && curSent + 1 >= readyCount()) {
    waitForMore();
    return;
  }
  playSentence(curSent + 1);
};

audio.onended = () => {
  // Only reachable on the last sentence (earlier ones are cut off by
  // ontimeupdate), so there is nothing left to advance to.
  if (!$('autoNext').checked || curSent < 0) { stopReading(); return; }
  // 合成中：前缀音频放完了 —— 等新片段，而不是以为读到了章末
  if (splitDoc && splitDoc.partial && curSent + 1 >= readyCount()) {
    waitForMore();
    return;
  }
  playSentence(curSent + 1);
};

$('rdPrev').onclick = () => turnPage(-1);
$('rdNext').onclick = () => turnPage(1);

// ← → turn pages while the reader tab is open (and no text field has focus).
document.addEventListener('keydown', e => {
  if (!$('paneRead').classList.contains('active')) return;
  const t = document.activeElement && document.activeElement.tagName;
  if (t === 'INPUT' || t === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') { turnPage(-1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { turnPage(1); e.preventDefault(); }
});

// ── 外观：字号 / 颜色 / 背景图 / 字体（自行上传，尽量记住）──────
const RD_FONT_STACK = '-apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
const RD_SWATCHES = ['#eeeeee', '#d8d0c0', '#c9d6c0', '#1b1b1b', '#3b2f2f'];
const rdStore = {
  get(k, d) { try { return localStorage.getItem('bsai.rd.' + k) || d; } catch(_) { return d; } },
  set(k, v) { try { localStorage.setItem('bsai.rd.' + k, v); return true; } catch(_) { return false; } },
  del(k) { try { localStorage.removeItem('bsai.rd.' + k); } catch(_) {} },
};
let rdFontFamily = null;
let rdBgSession = '';     // background too large to remember across reloads

// Appearance: text settings go on the STAGE (pages and the hidden measuring
// probe inherit them); the background image goes on the whole reader PANE so it
// fills the screen rather than only the reading band. The pane's CSS carries
// `background-size:cover; background-position:center`, which scales the image
// proportionally about its centre to cover the area — never stretched or tiled.
function applyAppearance() {
  const stage = $('rdStage');
  const pane = $('paneRead');
  const size = rdStore.get('size', '18');
  const color = rdStore.get('color', '#eeeeee');
  $('rdFontSize').value = size;
  $('rdFontVal').textContent = size + 'px';
  stage.style.fontSize = size + 'px';
  stage.style.color = color;
  [...$('rdColors').children].forEach(sw =>
    sw.classList.toggle('sel', sw.dataset.color === color));
  const bg = rdStore.get('bg', '') || rdBgSession;
  // The gradient is a readability scrim layered over the image, so text stays
  // legible whatever picture was uploaded. The url() MUST be quoted: an
  // unquoted url() token may not contain quotes, spaces or parentheses, and a
  // browser silently drops the whole declaration when it does — which would
  // make the background disappear with no error.
  pane.style.backgroundImage = bg
    ? `linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url("${bg}")`
    : '';
  stage.style.fontFamily = rdFontFamily ? `"${rdFontFamily}", ${RD_FONT_STACK}` : RD_FONT_STACK;
}

function buildSwatches() {
  const wrap = $('rdColors');
  wrap.innerHTML = '';
  RD_SWATCHES.forEach(c => {
    const sw = document.createElement('button');
    sw.className = 'rd-swatch';
    sw.dataset.color = c;
    sw.style.background = c;
    sw.title = c;
    sw.onclick = () => { rdStore.set('color', c); applyAppearance(); };
    wrap.appendChild(sw);
  });
  const custom = document.createElement('input');
  custom.type = 'color';
  custom.id = 'rdColorPick';
  custom.title = '自定义颜色';
  custom.oninput = () => { rdStore.set('color', custom.value); applyAppearance(); };
  custom.value = rdStore.get('color', '#eeeeee');
  wrap.appendChild(custom);
}

function readAsDataURL(file) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = () => rej(fr.error);
    fr.readAsDataURL(file);
  });
}

$('rdSettingsBtn').onclick = () => {
  const box = $('rdSettings');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
};

$('rdFontSize').oninput = e => {
  rdStore.set('size', e.target.value);
  applyAppearance();
  // Type size changes the line breaks, so the page layout has to be redone.
  stopReading();
  paginate();
  renderPage();
};

$('rdBgFile').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const url = await readAsDataURL(f);
    // Too big for localStorage: keep it for this session only, and say so.
    // It still goes through applyAppearance(), so the screen-filling scaling
    // and the readability scrim apply exactly as for a remembered image.
    rdBgSession = rdStore.set('bg', url) ? '' : url;
    applyAppearance();
    if (rdBgSession) {
      $('readerHint').textContent = '⚠️ 背景图已应用，但太大无法记住（刷新后需重新上传）';
    }
  } catch(err) {
    $('readerHint').textContent = '❌ 背景图读取失败: ' + err;
  }
};

$('rdBgClear').onclick = () => {
  rdStore.del('bg');
  rdBgSession = '';
  $('rdBgFile').value = '';
  applyAppearance();
};

$('rdFontFile').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const url = await readAsDataURL(f);
    const name = 'rdUserFont-' + Date.now();
    // Quoted for the same reason as the background: an unquoted url() token
    // rejects data URLs containing quotes/spaces and would throw here.
    const face = new FontFace(name, `url("${url}")`);
    await face.load();
    document.fonts.add(face);
    rdFontFamily = name;
    if (!rdStore.set('font', url)) {
      $('readerHint').textContent = '⚠️ 字体已应用，但太大无法记住（刷新后需重新上传）';
    }
    applyAppearance();
    // A different face re-flows the text, so re-paginate.
    stopReading();
    paginate();
    renderPage();
  } catch(err) {
    $('readerHint').textContent = '❌ 字体加载失败（需要 .ttf / .otf / .woff2 字体文件）: ' + err;
  }
};

$('rdFontClear').onclick = () => {
  rdStore.del('font');
  rdFontFamily = null;
  $('rdFontFile').value = '';
  applyAppearance();
  stopReading();
  paginate();
  renderPage();
};

// Loads a previously uploaded font from storage, once. Called before layout so
// the page breaks match the face the text will actually be rendered in.
async function ensureFontReady() {
  if (rdFontFamily) return;
  const url = rdStore.get('font', '');
  if (!url) return;
  try {
    const name = 'rdUserFont';
    const face = new FontFace(name, `url(${url})`);
    await face.load();
    document.fonts.add(face);
    rdFontFamily = name;
  } catch(_) {
    rdStore.del('font');   // stale entry (e.g. the stored data is corrupt)
  }
}

// ── 上传：整章音频(.wav) + 分割文件(.json) ──────────────────────
$('showImport').onclick = () => {
  const box = $('importBox');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
};

$('importBtn').onclick = async () => {
  const splitFile = $('importSplit').files[0];
  const audioFile = $('importAudio').files[0];
  if (!splitFile) { status.textContent = '请选择分割文件(.json)'; return; }
  const btn = $('importBtn');
  btn.disabled = true;
  status.textContent = '正在上传…';
  try {
    const fd = new FormData();
    fd.append('split', splitFile);
    if (audioFile) fd.append('audio', audioFile);
    const r = await fetch(api('/api/tts/import'), { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || '上传失败');
    showTaskInfo(d.task_id);
    $('importBox').style.display = 'none';
    longBtn.disabled = true;
    if (d.reused_audio) {
      status.textContent = `✅ 已恢复 ${d.total} 句（复用上传的音频）`;
      showProgress(100, '✅ 点读就绪');
    } else {
      status.textContent = (d.warning ? '⚠️ ' + d.warning + '；' : '')
        + `已上传 ${d.total} 句，按分割文件重新合成中…`;
      showProgress(1, '正在重新合成…');
    }
    pollTask(d.task_id);
  } catch(e) {
    status.textContent = '❌ ' + e;
  } finally {
    btn.disabled = false;
  }
};

// ── 后端引擎状态条：初始快照 + SSE 实时推送 + 兜底轮询 ──
let bkPool = [], bkEvents = [], bkPulling = false;

function bkRender() {
  const chips = $('bkChips'), msg = $('bkMsg');
  const warm = bkPool.filter(b => b.state === 'warming');
  const hot  = bkPool.filter(b => b.state === 'hot');
  const cold = bkPool.filter(b => b.state === 'cold');
  chips.innerHTML = '';
  [...hot, ...warm, ...cold].forEach(b => {
    const el = document.createElement('span');
    el.className = 'bk-chip ' + b.state;
    let label = b.port + (b.model ? ' · ' + b.model : '');
    if (b.state === 'hot') label += ' · 就绪';
    else if (b.state === 'warming') label += ' · 加载中' + (b.elapsed != null ? ` ${b.elapsed}s` : '');
    else label += ' · 未加载';
    el.innerHTML = `<span class="bk-dot"></span>${label}`;
    chips.appendChild(el);
  });
  const last = bkEvents.length ? bkEvents[bkEvents.length - 1] : null;
  if (warm.length) {
    msg.textContent = last && last.state === 'warming'
      ? last.msg
      : `后端 ${warm.map(b => b.port).join('/')} 正在加载模型…首次合成约需等待 90 秒`;
    msg.style.color = '#fde68a';
  } else if (last) {
    msg.textContent = last.msg;
    msg.style.color = '#a78bfa';
  } else if (!hot.length) {
    msg.textContent = '空闲：首次合成会自动加载模型（约 90 秒）';
    msg.style.color = '#888';
  } else {
    msg.textContent = '';
  }
}

// 拉一次完整快照。SSE 只是「快」的那条路，不能当成唯一来源：连接被代理
// 缓冲 / 手机休眠后不报错也不再送事件时，界面就会永久停在旧状态。
function bkPull() {
  if (bkPulling) return;
  bkPulling = true;
  fetch(api('/api/backend/status'), { cache: 'no-store' })
    .then(r => r.json())
    .then(d => { bkPool = d.backends || []; bkEvents = (d.events || []).slice(-1); bkRender(); })
    .catch(() => {})
    .then(() => { bkPulling = false; });
}

let bkEventSource = null;
function initBackendStatus() {
  bkPull();
  // 兜底轮询：状态条 5 s 对齐一次（也让「加载中 N s」跟着走）
  setInterval(bkPull, 5000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) bkPull();     // 切回前台/标签页立刻对齐
  });
  try {
    if (bkEventSource) bkEventSource.close();   // 切服务器时先断旧连接
    const es = bkEventSource = new EventSource(api('/api/backend/events'));
    es.onopen = () => bkPull();         // 重连成功：断线期间的事件已丢，补一次快照
    es.onmessage = (e) => {
      try {
        const m = JSON.parse(e.data);
        if (m.type === 'snapshot') {
          bkPool = m.backends || [];
          bkEvents = (m.events || []).slice(-1);
        } else if (m.type === 'backend') {
          if (!bkPool.some(p => p.port === m.port)) { bkPull(); return; }  // 池子变了
          bkPool = bkPool.map(p => p.port === m.port
            ? { ...p, state: m.state, elapsed: m.state === 'cold' ? null : p.elapsed }
            : p);
          bkEvents = [{ msg: m.msg, state: m.state }];
        } else {
          return;                        // ping 等保活帧：不改状态
        }
        bkRender();
      } catch(_) {}
    };
    es.onerror = () => { bkPull(); };    // EventSource 会自连；同时立刻补一次
  } catch(_) {}
}

// ── 深链：?task=<id>&key=<key> 直接打开某个任务的点读（可收藏/分享）──
(function initFromUrl() {
  const q = new URLSearchParams(location.search);
  const id = q.get('task');            // 旧链接里多余的 &key=... 直接忽略
  if (!id) return;
  showTaskInfo(id);
  player.style.display = 'block';
  showProgress(100, '已从链接载入任务');
  status.textContent = '已载入任务 ' + id;
  loadReader(id);   // 合成中/已完成都能读：内部挑前缀或最终音频
})();

initBackendStatus();
buildSwatches();
applyAppearance();
loadServers();   // 读 /api/servers → 渲染下拉 → applyServer() 里会调 loadVoices()
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return WEB_HTML


@app.get("/api/health")
async def health():
    pool = _backend_snapshot()
    hot = [b["port"] for b in pool if b["state"] == "hot"]
    warming = [b["port"] for b in pool if b["state"] == "warming"]
    if hot:
        status = "ok"
    elif warming:
        status = "warming"
    else:
        status = "idle"
    return {"status": status, "backend": hot or warming or "none-loaded",
            "voices": len(VOICES), "pool": pool}


@app.get("/api/backend/status")
async def backend_status():
    """Current backend state snapshot + recent lifecycle events (for polling UIs)."""
    with _backend_event_lock:
        recent = list(_backend_events[-20:])
    return {"backends": _backend_snapshot(), "events": recent}


@app.get("/api/backend/events")
async def backend_events_sse():
    """Server-Sent Events: backend lifecycle (starting / loading N s / ready /
    stopped). Sends a snapshot on connect, then pushes live events. Any
    frontend (this page or an external reader UI) can consume it with
    `new EventSource('/api/backend/events')`."""
    import asyncio
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    with _backend_event_lock:
        if len(_backend_subs) >= 64:   # sanity cap — drop stale clients
            _backend_subs.clear()
        _backend_subs.append((loop, q))

    def frame(obj):
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    async def gen():
        try:
            with _backend_event_lock:
                recent = list(_backend_events[-20:])
            yield frame({"type": "snapshot", "backends": _backend_snapshot(),
                         "events": recent})
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield frame({"type": "ping"})   # keep the connection warm
                    continue
                yield frame(ev)
        finally:
            with _backend_event_lock:
                try:
                    _backend_subs.remove((loop, q))
                except ValueError:
                    pass

    return StreamingResponse(gen(), media_type="text/event-stream")


_SRV_PROBE_CACHE = {}      # url -> (ts, kind, detail)
_SRV_PROBE_TTL = 60.0      # 秒；别让下拉框每次都去打远端


def _probe_server(url):
    """这台远端能承担到什么程度。前端请求都由本机转发，所以这里只看「对端有什么接口」。

    返回 (kind, detail)：
      proxy   —— 有 /api/servers：完整的 bsai 代理，音色/长文/点读全都有
      backend —— 有 /api/voices 但没有 /api/servers：装了我们补丁的后端/池服务，
                 能自报模型与音色、能合成，但没有长文任务与点读
      pool    —— 只有 /tts_batch 之类的纯池 API
      offline —— 连不上
    """
    import time as _t
    now = _t.time()
    hit = _SRV_PROBE_CACHE.get(url)
    if hit and now - hit[0] < _SRV_PROBE_TTL:
        return hit[1], hit[2]
    kind, detail = "offline", ""
    base = url.rstrip("/")
    try:
        # 先看是不是完整代理（只有它才有 /api/servers）
        try:
            sr = requests.get(base + "/api/servers", timeout=2.0)
        except Exception:
            sr = None
        if sr is not None and sr.status_code == 200:
            kind = "proxy"
            try:
                d = sr.json()
                detail = f"{len(d.get('servers') or [])} 台互联"
            except Exception:
                detail = ""
            _SRV_PROBE_CACHE[url] = (now, kind, detail)
            return kind, detail

        r = requests.get(base + "/api/voices", timeout=2.0)
        if r.status_code == 200:
            kind = "backend"
            try:
                d = r.json()
                detail = f"{len(d.get('versions') or [])} 个模型 / {len(d.get('voices') or [])} 音色"
            except Exception:
                detail = ""
        else:
            # 不是代理，看是不是池 API
            try:
                h = requests.get(url.rstrip("/") + "/health", timeout=2.0)
                j = h.json() if h.status_code == 200 else {}
                if isinstance(j, dict) and "pool_size" in j:
                    kind = "pool"
                    detail = f"{j.get('pool_size')} 实例 / {','.join(j.get('devices') or [])}"
                elif h.status_code == 200:
                    kind = "other"
                    detail = f"HTTP {h.status_code}"
                else:
                    kind = "offline"
                    detail = f"HTTP {r.status_code}"
            except Exception as e:
                kind, detail = "offline", str(e)[:60]
    except Exception as e:
        kind, detail = "offline", str(e)[:60]
    _SRV_PROBE_CACHE[url] = (now, kind, detail)
    return kind, detail


@app.get("/api/servers")
async def servers():
    """前端服务器下拉框的数据源 —— 完全由 config_intel.yaml 的 servers: 驱动。

    url 为空 = 同源相对路径（本机）；填完整根地址 = 前端把请求发到那台机器。
    每台都会探测一次（60 s 缓存），告诉前端它到底是「完整代理」还是「纯池 API」，
    以及会不会被浏览器的混合内容策略拦掉 —— 免得切过去才发现用不了。
    """
    import asyncio as _aio
    raw = [s for s in (config.get("servers") or [])
           if isinstance(s, dict) and s.get("name")]
    if not raw:
        raw = [{"name": "本机", "url": "", "desc": "同源"}]

    async def one(s):
        url = str(s.get("url") or "")
        item = {"name": s["name"], "url": url, "desc": s.get("desc", "")}
        if not url:
            item.update(kind="local", detail="同源", engine=False)
            return item
        kind, detail = await _aio.to_thread(_probe_server, url)
        item.update(kind=kind, detail=detail,
                    # 配了 engine 的机器只借算力，任务/点读仍在本机
                    engine=bool(_engine_config(s["name"])))
        return item

    out = await _aio.gather(*[one(s) for s in raw])
    return {"servers": list(out), "current": out[0]["name"]}


@app.get("/api/voices")
async def voices():
    return {
        "voices": [{"name": k, "description": v.get("description", "")}
                   for k, v in VOICES.items()],
        "versions": [{"name": k,
                      "label": MODEL_PROFILES[k].get("label", k),
                      "sample_steps": MODEL_PROFILES[k].get("sample_steps")}
                     for k in MODEL_KEYS],
        "default_version": DEFAULT_MODEL,
        # 这台机器的能力来源：配置 ∩ 后端声明。前端拿它判断"该服务器支持什么"，
        # 也为将来多机器合并展示留字段。
        "backend_capabilities": BACKEND_CAPS,
        "server_name": (config.get("servers") or [{}])[0].get("name", ""),
    }


class TTSRequest(BaseModel):
    speaker: str = ""  # 发送者名字，可为空（测试/系统消息）
    text: str = Field(..., min_length=1, max_length=200)
    voice: str = "default"
    version: str = ""  # v4 / v5turbo; empty = server default
    speed: float = Field(1.0, ge=0.5, le=2.0)


def _candidate_ports(voice, model_key=None):
    """Affinity port first, then the other ports of the same model (failover)."""
    key = _resolve_model_key(model_key)
    affinity = voice_backend_port(voice, key)
    base = MODEL_PORT_BASE[key]
    others = [p for p in range(base, base + PORTS_PER_MODEL) if p != affinity]
    return [affinity] + others


# Silence-verdict calibration (measured on this setup):
#   crash        → 1 s of all-zero samples (every window rms ≈ 0)
#   frontend bug → near-silent clip, global rms ~0.001, worst window ≲ 0.005
#   real speech  → audible-region rms ≳ 0.02, even for a one-word answer
_SIL_WIN_MS = 25          # analysis window for the energy profile
_SIL_SPEAK_RMS = 0.010    # no window this loud → nothing speech-like at all
_SIL_AUDIBLE_RMS = 0.004  # windows above this count as audible content
_SIL_MIN_ACTIVE_S = 0.04  # rejects single-window (25 ms) pops; the level
                          # check above does the real gating — keep this low
                          # so soft one-syllable answers are never flagged


def _energy_profile(data, sr):
    """(peak_window_rms, active_seconds) — windowed energy stats of 1-D audio.
    Caller guarantees len(data) >= one window."""
    win = max(1, int(sr * _SIL_WIN_MS / 1000))
    seg = data[: (len(data) // win) * win].reshape(-1, win)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    active = float((rms > _SIL_AUDIBLE_RMS).sum()) * win / sr
    return float(rms.max()), active


def _describe_wav(wav_bytes):
    """One-line energy summary for retry logs, e.g.
    'dur=0.98s gRMS=0.0021 peakWin=0.0034 active=0.00s' — makes every SILENT
    verdict auditable from the log alone (crash audio: peakWin≈0; soft but
    real speech: peakWin ≳ 0.02)."""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return "unparseable"
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        return "empty"
    g = float((data ** 2).mean() ** 0.5)
    win = max(1, int(sr * _SIL_WIN_MS / 1000))
    if len(data) < win:
        return f"dur={len(data)/sr:.2f}s gRMS={g:.4f} (sub-window clip)"
    peak, active = _energy_profile(data, sr)
    return f"dur={len(data)/sr:.2f}s gRMS={g:.4f} peakWin={peak:.4f} active={active:.2f}s"


def _audio_quality(wav_bytes, text):
    """整段不静音还不够 —— 模型偶尔会「空转」：AR 一直吐 token 却不发声，
    结果前半段（甚至 20 多秒）全是静音，直到 early-stop。听感就是"这句开头
    半天没声音"。按「每字时长」和「最长连续静音」判一次，不合格就重合成。

    返回 (ok, reason)。阈值取自实测：正常朗读 0.20~0.40 s/字，
    坏样本 0.43~0.60 s/字且伴随 4~23 s 的静音段。"""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return True, ""                      # 读不出来就交给上层原有逻辑
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(1)
    if not len(data) or not sr:
        return True, ""
    seconds = len(data) / sr
    per_char = seconds / max(1, len(text or ""))
    # 最长连续静音（幅度 < 0.01 视为没声音）
    silent = np.abs(data) < 0.01
    edges = np.flatnonzero(np.diff(np.concatenate(([0], silent.view(np.int8), [0]))))
    longest = max([(edges[i + 1] - edges[i]) / sr for i in range(0, len(edges), 2)] or [0])
    if longest > 4.0 and longest > seconds * 0.2:
        return False, f"{longest:.1f}s 静音 / {seconds:.1f}s"
    if per_char > 0.62:
        return False, f"{per_char:.2f}s 每字（{seconds:.1f}s / {len(text)}字）"
    return True, ""


def _is_silent_wav(wav_bytes):
    """Return True if a WAV's audio contains no audible speech.

    GPT-SoVITS returns HTTP 200 with a 1s 16kHz all-zero WAV when its
    internal inference crashes (e.g. bad allocation on XPU), and near-silent
    clips (rms ~0.001, a faint blip over <1 s) when the frontend can't map
    the text (e.g. English forced through text_lang=zh). Both must be treated
    as failures so the retry/failover path runs instead of persisting a silent
    fragment.

    The verdict is deliberately NOT a global-RMS threshold: the vocoder pads
    ~0.5 s of silence in front of every clip, so a short answer ("嗯。" ≈ 0.3 s
    of speech) is mostly padding and its global RMS dilutes toward the crash
    level — the old whole-file test misjudged such normal short texts as
    silent, burning retries and triggering the "嗯嗯。" padding fallback. The
    signal is windowed instead and judged on its audible part, which padding
    cannot dilute:
      1. level    — the loudest window must reach speech level (crash audio
                    and faint blips never do; real speech peaks well above);
      2. duration — audible content must span more than one window, so an
                    isolated vocoder pop cannot pass the level check alone.
    """
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return True  # unparseable → treat as bad
    if data.size == 0:
        return True
    if data.ndim > 1:
        data = data.mean(axis=1)
    win = max(1, int(sr * _SIL_WIN_MS / 1000))
    if len(data) < win:  # sub-window clip — judge it whole on level
        return float((data ** 2).mean() ** 0.5) <= _SIL_SPEAK_RMS
    peak, active = _energy_profile(data, sr)
    return peak < _SIL_SPEAK_RMS or active < _SIL_MIN_ACTIVE_S


SILENT_DUMP_DIR = SCRIPT_DIR / "silent_dumps"  # rejected silent-200 responses
SILENT_DUMP_KEEP = 20                          # newest files kept


def _dump_rejected_wav(text, wav_bytes):
    """Persist a rejected silent-200 response so "was it REALLY silent?" is
    answerable after the fact — the retry loop would otherwise discard the
    bytes and the verdict would be uncheckable. Keeps only the newest
    SILENT_DUMP_KEEP files. Returns the file name for the log ('-' on failure)."""
    try:
        import re as _re
        SILENT_DUMP_DIR.mkdir(exist_ok=True)
        tag = time.strftime("%m%d-%H%M%S")
        safe = _re.sub(r'[^\w\u4e00-\u9fff]+', '_', text)[:12].strip('_') or 'n_a'
        path = SILENT_DUMP_DIR / f"{tag}_{safe}.wav"
        path.write_bytes(wav_bytes)
        dumps = sorted(SILENT_DUMP_DIR.glob("*.wav"),
                       key=lambda p: p.stat().st_mtime)
        for old in dumps[:-SILENT_DUMP_KEEP]:
            old.unlink(missing_ok=True)
        return path.name
    except Exception as e:
        logger.warning("Silent-dump failed: %s", e)
        return "-"


def _pad_short_text(text):
    """If a cleaned text's first sentence has <4 speakable chars, repeat its
    core until it reaches 4 (e.g. "咔嗒。" → "咔嗒咔嗒。", "6" → "6666。").

    Why 4: GPT-SoVITS's TextPreprocessor pre_seg_text() used to prefix such
    texts with "。", which on this setup makes T2S decode immediate-EOS →
    noise-floor audio (silent_dumps forensics). The backend-side prefix is
    patched out, but this keeps the proxy safe against a revert/upgrade.
    Single-shot danmaku path ONLY (pad_short=True) — audiobooks and speaker
    tags are never padded, a repeated phrase would corrupt the narration.
    Only used as a fallback after a silent result — text that already
    synthesizes is never altered."""
    core = text.rstrip('。！？!?；;,.…～~、 \t　"“”‘’')
    if len(core) >= 4 or not core:
        return text
    reps = -(-4 // len(core))  # ceil(4/len) — smallest k with k*len(core) ≥ 4
    return core * reps + text[len(core):]


# ── Silence-failover policy ─────────────────────────────────────
# A GPT-SoVITS HTTP 200 with inaudible audio is almost always TEXT-level (the
# text frontend can't map the phrase) or a transient XPU glitch (a "bad
# allocation" crashes that one inference to silence) — NOT a defect of the
# specific backend. Another pool member runs the identical model + text
# frontend, so failing over on silence would cold-load a second ~90 s model to
# reproduce the same silence, and both models would stay resident. We therefore
# NEVER switch backends because of silence: we retry the same instance a few
# times (a crashed inference usually clears), then drop the text (the padding
# fallback handles ultra-short input). Backends are only swapped after a HARD
# failure (timeout / unreachable / OOM), where the dead instance is stopped
# first so at most one model is ever resident.
#
# Safety net: an instance silent across several whole sentences is presumed
# wedged (corrupted XPU context) and is stopped, so the NEXT sentence reloads
# it fresh — one model, reloaded in place, never a second resident. The
# cooldown prevents retry storms from reloading repeatedly.
SILENT_RETRY_SAME = 2       # extra POSTs to the same instance after a silent 200
SILENT_RUN_LIMIT = 3        # silent sentences after which an instance is reloaded
SILENT_STOP_COOLDOWN = 300  # s — min gap between two silent-triggered reloads
_silent_run = {}            # port -> consecutive silent sentence count (under _silent_lock)
_silent_last_stop = {}      # port -> epoch of last silent-triggered reload


def _synthesize_batch_stream(texts, voice, speed, model_key, on_item,
                             deadline=None, engine=None):
    """一次请求提交整段句子，**收到一句就回调一句**（on_item(seq_index, wav)）。

    后端 /tts_batch_stream 是动态接活的流式接口：槽位一空就补下一句，每句
    生成完立刻发一行 {index, wav}。所以我们这里边收边落盘 —— 点读的「已就绪」
    和进度条都不必等整批。

    返回处理成功的句数；不可用时返回 None（调用方退回静态批/逐句）。
    """
    if not texts or len(texts) < 2:
        return None
    if engine:
        return None            # 引擎模式：退回静态批 /tts_batch（远端没有动态接活协议）
    key = _resolve_model_key(model_key)
    profile = MODEL_PROFILES[key]
    v = VOICES.get(voice, VOICES["default"])
    default_lang = v.get("text_lang", "zh")
    text_lang = _pick_lang(texts[0], default_lang)
    if any(_pick_lang(t, default_lang) != text_lang for t in texts):
        return None
    payload = {
        "texts": list(texts),
        "text": "",
        "text_lang": text_lang,
        "ref_audio_path": v.get("ref_audio_path", ""),
        "prompt_text": v.get("prompt_text", ""),
        "prompt_lang": v.get("prompt_lang", "zh"),
        "speed_factor": speed,
        "batch_size": BATCH_SIZE,       # 并发槽位数（不是句数）
        "dynamic_batch": True,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "seed": -1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "streaming_mode": False,
    }
    if profile.get("sample_steps") is not None:
        payload["sample_steps"] = int(profile["sample_steps"])

    port = _acquire_port(voice, key)
    tries = 0
    try:
        if not _ensure_backend_hot(port, key):
            return None
        r = requests.post(f"http://127.0.0.1:{port}/tts_batch_stream",
                          json=payload, stream=True, timeout=(15, 1800))
        if r.status_code == 404:
            logger.info("Backend %s has no /tts_batch_stream — fallback", port)
            return None
        if r.status_code != 200:
            logger.warning("Stream batch rejected (%s): %s — fallback",
                           r.status_code, r.text[:160])
            return None
        got = 0
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            item = json.loads(line)
            if "error" in item:
                logger.warning("Stream batch error after %d item(s): %s", got, item["error"])
                return got or None
            idx = int(item["index"])
            if not (0 <= idx < len(texts)):
                continue
            wav = base64.b64decode(item["wav"])
            if _is_silent_wav(wav):
                # 静音：不落盘，留给调用方逐句重试这一句
                logger.warning("Stream batch: sentence %d silent — will retry singly", idx + 1)
                tries += 1
                continue
            on_item(idx, wav)
            got += 1
        return got
    except Exception as e:
        logger.warning("Stream batch failed (%s) — fallback", e)
        return None
    finally:
        _release_port(port)


def _synthesize_batch(texts, voice, speed, model_key=None, engine=None):
    """一次请求合成多句（同一种 text_lang），逐句返回 WAV 字节；不可用时 None。

    长文的瓶颈是逐句串行的自回归解码：后端是单进程、请求串行的，一句一请求
    时 batch_size 再大也只会用到「这一句被切成几段」。把相邻几句拼成一次请求，
    模型就能在同一批前向里一起解码（实测 4 句 ≈ 逐句耗时的 1/3），而 /tts_batch
    会把每句的音频单独交回来，点读的时间点一分不差。

    返回逐句 WAV 列表（个别句子静音的位置是 None，调用方只补那几句）；
    整批不可用时返回 None，调用方全部退回逐句。异常、老后端没有 /tts_batch
    都走这条退路 —— 批量只可能更快，不会更脆。"""
    if not texts or len(texts) < 2:
        return None
    if engine:
        return _engine_synth_batch(engine, texts, voice, speed)
    key = _resolve_model_key(model_key)
    profile = MODEL_PROFILES[key]
    v = VOICES.get(voice, VOICES["default"])
    default_lang = v.get("text_lang", "zh")
    text_lang = _pick_lang(texts[0], default_lang)
    if any(_pick_lang(t, default_lang) != text_lang for t in texts):
        return None                      # 混语言：各句走自己的前端，别硬凑
    payload = {
        "texts": list(texts),
        "text": "",                      # /tts_batch 走 texts 分支，text 仅占位
        "text_lang": text_lang,
        "ref_audio_path": v.get("ref_audio_path", ""),
        "prompt_text": v.get("prompt_text", ""),
        "prompt_lang": v.get("prompt_lang", "zh"),
        "speed_factor": speed,
        # 动态接活时 batch_size 是「并发槽位」数，句子数可以远多于它
        "batch_size": BATCH_SIZE if DYNAMIC_BATCH else len(texts),
        "dynamic_batch": DYNAMIC_BATCH,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "seed": -1,
        "parallel_infer": True,          # 批量本身就是并行，这里始终开
        "repetition_penalty": 1.35,
        "streaming_mode": False,
    }
    if profile.get("sample_steps") is not None:
        payload["sample_steps"] = int(profile["sample_steps"])

    port = _acquire_port(voice, key)
    try:
        if not _ensure_backend_hot(port, key):
            return None
        r = requests.post(f"http://127.0.0.1:{port}/tts_batch",
                          json=payload, timeout=600)
        if r.status_code == 404:
            logger.info("Backend %s has no /tts_batch — per-sentence fallback", port)
            return None
        if r.status_code != 200:
            logger.warning("Batch synth rejected (%s): %s — per-sentence fallback",
                           r.status_code, r.text[:160])
            return None
        wavs = [base64.b64decode(w) for w in r.json().get("wavs", [])]
        if len(wavs) != len(texts):
            logger.warning("Batch synth returned %d wavs for %d texts — fallback",
                           len(wavs), len(texts))
            return None
        # 采样是随机的，一批 8 句里偶尔会有 1-2 句抽到 immediate-EOS（静音）。
        # 不能因此整批重跑（那是 8 倍的浪费）：把静音的位置标成 None，调用方
        # 只补那几句。整批失败才返回 None 让全部退回逐句。
        out = []
        silent = 0
        for w in wavs:
            if _is_silent_wav(w):
                out.append(None)
                silent += 1
            else:
                out.append(w)
        if silent:
            logger.warning("Batch synth: %d/%d item(s) silent — retrying those singly",
                           silent, len(wavs))
        return out
    except Exception as e:
        logger.warning("Batch synth failed (%s) — per-sentence fallback", e)
        return None
    finally:
        _release_port(port)


def _synthesize(text, voice, speed, pad_short=True, model_key=None, engine=None):
    """Synthesize one text chunk via backend, with OOM/timeout failover.
    Tries voice-affinity backend first; on hard failure (OOM/unreachable/
    timeout) that instance is stopped and the next healthy one is loaded.
    A silent 200 is retried on the SAME instance and never triggers a
    cross-backend swap (see SILENT_RETRY_SAME above).

    Ultra-short texts (e.g. "咔嗒。") make GPT-SoVITS T2S decode immediate-EOS
    → near-silence. As a last resort for the single-shot danmaku path, a
    short silent text is PADDED (spoken repeated) and retried.

    pad_short=False (long-text audiobooks, speaker tags) must never alter
    what is spoken: a duplicated phrase mid-story breaks the narration, and
    a padded speaker clip would poison the speaker cache permanently. Those
    paths handle a stubbornly-silent sentence by failing/skipping it instead,
    and keep the full same-instance retries (a re-sample is the same text —
    no continuity cost)."""
    # 引擎模式：这一句交给远端池合成（本机只负责文本清洗、时间轴与落盘）
    if engine:
        return _engine_synth(engine, text, voice, speed)

    key = _resolve_model_key(model_key)
    profile = MODEL_PROFILES[key]

    def _attempt(t):
        v = VOICES.get(voice, VOICES["default"])
        if pad_short:
            # Immediate-EOS on this model is length-dependent (silent_dumps
            # forensics): first segments of ~3 chars NEVER survived sampling
            # (9/9 silent), 6-char texts recover within a couple of tries, ≥8
            # chars nearly always succeed first try. Don't burn same-instance
            # retries on hopeless lengths — escalate to the padded variant
            # sooner. Danmaku only: latency matters more than a rare repeat.
            core_len = _core_speak_len(t)
            same_retries = (SILENT_RETRY_SAME if core_len >= 8
                            else 1 if core_len >= 4 else 0)
        else:
            same_retries = SILENT_RETRY_SAME
        # Per-sentence language: an English/Japanese line inside a Chinese
        # novel must not be forced through the zh frontend (→ near-silence).
        text_lang = _pick_lang(t, v.get("text_lang", "zh"))
        if text_lang == "en":
            # The en frontend chokes on full-width CJK punctuation — swap to
            # ASCII so “My pleasure，sweetie。” is read as real English.
            t = t.translate(str.maketrans({
                "，": ",", "。": ".", "、": ",", "；": ";",
                "！": "!", "？": "?", "：": ":", "…": "...",
            }))
        payload = {
            "text": t,
            "text_lang": text_lang,
            "ref_audio_path": v.get("ref_audio_path", ""),
            "prompt_text": v.get("prompt_text", ""),
            "prompt_lang": v.get("prompt_lang", "zh"),
            "speed_factor": speed,
            "text_split_method": "cut2",  # 凑50字一段 — 避免 cut5 把首尾短片段吞掉
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "seed": -1,
            "batch_size": BATCH_SIZE,
            # 实例内并行由启动配置 parallel_infer 控制；静音时下方自动回退串行。
            "parallel_infer": PARALLEL_INFER,
            "repetition_penalty": 1.35,
            "streaming_mode": False,
        }
        if profile.get("sample_steps") is not None:
            payload["sample_steps"] = int(profile["sample_steps"])

        # Take one backend lease. Held for the whole synthesis (including
        # same-instance retries) and released in `finally`, so at most
        # len(ALL_BACKEND_PORTS)=4 requests run concurrently and no port is
        # POSTed twice at once. All ports are the same v5Turbo model.
        port = _acquire_port(voice, key)
        try:
            if not _ensure_backend_hot(port, key):
                logger.warning("Backend %s (%s) could not be loaded — skip", port, key)
                return None
            backend_url = f"http://127.0.0.1:{port}"
            hard_fail = False
            serial_fallback = False
            for try_i in range(1 + same_retries):
                attempt_payload = payload
                if serial_fallback:
                    # Parallel batch inferred silent once — retry this same
                    # instance in plain serial mode before giving up.
                    attempt_payload = dict(payload, parallel_infer=False, batch_size=1)
                try:
                    r = requests.post(f"{backend_url}/tts", json=attempt_payload, timeout=300)
                except requests.exceptions.Timeout as e:
                    # Backend hung — stop it so the next request reloads fresh.
                    logger.warning("Backend %s timed out (%s) — stopping", port, e)
                    _stop_backend(port)
                    hard_fail = True
                    break
                except requests.exceptions.ConnectionError:
                    logger.warning("Backend %s unreachable — clearing state", port)
                    _stop_backend(port)
                    hard_fail = True
                    break
                if r.status_code == 200:
                    if not _is_silent_wav(r.content):
                        ok_q, why_q = _audio_quality(r.content, t)
                        if not ok_q:
                            # 空转出来的残缺音频：不算成功，换一次采样再来
                            logger.warning("Backend %s BAD AUDIO for %r: %s — retry",
                                           port, t[:20], why_q)
                            _dump_rejected_wav(t, r.content)
                            time.sleep(1)
                            continue
                        with _silent_lock:
                            _silent_run[port] = 0
                        return r.content
                    # Silent 200. If we were in parallel mode, fall back to
                    # serial on the SAME instance for the remaining retries.
                    stats = _describe_wav(r.content)
                    dump = _dump_rejected_wav(t, r.content)
                    if attempt_payload.get("parallel_infer") and not serial_fallback:
                        logger.warning("Backend %s SILENT in parallel mode for %r: %s — retrying SERIAL (dump %s)",
                                       port, t[:20], stats, dump)
                        serial_fallback = True
                        continue
                    logger.warning("Backend %s SILENT for %r (try %d/%d): %s — kept silent_dumps/%s — retry same instance",
                                   port, t[:20], try_i + 1, 1 + same_retries,
                                   stats, dump)
                    time.sleep(1)   # let a crashed inference unwind
                    continue
                err = r.text
                if "OUT_OF_DEVICE_MEMORY" in err or "OUT_OF_RESOURCES" in err:
                    logger.warning("Backend %s GPU OOM — stopping instance", port)
                    _stop_backend(port)
                    hard_fail = True
                    break
                raise HTTPException(502, f"Backend error {r.status_code}: {err[:200]}")

            if hard_fail:
                return None
            # Silent even after same-instance retries (parallel + serial).
            with _silent_lock:
                _silent_run[port] = _silent_run.get(port, 0) + 1
                now = time.time()
                do_reload = (_silent_run[port] >= SILENT_RUN_LIMIT
                             and now - _silent_last_stop.get(port, 0.0) > SILENT_STOP_COOLDOWN
                             and _backend_procs.get(port) is not None)
                if do_reload:
                    logger.warning("Backend %s silent on %d sentences — reloading fresh",
                                   port, _silent_run[port])
                    _silent_run[port] = 0
                    _silent_last_stop[port] = now
            if do_reload:
                _stop_backend(port)   # next request cold-loads it again
            return None
        finally:
            _release_port(port)

    result = _attempt(text)
    if result is None and pad_short:
        padded = _pad_short_text(text)
        if padded != text:
            logger.warning("All backends silent for short text %r — retrying padded as %r",
                           text[:20], padded[:20])
            result = _attempt(padded)
    if result is None:
        raise HTTPException(503, "All backends unavailable. Please retry.")
    return result


@app.post("/api/tts")
def tts(req: TTSRequest, request: Request):
    # NOTE: sync def (not async) — FastAPI runs sync endpoints in a thread pool,
    # so blocking requests.post here does NOT block other concurrent requests.
    # async def would serialize all requests on the event loop.
    voice = req.voice
    key = _resolve_model_key(req.version)
    engine = _engine_config(_server_key(request))   # 选了「引擎型」服务器就走远端算力
    t0 = time.time()
    clean_text = _clean_text(req.text)  # drop unsupported chars (emojis etc)
    if not clean_text:
        raise HTTPException(400, "文本清洗后为空（仅含 emoji/特殊符号）")

    try:
        # Speaker audio (cached) — "名字说" is highly repeatable
        speaker_wav = None
        if req.speaker:
            speaker_wav = _get_cached_speaker(req.speaker, voice, req.speed, key)
            if speaker_wav is None:
                speaker_text = f"{req.speaker}说"
                speaker_wav = _synthesize(speaker_text, voice, req.speed, pad_short=False,
                                          model_key=key, engine=engine)
                _cache_speaker(req.speaker, voice, req.speed, speaker_wav, key)

        # Text audio (always generated, with failover handled inside)
        text_wav = _synthesize(clean_text, voice, req.speed, model_key=key, engine=engine)
        text_wav = _trim_silence(text_wav)  # cut vocoder head/tail padding

        # Merge: speaker + text
        if speaker_wav:
            speaker_wav = _trim_silence(speaker_wav)
            merged, sr = _concat_wav(speaker_wav, text_wav)
            elapsed = time.time() - t0
            logger.info(f"TTS: voice={voice} version={key} speaker={req.speaker} len={len(req.text)} {elapsed:.1f}s")
            return StreamingResponse(merged, media_type="audio/wav",
                headers={"X-TTS-Time": f"{elapsed:.1f}", "X-TTS-Version": key})
        else:
            elapsed = time.time() - t0
            size_kb = len(text_wav) / 1024
            logger.info(f"TTS: voice={voice} version={key} len={len(req.text)} {elapsed:.1f}s {size_kb:.0f}KB")
            return StreamingResponse(io.BytesIO(text_wav), media_type="audio/wav",
                headers={"X-TTS-Time": f"{elapsed:.1f}", "X-TTS-Version": key})
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "No v5Turbo backend reachable.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, str(e))


class LongTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)  # 100k chars ≈ full novel
    voice: str = "default"
    version: str = ""  # v4 / v5turbo; empty = server default
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.post("/api/tts/long")
def tts_long(req: LongTTSRequest, request: Request):
    """Start async long-text synthesis. Returns the task id — the credential."""
    key = _resolve_model_key(req.version)
    engine = _engine_config(_server_key(request))
    task_id = _new_task(req.text, req.voice, req.speed, key)
    if engine:
        # 任务本体仍在本机（分句/进度/点读/断点续传都不变），只有合成发给远端
        with _tts_tasks_lock:
            _tts_tasks[task_id]["engine"] = engine["name"]
        logger.info("Long-text task %s uses engine %s (%s)",
                    task_id, engine["name"], engine["url"])
    threading.Thread(target=_run_long_task,
                     args=(task_id, req.text, req.voice, req.speed, key),
                     kwargs={"engine": engine},
                     daemon=True).start()
    return {"task_id": task_id, "version": key,
            "engine": engine["name"] if engine else ""}


@app.get("/api/tts/progress/{task_id}")
def tts_progress(task_id: str):
    t = _get_task(task_id)
    # Also report backend engine state so a reader UI can show "正在加载模型…"
    # while a lazily-started backend cold-loads during this task.
    return {"status": t["status"], "done": t["done"], "total": t["total"],
            "error": t["error"], "pool": _backend_snapshot()}


def _range_response(request, wav):
    """Serve a WAV with HTTP Range support.

    点读 seeks to a sentence's offset inside this file, and a plain
    StreamingResponse tells the browser the resource is NOT seekable — it would
    have to buffer the whole chapter first. Answering 206 with the requested
    slice makes every sentence click instant (and lets the <audio> scrubber work
    on a multi-hour file). Used for the finished chapter AND for the prefix a
    running task serves, so 点读 can seek in both."""
    import re as _re
    total = len(wav)
    rng = request.headers.get("range")
    if rng:
        m = _re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if m:
            a, b = m.group(1), m.group(2)
            if a == "" and b:                     # suffix range: last N bytes
                start, end = max(0, total - int(b)), total - 1
            else:
                start = int(a or 0)
                end = min(int(b), total - 1) if b else total - 1
            if start > end or start >= total:
                return Response(status_code=416,
                                headers={"Content-Range": f"bytes */{total}"})
            chunk = wav[start:end + 1]
            return Response(content=chunk, status_code=206, media_type="audio/wav",
                            headers={"Content-Range": f"bytes {start}-{end}/{total}",
                                     "Accept-Ranges": "bytes",
                                     "Content-Length": str(len(chunk))})
    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav",
                             headers={"Accept-Ranges": "bytes"})


@app.get("/api/tts/result/{task_id}")
def tts_result(request: Request, task_id: str):
    t = _get_task(task_id)
    if t["status"] == "running":
        raise HTTPException(409, "Still synthesizing")
    if t["status"] == "error":
        raise HTTPException(500, t["error"] or "Synthesis failed")
    if not t["wav"]:
        raise HTTPException(500, "No audio produced")
    return _range_response(request, t["wav"])


# ── 点读：在整章音频上按句跳转 + 存档 / 再上传 ────────────────────
# 点读 does NOT keep per-sentence clips. The chapter WAV is the plain
# concatenation of the sentence clips (nothing is trimmed out of it), so each
# sentence's start is just a running sum of the recorded durations. Its end is
# the next sentence's start, and the last sentence simply plays out the file —
# so only `starts` is stored. Clicking a sentence is a seek on the one audio
# file: no extra disk, no extra requests, no end array, no rebasing.
#
# The two artifacts a reader needs are exactly the two files this server keeps
# per task — {task_id}.wav and {task_id}.json — so downloading and re-uploading
# them is a faithful archive: no bespoke container format to keep in sync.

def _get_task(task_id):
    """Look a chapter up by its id — the id IS the credential now.

    Nothing anywhere lists tasks (no list endpoint, no static mount over
    task_cache/), so a task is reachable only by knowing its random id. The
    retired access key is gone: an old ?key=... in a bookmark or a stale `key`
    field inside a re-imported archive is accepted and ignored, never checked.
    """
    with _tts_tasks_lock:
        t = _tts_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found or expired")
    return t


def _split_doc(task_id, t):
    """分割文件 — the task's own stored metadata verbatim (everything
    _save_task keeps, except the audio blob, which is the other file), plus the
    sentence count and audio length for the reader UI. The access key stays in:
    it is the credential that finds the chapter again after a lost connection,
    and the file is only reachable by someone who already holds it. Re-uploading
    this file is the supported way to restore a task after this cache is wiped.

    The per-sentence times are `starts` / `ends` (seconds into the stored WAV),
    so a consumer can slice sentence k out of the audio without knowing anything
    about how it was synthesized."""
    sents = t.get("sents") or []
    # `key` is excluded as well: tasks restored from an older cache still
    # carry the retired field, and it must never reach a downloaded file.
    doc = {k: v for k, v in t.items()
           if k not in ("wav", "pool", "sents", "key")
           and not k.startswith("_")}
    doc.update({
        "task_id": task_id,          # the stem of both stored files
        "total": len(sents),
        "sents": sents,
        "audio_seconds": round(_wav_seconds(t["wav"]), 3) if t.get("wav") else None,
    })
    return doc


@app.get("/api/tts/split/{task_id}")
def tts_split(task_id: str):
    """下载切割文件 = 服务端存的 {task_id}.json（句子 / 时长 / 偏移）。"""
    return JSONResponse(_split_doc(task_id, _get_task(task_id)),
                        headers={"Content-Disposition":
                                 f'attachment; filename="{task_id}.json"'})


# ── 合成中提前点读：已完成句子的前缀音频 ─────────────────────────
# A task that is still synthesizing can already be read: the fragments on disk
# (one file per finished sentence) concatenate to a PREFIX of the final chapter
# audio — the chapter is the plain concatenation of those very clips and
# _concat_wav is the same PCM_16 path in both cases, so the samples are
# identical and every running-sum start stays valid once the rest lands. The
# reader serves this prefix now and swaps to /api/tts/result when the task
# completes, without changing a single offset.
#
# The FINAL split document is deliberately left alone: it must only ever
# describe the finished chapter, or a downloaded archive would carry half-built
# times. The in-progress view is this separate endpoint instead.


def _partial_state(task_id, t):
    """(wav_bytes|None, durations, ready) — the playable prefix of a running
    task, built exactly the way the worker will merge the chapter.

    `ready` is task["done"] clamped to the sentence list: it counts finalized
    sentences, skipped ones included. A skipped sentence has no fragment and a
    0.0 duration, so it collapses to a zero-length slot and shifts nothing."""
    with _tts_tasks_lock:
        total = len(t.get("sents") or [])
        done = int(t.get("done") or 0)
    ready = min(done, total) if total else done
    chunks, durations = [], []
    for i in range(ready):
        f = TASK_CACHE_DIR / f"{task_id}.{i}.wav"
        if not f.exists():
            durations.append(0.0)      # skipped for silence → contributed no audio
            continue
        raw = f.read_bytes()           # only i < done is ever read: the worker
        chunks.append(raw)             # sets done AFTER the file is written
        durations.append(_wav_seconds(raw))
    if not chunks:
        return None, durations, ready
    merged, _ = _concat_wav(*chunks)
    return (merged.getvalue() if merged else None), durations, ready


def _partial_audio(task_id, t):
    """The prefix WAV for a running task, with a one-slot cache keyed by the
    done count — rebuilding the whole prefix on every poll would be O(chapter)
    work per request. Rebuilt only when a new sentence lands."""
    with _tts_tasks_lock:
        done = int(t.get("done") or 0)
        cached = t.get("_partial_cache")
    if cached and cached[0] == done:
        return cached[1], cached[2], cached[3]
    wav, durations, ready = _partial_state(task_id, t)
    if wav:
        with _tts_tasks_lock:
            t["_partial_cache"] = (done, wav, durations, ready)
    return wav, durations, ready


def _partial_doc(task_id, t):
    """Reader document for a task that is still synthesizing.

    Same shape as _split_doc plus `partial` / `ready`: `sents` is the complete
    split, `starts` runs the full sentence count with the not-yet-synthesized
    tail pinned to the end of the playable audio, and `ready` says how many
    sentences can be played right now (the reader strikes through the rest).
    A finished task returns the final document with partial=false."""
    if t.get("status") == "done" or (t.get("status") == "error"
                                     and not int(t.get("done") or 0)):
        doc = _split_doc(task_id, t)
        doc["partial"] = False
        doc["ready"] = len(doc.get("sents") or [])
        return doc
    wav, durations, ready = _partial_audio(task_id, t)
    doc = _split_doc(task_id, t)
    sents = doc.get("sents") or []
    starts = _sentence_starts(durations)
    # The padding value is the END of the ready region (= the length of the
    # prefix audio), so sentence k-1's end and every pending sentence's start
    # land exactly on the playable edge — not on the last ready start.
    tail = (starts[-1] + durations[-1]) if starts else 0.0
    starts = starts + [tail] * (len(sents) - len(starts))
    durations = durations + [0.0] * (len(sents) - len(durations))
    doc.update({
        "partial": True,
        "ready": ready,
        "done": ready,
        "starts": [round(s, 3) for s in starts],
        "durations": [round(d, 3) for d in durations],
        "audio_seconds": round(_wav_seconds(wav), 3) if wav else 0.0,
    })
    return doc


@app.get("/api/tts/partial/{task_id}")
def tts_partial(task_id: str):
    """点读用的「当前进度」文档：合成中给已就绪句子的前缀音频与时间点，
    完成后与 /api/tts/split 返回同一份最终文档（partial=false）。"""
    return JSONResponse(_partial_doc(task_id, _get_task(task_id)))


@app.get("/api/tts/partial/{task_id}/audio")
def tts_partial_audio(request: Request, task_id: str):
    """已完成部分的前缀音频（支持 Range，可 seek），与最终整章音频逐样本一致。"""
    t = _get_task(task_id)
    if t.get("status") == "done" and t.get("wav"):
        return _range_response(request, t["wav"])
    wav, _, ready = _partial_audio(task_id, t)
    if not wav:
        raise HTTPException(409, f"第 1 句还没合成完（{ready}/{t.get('total', 0)}）")
    return _range_response(request, wav)


@app.post("/api/tts/import")
async def tts_import(split: UploadFile = File(...),
                     audio: UploadFile = File(None)):
    """上传「整章音频 + 分割文件」= 服务端原本存的那两个文件 → 新任务。

    multipart/form-data, field names `split` (.json) and `audio` (.wav, optional):
      * both files     → the chapter is readable again immediately, seeking by
        the recorded offsets; nothing is re-synthesized;
      * split only     → the same sentence split is re-synthesized, so the
        offsets stay valid and 点读 works even without the original audio.
    """
    raw = await split.read()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "分割文件不是合法的 JSON")
    if not isinstance(doc, dict):
        raise HTTPException(400, "分割文件必须是一个 JSON 对象")

    sents = [s for s in (doc.get("sents") or []) if isinstance(s, str)]
    if not sents:
        raise HTTPException(400, "分割文件里没有句子（sents 为空）")
    voice = doc.get("voice") if doc.get("voice") in VOICES else "default"
    version = _resolve_model_key(doc.get("version"))
    try:
        speed = min(2.0, max(0.5, float(doc.get("speed") or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0

    full_wav = await audio.read() if audio is not None else b""
    if full_wav and full_wav[:4] != b"RIFF":
        raise HTTPException(400, "音频不是 WAV 文件（缺 RIFF 头）")

    # Per-sentence start times. That is all 点读 needs to seek: a sentence runs
    # until the next one starts, and the last plays out the file. `offsets` is
    # the pre-rename name of `starts`, still accepted so a file downloaded from
    # an older build re-imports.
    starts = doc.get("starts") or doc.get("offsets") or []
    durations_in = doc.get("durations") or []

    text = doc.get("text") or "".join(sents)
    task_id = _new_task(text, voice, speed, version)
    warning = None
    with _tts_tasks_lock:
        task = _tts_tasks[task_id]
        task["sents"] = list(sents)
        task["total"] = len(sents)
        if full_wav:
            seconds = _wav_seconds(full_wav)
            aligned = len(starts) == len(sents)
            if not aligned:
                warning = (f"分割文件缺少每句时间点（starts {len(starts)} 项 / "
                           f"{len(sents)} 句），无法定位，改为重新合成")
            else:
                # The times only describe THIS audio. If the two files came from
                # different runs, seeking would land in the wrong sentence, so
                # say so instead of silently serving a misaligned reader.
                claim = doc.get("audio_seconds")
                if claim and abs(float(claim) - seconds) > 1.0:
                    warning = (f"分割文件记录的音频长度 {float(claim):.1f}s 与上传的 "
                               f"{seconds:.1f}s 不符，点读偏移可能不准")
                elif float(starts[-1]) >= seconds:
                    warning = (f"分割文件的时间点（最末 {float(starts[-1]):.1f}s）超出音频"
                               f"长度 {seconds:.1f}s，两者可能不是同一次合成")
            if aligned and not warning:
                task["durations"] = [round(float(d), 3) for d in durations_in] \
                    or [0.0] * len(sents)
                task["starts"] = [round(float(s), 3) for s in starts]
                task["audio_seconds"] = round(seconds, 3)
                task["wav"] = full_wav
                task["status"] = "done"
                task["done"] = len(sents)   # imported whole, nothing to build
                _save_task(task_id, dict(task))     # write the audio file out
                logger.info("Imported %s: %d sentences, audio %.1fs, 点读 ready "
                            "(starts %.2fs…%.2fs)", task_id, len(sents), seconds,
                            float(task["starts"][0]), float(task["starts"][-1]))
                return {"task_id": task_id, "total": len(sents),
                        "reused_audio": True, "resynthesized": False}
    # No audio, or metadata that cannot be trusted against the audio → rebuild
    # from the split, which reproduces the same boundaries with fresh times.
    logger.info("Imported split only → %s: re-synthesizing %d sentences%s",
                task_id, len(sents), f" ({warning})" if warning else "")
    with _tts_tasks_lock:
        _tts_tasks[task_id]["durations"] = []
        _tts_tasks[task_id]["starts"] = []
        _tts_tasks[task_id]["wav"] = None
        _tts_tasks[task_id]["status"] = "queued"
        _tts_tasks[task_id]["done"] = 0
    threading.Thread(target=_run_long_task, args=(task_id, text, voice, speed, version),
                     daemon=True).start()
    return {"task_id": task_id, "version": version, "total": len(sents),
            "reused_audio": False, "resynthesized": True, "warning": warning}


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, threading
    sc = config.get("server", {})
    host = sc.get("host", "0.0.0.0")
    listeners = sc.get("listeners", [{"port": sc.get("port", 8000)}])

    # Daemons: task cleanup + backend keep-alive
    threading.Thread(target=_cleanup_tasks, daemon=True).start()
    threading.Thread(target=_monitor_backends, daemon=True).start()

    # Backends load lazily on first use (each ~2.2 GiB). With one v5Turbo
    # profile and 4 ports, up to 4 instances stay resident and serve 4
    # concurrent requests; cold loads are serialized to cap memory spikes.
    mode = "in-process batch parallel" if PARALLEL_INFER else "multi-instance concurrent"
    logger.info("v5Turbo-only pool (%s): %s backend(s) on %s, batch_size=%s, lazy load, no idle eviction.",
                mode, len(ALL_BACKEND_PORTS), ALL_BACKEND_PORTS, BATCH_SIZE)

    # Resume long-text tasks from disk. Each resume thread starts its own
    # backend on demand (see _run_long_task), so no warm-up is needed here.
    _load_tasks_from_disk()

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