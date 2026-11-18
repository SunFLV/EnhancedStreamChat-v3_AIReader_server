"""Probe /tts_batch with dynamic_batch (slot refill) against a standalone backend.

    python test_dynamic_batch.py [port]

Prints wall time, how many WAVs came back, and whether each WAV's duration
still tracks its text length (i.e. the audio really belongs to that sentence).
"""
import base64
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9890
URL = f"http://127.0.0.1:{PORT}/tts_batch"
STREAM_URL = f"http://127.0.0.1:{PORT}/tts_batch_stream"

# 故意混排：极短句 + 长句，动态接活的收益就在这里
SENTS = [
    "雨停了。",
    "她在书架最深处翻到一本已经发霉的旧书，封皮上没有任何字，只有一层薄薄的灰，翻开时纸页发出一种很轻的响声。",
    "窗外的雨还在下，玻璃上滑过一条条水痕。",
    "门开了。",
    "她忽然想起很多年前的一个夏天，也是这样的雨，也是这样的夜。",
    "怎么了？",
    "走廊尽头的窗户透进一点灰蓝色的光，照在地板上。",
    "快走！",
    "那本书里夹着一张旧车票，日期已经模糊了，只看得清一个地名。",
    "她笑了。",
    "她把车票放回书里，像把某个夏天重新合上。",
    "没有人回答。",
    "灯灭的时候，整栋楼只剩下雨水的味道。",
    "倒计时：3、2、1。",
    "第三排的架子上标着 A+B=12 的编号，温度计显示 21℃，她抬头看了很久。",
    "然后，一切安静下来。",
]

VOICE = {
    "ref_audio_path": "D:/BSAIReader/server/voices/default_ref.mp3",
    "prompt_text": "そよちゃん、部活でコントラバス、だっけ",
    "prompt_lang": "ja",
}
BASE = {
    "text_lang": "zh", **VOICE, "speed_factor": 1.0, "top_k": 5, "top_p": 1.0,
    "temperature": 1.0, "seed": -1, "parallel_infer": True,
    "repetition_penalty": 1.35, "streaming_mode": False, "sample_steps": 4,
    "text_split_method": "cut0",
}


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return time.perf_counter() - t0, body


def wav_seconds(raw):
    import io
    import wave
    with wave.open(io.BytesIO(raw)) as w:
        return w.getnframes() / float(w.getframerate())


def stream(url, payload, timeout=1800):
    """读 NDJSON：每收到一行就是一句合成好了 —— 打印到达时刻。"""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    import base64 as _b64
    got = {}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            if "error" in d:
                print("ERROR:", d["error"])
                break
            got[int(d["index"])] = _b64.b64decode(d["wav"])
            print(f"  +{time.perf_counter()-t0:6.1f}s  第 {d['index']+1} 句到达")
    return time.perf_counter() - t0, got


def main():
    mode = sys.argv[2] if len(sys.argv) > 2 else "dynamic"
    if mode == "stream":
        t, got = stream(STREAM_URL, dict(BASE, texts=SENTS, batch_size=8))
        durs = [round(wav_seconds(got[i]), 2) if i in got else None for i in range(len(SENTS))]
        print(f"[stream] {t:.1f}s  {len(got)}/{len(SENTS)} 句  {t/max(1,len(got)):.2f}s per sentence")
        print("chars:", [len(s) for s in SENTS])
        print("durs :", durs)
        return
    payload = dict(BASE, texts=SENTS, batch_size=8,
                   dynamic_batch=(mode == "dynamic"))
    t, body = post(URL, payload)
    data = json.loads(body)
    wavs = [base64.b64decode(w) for w in data["wavs"]]
    durs = [round(wav_seconds(w), 2) for w in wavs]
    print(f"[{mode}] {t:.1f}s  wavs={len(wavs)}/{len(SENTS)}  "
          f"{t/len(wavs):.2f}s per sentence")
    print("chars:", [len(s) for s in SENTS])
    print("durs :", durs)
    xs = [len(s) for s in SENTS[:len(durs)]]
    mx, md = sum(xs) / len(xs), sum(durs) / len(durs)
    num = sum((a - mx) * (b - md) for a, b in zip(xs, durs))
    den = (sum((a - mx) ** 2 for a in xs) ** 0.5) * (sum((b - md) ** 2 for b in durs) ** 0.5)
    print(f"correlation(chars, duration) = {num/den:.3f}")
    if len(wavs) != len(SENTS):
        print("MISMATCH — 句数与音频数不一致")


if __name__ == "__main__":
    main()
