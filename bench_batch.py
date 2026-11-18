"""/tts_batch vs one-request-per-sentence, against a standalone backend.

    python bench_batch.py [port]

Reports wall time for 4 sentences done serially vs in one batched request, and
checks the batch response really carries one WAV per sentence (that is what
keeps 点读 offsets exact).
"""
import base64
import json
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9890
SERIAL_URL = f"http://127.0.0.1:{PORT}/tts"
BATCH_URL = f"http://127.0.0.1:{PORT}/tts_batch"

SENTS = [
    "深夜的图书馆只剩下一盏灯还亮着。",
    "她翻过一页，纸的声音在空荡的大厅里显得格外清楚。",
    "窗外的雨还在下，玻璃上滑过一条条水痕。",
    "她忽然想起很多年前的一个夏天，也是这样的雨。",
]

VOICE = {
    "ref_audio_path": "D:/BSAIReader/server/voices/default_ref.mp3",
    "prompt_text": "そよちゃん、部活でコントラバス、だっけ",
    "prompt_lang": "ja",
}

BASE = {
    "text_lang": "zh",
    **VOICE,
    "speed_factor": 1.0,
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "seed": -1,
    "parallel_infer": True,
    "repetition_penalty": 1.35,
    "streaming_mode": False,
    "sample_steps": 4,
}


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return time.perf_counter() - t0, body


def main():
    t, body = post(SERIAL_URL, dict(BASE, text=SENTS[0], batch_size=4,
                                    text_split_method="cut2"))
    print(f"warmup {t:.2f}s ({len(body)/1024:.0f} KiB)")

    serial = []
    for s in SENTS:
        t, body = post(SERIAL_URL, dict(BASE, text=s, batch_size=4,
                                        text_split_method="cut2"))
        serial.append(t)
    print(f"SERIAL  4 sentences: {sum(serial):.2f}s "
          f"(mean {statistics.mean(serial):.2f}s each)")

    for batch in (2, 4):
        t, body = post(BATCH_URL, dict(BASE, texts=SENTS[:batch],
                                       batch_size=batch, text_split_method="cut0"))
        data = json.loads(body)
        sizes = [len(base64.b64decode(w)) / 1024 for w in data["wavs"]]
        per = t / batch
        print(f"BATCH   {batch} sentences: {t:.2f}s ({per:.2f}s each, "
              f"x{sum(serial[:batch])/t:.2f} vs serial) "
              f"wavs={len(sizes)} size={[round(s) for s in sizes]} KiB")


if __name__ == "__main__":
    main()
