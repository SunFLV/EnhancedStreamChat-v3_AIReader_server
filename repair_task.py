"""Re-synthesize specific sentences of a task and overwrite their fragments.

Uses the server's own `_synthesize()` (so the new audio-quality check applies),
splits over-long sentences at commas before synthesis, then concatenates the
pieces back into ONE fragment — the sentence count / 点读 indices stay untouched.

    python repair_task.py <task_id> 168 104 91
"""
import io
import json
import sys
import re
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import server_intel as S


def split_long(text, limit=45):
    """按逗号把长句切成 <=limit 字的段，尽量不切断词。"""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[，、；：,])", text):
        if cur and len(cur) + len(piece) > limit:
            parts.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        parts.append(cur)
    return parts


def main():
    tid = sys.argv[1]
    idxs = [int(x) - 1 for x in sys.argv[2:]]
    d = json.load(open(f"task_cache/{tid}.json", encoding="utf-8"))
    sents = d["sents"]
    voice = d.get("voice", "default")
    speed = float(d.get("speed", 1.0))
    key = d.get("version", "v5turbo")

    for i in idxs:
        raw = sents[i]
        text = S._clean_text(raw)
        parts = split_long(text)
        print(f"#{i+1}: {len(text)} 字 -> {len(parts)} 段")
        for p in parts:
            print(f"    · {p[:36]}")
        wavs = []
        for p in parts:
            piece = None
            for attempt in range(6):
                piece = S._synthesize(p, voice, speed, pad_short=False, model_key=key)
                if piece:
                    ok, why = S._audio_quality(piece, p)
                    if ok:
                        break
                    print(f"    质检不过（{why}），重采 {attempt + 1}/6")
                    piece = None
            if not piece:
                print(f"    !! 段合成失败：{p[:20]} —— 保留原片段")
                wavs = []
                break
            wavs.append(piece)
        if not wavs:
            continue
        merged, sr = S._concat_wav(*wavs)
        data = merged.getvalue()
        f = S.TASK_CACHE_DIR / f"{tid}.{i}.wav"
        f.write_bytes(data)
        ok, why = S._audio_quality(data, text)
        print(f"  -> 写入 {f.name}: {S._wav_seconds(data):.1f}s / {len(text)} 字 "
              f"({S._wav_seconds(data)/max(1,len(text)):.2f}s/字) 质检={'OK' if ok else why}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
