"""Compare long-text timing with and without the batch length constraint.

Same text, same backend, two server instances (or one restarted with a
different `parallel_batch_len_ratio`). Prints wall time, how the sentences
were grouped and whether every fragment came back in the right place.

    python test_len_ratio.py http://127.0.0.1:8010
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"

# 真实小说的自然长度分布（15-35 字）。要测极端长短混批，用 /api/tts/import
# 传入自定义分割文件，那里不会被服务端重新合并短句。
TEXTS = [
    "深夜的图书馆只剩下一盏灯还亮着。",
    "她翻过一页，纸的声音在空荡的大厅里显得格外清楚。",
    "窗外的雨还在下，玻璃上滑过一条条水痕。",
    "她忽然想起很多年前的一个夏天，也是这样的雨。",
    "那时候她还不知道，有些告别是没有预告的。",
    "书架上的灰尘在灯光里浮着，像很慢很慢的雪。",
    "她把书合上，听见自己的呼吸声。",
    "走廊尽头的窗户透进一点灰蓝色的光。",
    "第三排的架子上标着 A+B=12 的编号，温度计显示 21℃，她抬头看了很久。",
    "有人在楼下喊她的名字，声音被雨声切碎了。",
    "她没有回头，只是把书塞回原处。",
    "那本书里夹着一张旧车票，日期已经模糊了，只看得清一个地名。",
    "她想起来了，那是她第一次离开这座城市的车票。",
    "雨停了，天边露出一线很淡的白。",
    "她把车票放回书里，像把某个夏天重新合上。",
    "灯灭的时候，整栋楼只剩下雨水的味道。",
]
TEXT = "".join(TEXTS)


def get(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())


def main():
    body = json.dumps({"text": TEXT, "voice": "v4_default",
                       "version": "v5turbo", "speed": 1.0}).encode()
    req = urllib.request.Request(f"{BASE}/api/tts/long", data=body,
                                 headers={"Content-Type": "application/json"})
    tid = json.loads(urllib.request.urlopen(req, timeout=30).read())["task_id"]

    t0 = time.time()
    last = -1
    steps = []
    while True:
        p = get(f"{BASE}/api/tts/progress/{tid}")
        if p["done"] != last:
            last = p["done"]
            steps.append(last)
            print(f"  +{time.time()-t0:6.1f}s done={last}/{p['total']}")
        if p["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    print("progress steps:", steps)
    wall = time.time() - t0

    doc = get(f"{BASE}/api/tts/split/{tid}")
    sents, starts, durs = doc["sents"], doc["starts"], doc["durations"]
    print(f"wall {wall:.1f}s for {len(sents)} sentences "
          f"({wall/len(sents):.2f}s each), audio {doc['audio_seconds']:.1f}s")
    print(f"starts monotonic: {all(b > a for a, b in zip(starts, starts[1:]))}")
    print(f"chars vs duration: {[len(s) for s in sents][:6]} -> "
          f"{[round(d, 1) for d in durs][:6]}")
    # 顺序自检：字数与时长应高度相关（严格单调太脆：语速本身有 5% 波动）
    xs = [len(s) for s in sents]
    r = sum((a - sum(xs) / len(xs)) * (b - sum(durs) / len(durs))
            for a, b in zip(xs, durs))
    r /= ((sum((a - sum(xs) / len(xs)) ** 2 for a in xs) ** 0.5)
          * (sum((b - sum(durs) / len(durs)) ** 2 for b in durs) ** 0.5)) or 1
    print(f"correlation(chars, duration) = {r:.3f} "
          f"({'OK' if r > 0.9 else 'SUSPECT'})")


if __name__ == "__main__":
    main()
