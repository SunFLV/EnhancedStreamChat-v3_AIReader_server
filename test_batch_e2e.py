"""End-to-end check of the batched long-text path against a test instance."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"

TEXT = ("深夜的图书馆只剩下一盏灯还亮着。她翻过一页，纸的声音在空荡的大厅里显得格外清楚。"
        "窗外的雨还在下，玻璃上滑过一条条水痕。她忽然想起很多年前的一个夏天，也是这样的雨。"
        "那时候她还不知道，有些告别是没有预告的。书架上的灰尘在灯光里浮着，像很慢很慢的雪。"
        "她把书合上，听见自己的呼吸声。走廊尽头的窗户透进一点灰蓝色的光。"
        "第三排的架子上标着 A+B=12 的编号，温度计显示 21℃，她抬头看了很久。")


def get(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())


def main():
    body = json.dumps({"text": TEXT, "voice": "v4_default",
                       "version": "v5turbo", "speed": 1.0}).encode()
    req = urllib.request.Request(f"{BASE}/api/tts/long", data=body,
                                 headers={"Content-Type": "application/json"})
    tid = json.loads(urllib.request.urlopen(req, timeout=30).read())["task_id"]
    print("task", tid, flush=True)

    t0 = time.time()
    seen = []
    while True:
        p = get(f"{BASE}/api/tts/progress/{tid}")
        if not seen or seen[-1] != p["done"]:
            seen.append(p["done"])
            print(f"  {time.time()-t0:6.1f}s done={p['done']}/{p['total']} {p['status']}",
                  flush=True)
        if p["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    print("progress steps:", seen, f"wall {time.time()-t0:.1f}s", flush=True)

    doc = get(f"{BASE}/api/tts/split/{tid}")
    print("sents", len(doc["sents"]), "starts", len(doc["starts"]),
          "audio", doc["audio_seconds"], flush=True)
    print("last sentence:", doc["sents"][-1], flush=True)
    print("symbol line kept for display:",
          [s for s in doc["sents"] if "A+B" in s][:1], flush=True)


if __name__ == "__main__":
    main()
