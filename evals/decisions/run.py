import json, sys, time, urllib.request
port = sys.argv[1]
rows = [json.loads(l) for l in open(sys.argv[2])]
out = []
for r in rows:
    body = json.dumps({"model": "auto", "messages": [{"role": "user", "content": r["q"]}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
    t = time.time()
    try:
        v = json.load(urllib.request.urlopen(req, timeout=180))
    except Exception as e:
        v = {"error": {"message": str(e)}}
    ms = (time.time() - t) * 1000
    served = v.get("pankhllm", {}).get("model", "ERROR")
    got = served.split(":", 1)[1] if served.split(":")[0] in ("decision", "planner", "plan-route") else "UNSUPPORTED"
    ok = got == r["op"]
    text = (v.get("choices") or [{}])[0].get("message", {}).get("content") or v.get("error", {}).get("message", "")
    out.append((ms, served, ok))
    print(f"{ms:8.1f} ms  {'ok ' if ok else 'BAD'}  {served:<28} {r['q'][:44]:<45} {text[:60]!r}")
lat = sorted(m for m, _, _ in out)
print(f"\ncorrect {sum(1 for *_, ok in out if ok)}/{len(out)}  p50 {lat[len(lat)//2]:.0f} ms  max {lat[-1]:.0f} ms")
