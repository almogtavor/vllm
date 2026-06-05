import json

R = {m: json.load(open("/results/pr44584/out/results_mode%d.json" % m)) for m in (0, 1, 2)}
summary = {"bench2d": [], "bench3d": []}

print("==== 2D path (TILE=32): OLD(mode0) vs V1(mode1) ====")
print("%6s %10s %10s %9s %8s" % ("sw", "OLD ms", "V1 ms", "speedup", "saved%"))
for a, b in zip(R[0]["bench2d"], R[1]["bench2d"]):
    sp = a["ms"] / b["ms"]
    saved = 100 * (1 - b["ms"] / a["ms"])
    print("%6d %10.4f %10.4f %8.3fx %7.1f%%" % (a["sw"], a["ms"], b["ms"], sp, saved))
    summary["bench2d"].append(dict(sw=a["sw"], old_ms=a["ms"], new_ms=b["ms"], speedup=sp, saved_pct=saved))

print()
print("==== 3D path (TILE=16): OLD(mode0) vs V2(mode2) ====")
print("%6s %4s %10s %10s %9s %8s" % ("sw", "ns", "OLD ms", "V2 ms", "speedup", "saved%"))
for a, b in zip(R[0]["bench3d"], R[2]["bench3d"]):
    sp = a["ms"] / b["ms"]
    saved = 100 * (1 - b["ms"] / a["ms"])
    print("%6d %4d %10.4f %10.4f %8.3fx %7.1f%%" % (a["sw"], a["num_seqs"], a["ms"], b["ms"], sp, saved))
    summary["bench3d"].append(dict(sw=a["sw"], num_seqs=a["num_seqs"], old_ms=a["ms"], new_ms=b["ms"], speedup=sp, saved_pct=saved))

# also V1 vs V2 on 2D should match (V2 doesn't change 2D)
json.dump(summary, open("/results/pr44584/out/bench_summary.json", "w"), indent=2)
print("\nwrote bench_summary.json")
