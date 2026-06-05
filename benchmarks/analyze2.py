import json

R = {m: json.load(open("/results/pr44584/out/bench2_mode%d.json" % m)) for m in (0, 1, 2)}


def cmp(key, baseline, newm, label, T):
    import math
    print("\n==== %s ====" % label)
    print("%6s %8s %10s %10s %9s %8s" % ("sw", "tiles", "OLD min", "NEW min", "speedup", "saved%"))
    out = []
    for a, b in zip(R[baseline][key], R[newm][key]):
        sw = a["sw"]
        n = math.ceil(sw / T)            # aligned tiles
        sp = a["min"] / b["min"]
        saved = 100 * (1 - b["min"] / a["min"])
        pred = 100 * (1.0 / (n + 1))     # predicted: 1 of (n+1) floor tiles removed
        extra = " (ns=%d)" % a.get("num_seqs", 0) if "num_seqs" in a else ""
        print("%6d%s %4d->%-4d %10.4f %10.4f %8.3fx %6.1f%% (pred %.1f%%)"
              % (sw, extra, n + 1, n, a["min"], b["min"], sp, saved, pred))
        out.append(dict(sw=sw, num_seqs=a.get("num_seqs"), tiles_old=n + 1, tiles_new=n,
                        old_min=a["min"], new_min=b["min"], speedup=sp, saved_pct=saved, pred_pct=pred))
    return out


s2 = cmp("straddle2d", 0, 1, "2D straddle (TILE=32): OLD vs V1  [tile-saving step]", 32)
a2 = cmp("avg2d", 0, 1, "2D residue-averaged (TILE=32): OLD vs V1", 32)
s3 = cmp("straddle3d", 0, 2, "3D straddle (TILE=16): OLD vs V2  [tile-saving step]", 16)

json.dump(dict(straddle2d=s2, avg2d=a2, straddle3d=s3),
          open("/results/pr44584/out/bench2_summary.json", "w"), indent=2)
print("\nwrote bench2_summary.json")
