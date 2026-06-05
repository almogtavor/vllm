"""Per-mode correctness vs fp32 ref; saves outputs for cross-mode compare."""
import os
import bench_kernel as bk
import torch

CASES = [(600, 256), (1100, 512), (2200, 1024), (905, 300), (777, 257), (4099, 2048)]
maxd2 = maxd3 = 0.0
for sl, sw in CASES:
    inp = bk.make_decode_inputs(3, sl, 16, 8, 128, 16, seed=sl)
    ref = bk.ref_decode(inp, sw)
    o2 = bk.run_kernel(inp, sw, False).float()
    o3 = bk.run_kernel(inp, sw, True).float()
    maxd2 = max(maxd2, (o2 - ref).abs().max().item())
    maxd3 = max(maxd3, (o3 - ref).abs().max().item())
    torch.save({"o2": o2.cpu(), "o3": o3.cpu()},
               os.path.join(bk.OUT_DIR, "cc_m%d_%d_%d.pt" % (bk.MODE, sl, sw)))
print("mode %d: worst 2D|k-ref|=%.4e  worst 3D|k-ref|=%.4e" % (bk.MODE, maxd2, maxd3))
