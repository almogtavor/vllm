"""Cross-mode equivalence + bit-exactness checks."""
import torch

CASES = [(600, 256), (1100, 512), (2200, 1024), (905, 300), (777, 257), (4099, 2048)]
OUT = "/results/pr44584/out"
print("case                 |V1_2D-OLD_2D|  v1_bitexact   |V2_3D-OLD_3D|  v2_bitexact")
all_ok = True
for sl, sw in CASES:
    m0 = torch.load("%s/cc_m0_%d_%d.pt" % (OUT, sl, sw))
    m1 = torch.load("%s/cc_m1_%d_%d.pt" % (OUT, sl, sw))
    m2 = torch.load("%s/cc_m2_%d_%d.pt" % (OUT, sl, sw))
    d_v1 = (m1["o2"] - m0["o2"]).abs().max().item()
    d_v2 = (m2["o3"] - m0["o3"]).abs().max().item()
    be1 = torch.equal(m1["o2"], m0["o2"])
    be2 = torch.equal(m2["o3"], m0["o3"])
    # also: does V2's 3D match V2's 2D (same logical attention)?
    print("sl=%-5d sw=%-5d   %.3e       %-5s         %.3e       %-5s"
          % (sl, sw, d_v1, be1, d_v2, be2))
    # correctness gate: V1/V2 must agree with baseline within bf16 ULP (<= 2e-2)
    if d_v1 > 2e-2 or d_v2 > 2e-2:
        all_ok = False
print("GATE_PASS" if all_ok else "GATE_FAIL")
