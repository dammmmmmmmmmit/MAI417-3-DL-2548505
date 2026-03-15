import json, pathlib

NB = pathlib.Path(r"C:\Users\adity\Desktop\college stuff\sem3\dL\lab4\2548505_lab4.ipynb")
nb = json.load(NB.open(encoding="utf-8"))

checks = {
    "Block 2.3 vectorised":     ("vectorised",           True),
    "Block 2.3 stable sigmoid": ("xp.clip(x, -15",       True),
    "Block 2.4 grad_clip":      ("grad_clip=1.0",         True),
    "Block 2.6 self-contained": ("self-contained",        True),
    "Block 2.6 lr=0.001":       ("lr=0.001",              True),
    "Block 2.6 NaN guard":      ("math.isfinite(loss)",   True),
    "Old triple loop GONE":     ("for n in range(N):",    False),
    "Old LR=0.01 GONE":         ("LR=0.01",               False),
}

all_src = "".join("".join(cell.get("source", [])) for cell in nb["cells"])

all_ok = True
for desc, (kw, should) in checks.items():
    found = kw in all_src
    ok = found == should
    label = "OK  " if ok else "FAIL"
    print(f"[{label}] {desc}: '{kw}' found={found} expected={should}")
    if not ok:
        all_ok = False

print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
