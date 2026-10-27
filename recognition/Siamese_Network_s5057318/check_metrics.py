"""Hand-calculated tests of the AUROC and PR-AUC functions in predict.py."""
import math

from predict import auroc, average_precision, wilson_interval

failures = 0


def check(name, got, expected, tol=1e-4):
    global failures
    ok = abs(got - expected) < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} -> {got:.4f} (expected {expected})")


a, y = [0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]
check("AUROC [0.1,0.4,0.35,0.8] / [0,0,1,1]", auroc(a, y), 0.75)
check("PR-AUC [0.1,0.4,0.35,0.8] / [0,0,1,1]", average_precision(a, y), 0.8333)
b = [0.1, 0.2, 0.8, 0.9]
check("AUROC [0.1,0.2,0.8,0.9] / [0,0,1,1]", auroc(b, y), 1.0)
check("PR-AUC [0.1,0.2,0.8,0.9] / [0,0,1,1]", average_precision(b, y), 1.0)
check("AUROC [0.1,0.4,0.35,0.8] / [1,1,0,0]", auroc(a, [1, 1, 0, 0]), 0.25)
check("AUROC all scores equal / [0,1,0,1]", auroc([0.5] * 4, [0, 1, 0, 1]), 0.5)
check("PR-AUC all scores equal / [0,1,0,1] (= prevalence)",
      average_precision([0.5] * 4, [0, 1, 0, 1]), 0.5)
print("AUROC with one class is nan:", math.isnan(auroc([0.1, 0.2], [0, 0])))

# Wilson 95% interval, hand-calculated, tolerance 0.002.
for k, n, (lo_exp, hi_exp) in ((9, 176, (0.0271, 0.0943)), (0, 10, (0.0, 0.2775))):
    lo, hi = wilson_interval(k, n)
    ok = abs(lo - lo_exp) < 0.002 and abs(hi - hi_exp) < 0.002
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  Wilson({k} of {n}) -> ({lo:.4f}, {hi:.4f}) "
          f"(expected about ({lo_exp}, {hi_exp}))")

print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
raise SystemExit(1 if failures else 0)
