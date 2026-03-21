#!/usr/bin/env python3
"""
DA4BCI Distribution Benchmark
==============================
Replicates the R package's ParallelTEST experiment:

  6 original methods × 10 distribution‑shift scenarios
  + 4 additional Python methods (ART, PT, OT, M3D)

For every (method, distribution) pair the script records:
  - runtime
  - MMD, Energy distance, Mahalanobis before / after adaptation
  - before / after PCA scatter plots (saved as PNG)

Results are written to  results/  inside this directory.

Usage:
    python run_benchmark.py              # all methods, all distributions
    python run_benchmark.py --quick      # small subset for smoke testing
"""

import os, sys, time, json, argparse, warnings
import numpy as np

# ── Imports from da4bci ───────────────────────────────────────────────
from da4bci.methods.tca import domain_adaptation_tca
from da4bci.methods.sa import domain_adaptation_sa
from da4bci.methods.coral import domain_adaptation_coral
from da4bci.methods.gfk import domain_adaptation_gfk
from da4bci.methods.mida import domain_adaptation_mida
from da4bci.methods.riemannian import domain_adaptation_riemannian
from da4bci.methods.art import domain_adaptation_art
from da4bci.methods.pt import domain_adaptation_pt
from da4bci.methods.ot import domain_adaptation_ot
from da4bci.methods.m3d import domain_adaptation_m3d
from da4bci.metrics.kernels import sigma_med
from da4bci.metrics.distance import compute_mmd, compute_energy, compute_mahalanobis

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# =====================================================================
# 1.  Data generation  (mirrors R's generate_data exactly)
# =====================================================================

DIST_NAMES = {
    1:  "Standard Normal",
    2:  "Uniform",
    3:  "Normal (Different Means)",
    4:  "Exponential",
    5:  "Normal (Different SD)",
    6:  "Poisson",
    7:  "Student-t",
    8:  "Binomial",
    9:  "Normal (Another Variant)",
    10: "Normal + Cauchy",
}


def generate_data(n_s, n_t, dist_type, fs=50, t=3, p=50, seed=2025):
    """Generate source / target matrices matching the R benchmark.

    Returns (source, target, dist_name)  with shapes (n_s*fs*t, p).
    """
    rng = np.random.RandomState(seed + dist_type)  # reproducible per dist
    rows_s = n_s * fs * t
    rows_t = n_t * fs * t

    gen = {
        1:  (lambda n: rng.randn(n, p),
             lambda n: rng.randn(n, p)),
        2:  (lambda n: rng.uniform(size=(n, p)),
             lambda n: rng.uniform(size=(n, p))),
        3:  (lambda n: rng.randn(n, p) + 5,
             lambda n: rng.randn(n, p) - 5),
        4:  (lambda n: rng.exponential(size=(n, p)),
             lambda n: rng.exponential(size=(n, p))),
        5:  (lambda n: rng.randn(n, p) * 2,
             lambda n: rng.randn(n, p) * 0.5),
        6:  (lambda n: rng.poisson(lam=3, size=(n, p)).astype(float),
             lambda n: rng.poisson(lam=10, size=(n, p)).astype(float)),
        7:  (lambda n: rng.standard_t(df=5, size=(n, p)),
             lambda n: rng.standard_t(df=10, size=(n, p))),
        8:  (lambda n: rng.binomial(n=10, p=0.3, size=(n, p)).astype(float),
             lambda n: rng.binomial(n=10, p=0.7, size=(n, p)).astype(float)),
        9:  (lambda n: rng.randn(n, p),
             lambda n: rng.randn(n, p) * 3),
        10: (lambda n: rng.randn(n, p),
             lambda n: rng.standard_cauchy(size=(n, p))),
    }

    src_gen, tgt_gen = gen[dist_type]
    return src_gen(rows_s), tgt_gen(rows_t), DIST_NAMES[dist_type]


# =====================================================================
# 2.  Method registry
# =====================================================================

def _run_method(method_name, src, tgt, k, labels=None):
    """Dispatch a single DA method; returns (adapted_src, adapted_tgt)."""
    res = None
    if method_name == "tca":
        res = domain_adaptation_tca(src, tgt, k=k, sigma=10, mu=1e-5)
    elif method_name == "sa":
        res = domain_adaptation_sa(src, tgt, k=k)
    elif method_name == "coral":
        res = domain_adaptation_coral(src, tgt, lam=1e-5)
    elif method_name == "gfk":
        res = domain_adaptation_gfk(src, tgt, dim_subspace=k)
    elif method_name == "mida":
        res = domain_adaptation_mida(src, tgt, k=k, sigma=1, mu=0.1, maximize=True)
    elif method_name == "rd":
        res = domain_adaptation_riemannian(src, tgt)
    elif method_name == "art":
        res = domain_adaptation_art(src, tgt)
    elif method_name == "pt":
        res = domain_adaptation_pt(src, tgt)
    elif method_name == "ot":
        res = domain_adaptation_ot(src, tgt, eps=0.05, maxit=500)
    elif method_name == "m3d":
        if labels is None:
            labels = np.random.RandomState(42).choice([1, 2, 3], size=src.shape[0])
        res = domain_adaptation_m3d(
            src, labels, tgt,
            stage1={"method": "tca", "control": {"k": k, "sigma": 1}},
            stage2={"method": "sa",  "control": {"k": k}},
            l_iter=5, max_dim=k,
        )
    return res["weighted_source_data"], res["target_data"]


# =====================================================================
# 3.  Metric computation
# =====================================================================

def compute_metrics(src, tgt):
    """Return dict of distribution distance metrics."""
    sigma_bw = sigma_med(src, tgt)
    return {
        "mmd":    float(compute_mmd(src, tgt, sigma=sigma_bw)),
        "energy": float(compute_energy(src, tgt)),
        "mahal":  float(compute_mahalanobis(src, tgt)),
    }


# =====================================================================
# 4.  Plotting
# =====================================================================

def save_comparison_plot(src, tgt, Zs, Zt, method, dist_id, dist_name, out_dir):
    """Save a before/after PCA scatter plot (matching R's layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Before
    data_b = np.vstack([src, tgt])
    coords_b = PCA(n_components=2).fit_transform(data_b)
    n_s = src.shape[0]
    axes[0].scatter(coords_b[:n_s, 0], coords_b[:n_s, 1],
                    alpha=0.35, s=8, c="salmon", label="Source")
    axes[0].scatter(coords_b[n_s:, 0], coords_b[n_s:, 1],
                    alpha=0.35, s=8, c="mediumturquoise", label="Target")
    axes[0].set_title(f"Before {dist_name}")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8)

    # After
    data_a = np.vstack([Zs, Zt])
    coords_a = PCA(n_components=2).fit_transform(data_a)
    n_s2 = Zs.shape[0]
    axes[1].scatter(coords_a[:n_s2, 0], coords_a[:n_s2, 1],
                    alpha=0.35, s=8, c="salmon", label="Source")
    axes[1].scatter(coords_a[n_s2:, 0], coords_a[n_s2:, 1],
                    alpha=0.35, s=8, c="mediumturquoise", label="Target")
    axes[1].set_title(f"After {dist_name}")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Method: {method} | DistType: {dist_id}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fname = os.path.join(out_dir, f"{method}_dist_{dist_id}.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


# =====================================================================
# 5.  Summary tables
# =====================================================================

def print_metric_table(records, metric_key, metric_label):
    """Print a reduction‑% heatmap table: methods (rows) × dists (cols)."""
    methods_seen = list(dict.fromkeys(r["method"] for r in records))
    dists_seen   = sorted(set(r["dist"] for r in records))

    lookup = {}
    for r in records:
        lookup[(r["method"], r["dist"])] = r

    # Header
    hdr = f"{'':>6}"
    for d in dists_seen:
        hdr += f" {d:>6}"
    print(f"\n  {metric_label} reduction %  (positive = gap reduced)")
    print(hdr)
    print("  " + "-" * len(hdr))

    for m in methods_seen:
        row = f"{m:>6}"
        for d in dists_seen:
            r = lookup.get((m, d))
            if r and r[f"{metric_key}_before"] > 0:
                red = (1 - r[f"{metric_key}_after"] / r[f"{metric_key}_before"]) * 100
                row += f" {red:>6.1f}"
            else:
                row += f" {'N/A':>6}"
        print(row)


def print_runtime_table(records):
    methods_seen = list(dict.fromkeys(r["method"] for r in records))
    dists_seen   = sorted(set(r["dist"] for r in records))

    lookup = {}
    for r in records:
        lookup[(r["method"], r["dist"])] = r

    hdr = f"{'':>6}"
    for d in dists_seen:
        hdr += f" {d:>8}"
    hdr += f" {'avg':>8}"
    print(f"\n  Runtime (ms)")
    print(hdr)
    print("  " + "-" * len(hdr))

    for m in methods_seen:
        row = f"{m:>6}"
        times = []
        for d in dists_seen:
            r = lookup.get((m, d))
            if r:
                t = r["time_ms"]
                times.append(t)
                row += f" {t:>8.1f}"
            else:
                row += f" {'N/A':>8}"
        avg = np.mean(times) if times else 0
        row += f" {avg:>8.1f}"
        print(row)


# =====================================================================
# 6.  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run only 3 methods × 3 dists for quick testing")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pic_dir = os.path.join(OUT_DIR, "pic")
    os.makedirs(pic_dir, exist_ok=True)

    # R tested 6 methods; we add ART, PT, OT, M3D
    all_methods = ["tca", "sa", "mida", "rd", "coral", "gfk",
                   "art", "pt", "ot", "m3d"]
    all_dists = list(range(1, 11))

    if args.quick:
        all_methods = ["sa", "coral", "art"]
        all_dists = [1, 3, 5]

    # R uses: n_s=10, n_t=10 trials, fs=50, t=3 → 1500 rows, 50 features
    n_s, n_t, fs, t_sec, p = 10, 10, 50, 3, 50
    k = 10  # subspace dim (matches R's k=10)

    total = len(all_methods) * len(all_dists)
    records = []

    print(f"DA4BCI Python Benchmark: {len(all_methods)} methods × "
          f"{len(all_dists)} distributions = {total} runs")
    print(f"Data shape per run: {n_s*fs*t_sec} × {p}")
    print(f"Output: {OUT_DIR}\n")

    done = 0
    for method in all_methods:
        for dist_id in all_dists:
            done += 1
            src, tgt, dist_name = generate_data(n_s, n_t, dist_id,
                                                 fs=fs, t=t_sec, p=p)

            # Metrics before
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_before = compute_metrics(src, tgt)

            # Run DA
            t0 = time.perf_counter()
            try:
                Zs, Zt = _run_method(method, src, tgt, k)
                elapsed = (time.perf_counter() - t0) * 1000
                status = "ok"
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                status = f"FAIL: {e}"
                Zs = Zt = None

            # Metrics after
            if Zs is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        m_after = compute_metrics(Zs, Zt)
                    except Exception:
                        m_after = {"mmd": float("nan"), "energy": float("nan"),
                                   "mahal": float("nan")}
            else:
                m_after = {"mmd": float("nan"), "energy": float("nan"),
                           "mahal": float("nan")}

            rec = {
                "method": method, "dist": dist_id, "dist_name": dist_name,
                "time_ms": round(elapsed, 2), "status": status,
                "mmd_before": m_before["mmd"], "mmd_after": m_after["mmd"],
                "energy_before": m_before["energy"], "energy_after": m_after["energy"],
                "mahal_before": m_before["mahal"], "mahal_after": m_after["mahal"],
            }
            records.append(rec)

            # Progress
            tag = "OK" if status == "ok" else status
            print(f"  [{done:>3}/{total}] {method:>5} × dist {dist_id:>2} "
                  f"({dist_name:<25}) {elapsed:>8.1f} ms  {tag}")

            # Plot
            if Zs is not None:
                try:
                    save_comparison_plot(src, tgt, Zs, Zt,
                                         method, dist_id, dist_name, pic_dir)
                except Exception:
                    pass

    # ── Save JSON ─────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w") as f:
        json.dump(records, f, indent=2, default=str)

    # ── Summary tables ────────────────────────────────────────────────
    ok_records = [r for r in records if r["status"] == "ok"]

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print_runtime_table(ok_records)
    print_metric_table(ok_records, "mmd", "MMD")
    print_metric_table(ok_records, "energy", "Energy")
    print_metric_table(ok_records, "mahal", "Mahalanobis")

    # ── Per-method averages ───────────────────────────────────────────
    print(f"\n  Per-method average across distributions")
    print(f"{'method':>6} {'time_ms':>10} {'MMD_red%':>10} {'Energy_red%':>12} {'Mahal_red%':>12}")
    print("  " + "-" * 58)

    methods_seen = list(dict.fromkeys(r["method"] for r in ok_records))
    for m in methods_seen:
        rows = [r for r in ok_records if r["method"] == m]
        avg_t = np.mean([r["time_ms"] for r in rows])
        mmd_reds = [(1 - r["mmd_after"] / r["mmd_before"]) * 100
                    for r in rows if r["mmd_before"] > 0 and np.isfinite(r["mmd_after"])]
        eng_reds = [(1 - r["energy_after"] / r["energy_before"]) * 100
                    for r in rows if r["energy_before"] > 0 and np.isfinite(r["energy_after"])]
        mah_reds = [(1 - r["mahal_after"] / r["mahal_before"]) * 100
                    for r in rows if r["mahal_before"] > 0 and np.isfinite(r["mahal_after"])]
        print(f"{m:>6} {avg_t:>10.1f} {np.mean(mmd_reds):>10.1f} "
              f"{np.mean(eng_reds):>12.1f} {np.mean(mah_reds):>12.1f}")

    failures = [r for r in records if r["status"] != "ok"]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(f"    {r['method']} × dist {r['dist']}: {r['status']}")

    print(f"\nPlots saved to: {pic_dir}")
    print(f"JSON results:   {os.path.join(OUT_DIR, 'benchmark_results.json')}")


if __name__ == "__main__":
    main()
