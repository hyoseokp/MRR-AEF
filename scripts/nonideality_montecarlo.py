#!/usr/bin/env python3
"""Monte Carlo non-ideality validation for cascaded-MRR exponential block.

Outputs:
- nonideal_mc_cdf.png
- nonideal_softmax_cdf.png
- nonideality_summary.csv

Model scope (engineering-level, compact):
- Per-ring static detuning variation (sigma_a)
- Per-ring sensitivity variation (sigma_b_rel)
- Global thermal drift and crosstalk-like slope term
- Stage insertion loss + detector noise floor
- Control-channel noise
- One-point chip calibration at I=L (enforce y(L)=1)

This is not a full photonic circuit simulator; it is a first-order robustness sweep.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Scenario:
    name: str
    sigma_a: float
    sigma_b_rel: float
    sigma_th: float
    sigma_xt: float
    sigma_I: float
    il_stage_db_mu: float
    il_stage_db_sigma: float
    sigma_det_abs: float


def approx_exp_block(
    I: np.ndarray,
    *,
    a_nom: float,
    b_nom: float,
    N: int,
    rng: np.random.Generator,
    sc: Scenario,
    L: float,
) -> np.ndarray:
    # Per-ring static offsets and slope spread
    a_k = a_nom + rng.normal(0.0, sc.sigma_a, size=N)
    b_k = b_nom * (1.0 + rng.normal(0.0, sc.sigma_b_rel, size=N))

    # Global thermal drift + crosstalk-like I-dependent drift (chip-level)
    d_th = rng.normal(0.0, sc.sigma_th)
    d_xt = rng.normal(0.0, sc.sigma_xt)

    # Control noise
    I_noisy = np.clip(I + rng.normal(0.0, sc.sigma_I, size=I.shape), 0.0, L)

    # Multiplicative transmission
    y = np.ones_like(I_noisy)
    for k in range(N):
        u = a_k[k] + b_k[k] * I_noisy + d_th + d_xt * (I_noisy / L)
        y *= 1.0 / (1.0 + u * u)

    # Insertion loss (chip-level, static for a token)
    il_stages = rng.normal(sc.il_stage_db_mu, sc.il_stage_db_sigma, size=N)
    il_total_db = float(np.sum(il_stages))
    y *= 10.0 ** (-il_total_db / 10.0)

    # One-point chip calibration at I=L to match y(L)=1 (performed on noiseless calibration sample)
    yL_cal = np.ones(1)
    for k in range(N):
        uL = a_k[k] + b_k[k] * L + d_th + d_xt
        yL_cal *= 1.0 / (1.0 + uL * uL)
    yL_cal = float(yL_cal[0]) * 10.0 ** (-il_total_db / 10.0)
    C_cal = 1.0 / max(yL_cal, 1e-12)

    # Runtime detector noise (post-calibration measurement)
    y += rng.normal(0.0, sc.sigma_det_abs, size=I.shape)
    y = np.clip(y, 1e-12, None)

    y_hat = C_cal * y

    return y_hat


def run_block_metrics(
    *,
    L: float,
    N: int,
    a_nom: float,
    b_nom: float,
    scenario: Scenario,
    n_chips: int,
    n_grid: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    I = np.linspace(0.0, L, n_grid)
    target = np.exp(I - L)
    # Dynamic-range window used for robust block-level reporting (avoid detector-floor dominated tail)
    mask = target >= 1.0e-3

    max_rel = np.zeros(n_chips)
    p99_rel = np.zeros(n_chips)
    mean_rel = np.zeros(n_chips)
    einf_log = np.zeros(n_chips)

    for i in range(n_chips):
        y_hat = approx_exp_block(I, a_nom=a_nom, b_nom=b_nom, N=N, rng=rng, sc=scenario, L=L)
        y_win = y_hat[mask]
        target_win = target[mask]
        I_win = I[mask]
        rel = np.abs(y_win / target_win - 1.0)
        log_err = np.abs(np.log(np.maximum(y_win, 1e-15)) - (I_win - L))
        max_rel[i] = rel.max()
        p99_rel[i] = np.percentile(rel, 99)
        mean_rel[i] = rel.mean()
        einf_log[i] = log_err.max()

    return {
        "max_rel": max_rel,
        "p99_rel": p99_rel,
        "mean_rel": mean_rel,
        "einf_log": einf_log,
    }


def run_softmax_metrics(
    *,
    L: float,
    N: int,
    a_nom: float,
    b_nom: float,
    scenario: Scenario,
    n_chips: int,
    n_vec_per_chip: int,
    vec_len: int,
    seed: int,
):
    rng = np.random.default_rng(seed + 1000)

    kl_vals = []
    max_abs_p = []

    for _ in range(n_chips):
        # Random vectors in approximately practical logit range [−L, 0] after max-shift
        U = -rng.uniform(0.0, L, size=(n_vec_per_chip, vec_len))
        I = U + L  # [0,L]

        # Reuse one chip draw by evaluating all I points vectorized through same random params
        # Draw chip params
        a_k = a_nom + rng.normal(0.0, scenario.sigma_a, size=N)
        b_k = b_nom * (1.0 + rng.normal(0.0, scenario.sigma_b_rel, size=N))
        d_th = rng.normal(0.0, scenario.sigma_th)
        d_xt = rng.normal(0.0, scenario.sigma_xt)
        il_stages = rng.normal(scenario.il_stage_db_mu, scenario.il_stage_db_sigma, size=N)
        il_total = 10.0 ** (-float(np.sum(il_stages)) / 10.0)

        I_noisy = np.clip(I + rng.normal(0.0, scenario.sigma_I, size=I.shape), 0.0, L)
        y = np.ones_like(I_noisy)
        for k in range(N):
            u = a_k[k] + b_k[k] * I_noisy + d_th + d_xt * (I_noisy / L)
            y *= 1.0 / (1.0 + u * u)
        y *= il_total
        y += rng.normal(0.0, scenario.sigma_det_abs, size=y.shape)
        y = np.clip(y, 1e-12, None)

        # Chip one-point calibration using I=L reference
        yL = np.prod(1.0 / (1.0 + (a_k + b_k * L + d_th + d_xt) ** 2)) * il_total
        yL = max(float(yL), 1e-12)
        C_cal = 1.0 / yL
        y_hat = C_cal * y

        p_hat = y_hat / np.sum(y_hat, axis=1, keepdims=True)
        p_ref_raw = np.exp(U)
        p_ref = p_ref_raw / np.sum(p_ref_raw, axis=1, keepdims=True)

        eps = 1e-12
        kl = np.sum(p_ref * (np.log(p_ref + eps) - np.log(p_hat + eps)), axis=1)
        mad = np.max(np.abs(p_hat - p_ref), axis=1)
        kl_vals.append(kl)
        max_abs_p.append(mad)

    kl_vals = np.concatenate(kl_vals)
    max_abs_p = np.concatenate(max_abs_p)
    return {
        "kl": kl_vals,
        "max_abs_p": max_abs_p,
    }


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--a", type=float, default=-1.4588)
    ap.add_argument("--b", type=float, default=0.10202)
    ap.add_argument("--chips", type=int, default=2000)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--vec-len", type=int, default=128)
    ap.add_argument("--vec-per-chip", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260221)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        Scenario(
            name="Nominal",
            sigma_a=0.020,
            sigma_b_rel=0.020,
            sigma_th=0.015,
            sigma_xt=0.012,
            sigma_I=0.004,
            il_stage_db_mu=0.12,
            il_stage_db_sigma=0.03,
            sigma_det_abs=3.0e-6,
        ),
        Scenario(
            name="Stress",
            sigma_a=0.032,
            sigma_b_rel=0.032,
            sigma_th=0.025,
            sigma_xt=0.020,
            sigma_I=0.007,
            il_stage_db_mu=0.18,
            il_stage_db_sigma=0.05,
            sigma_det_abs=6.0e-6,
        ),
    ]

    summary_rows = []

    # --- block-level error CDF ---
    plt.figure(figsize=(6.4, 4.4))
    for idx, sc in enumerate(scenarios):
        bm = run_block_metrics(
            L=args.L,
            N=args.N,
            a_nom=args.a,
            b_nom=args.b,
            scenario=sc,
            n_chips=args.chips,
            n_grid=args.grid,
            seed=args.seed + 111 * idx,
        )
        # Convert worst-case log error to equivalent worst-case relative error bound: exp(E)-1
        x = np.sort(100.0 * (np.exp(bm["einf_log"]) - 1.0))
        x = np.clip(x, 1e-6, None)  # avoid log-scale singularity at x=0
        y = np.linspace(0.0, 1.0, len(x), endpoint=False)
        plt.plot(x, y, lw=2, label=f"{sc.name}")

        # softmax metrics
        sm = run_softmax_metrics(
            L=args.L,
            N=args.N,
            a_nom=args.a,
            b_nom=args.b,
            scenario=sc,
            n_chips=max(600, args.chips // 3),
            n_vec_per_chip=args.vec_per_chip,
            vec_len=args.vec_len,
            seed=args.seed + 777 * idx,
        )

        summary_rows.append(
            {
                "scenario": sc.name,
                "block_einf_equiv_rel_median_pct": pct(100.0 * (np.exp(bm["einf_log"]) - 1.0), 50),
                "block_einf_equiv_rel_p90_pct": pct(100.0 * (np.exp(bm["einf_log"]) - 1.0), 90),
                "block_einf_equiv_rel_p95_pct": pct(100.0 * (np.exp(bm["einf_log"]) - 1.0), 95),
                "block_mean_rel_median_pct": pct(100.0 * bm["mean_rel"], 50),
                "softmax_kl_median": pct(sm["kl"], 50),
                "softmax_kl_p95": pct(sm["kl"], 95),
                "softmax_max_abs_p_median_pct": pct(100.0 * sm["max_abs_p"], 50),
                "softmax_max_abs_p_p95_pct": pct(100.0 * sm["max_abs_p"], 95),
            }
        )

    plt.xscale("log")
    plt.xlabel("Equivalent worst-case relative bound from log-error, exp(E_inf)-1 (%) [log x-axis]")
    plt.ylabel("Empirical CDF")
    plt.title("Monte Carlo robustness of the N=10 exponential block (L=8)")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "nonideal_mc_cdf.png", dpi=260)

    # --- softmax p-error CDF ---
    plt.figure(figsize=(6.4, 4.4))
    for row in summary_rows:
        # Re-run quick sampling for plotting consistency
        sc = next(s for s in scenarios if s.name == row["scenario"])
        sm = run_softmax_metrics(
            L=args.L,
            N=args.N,
            a_nom=args.a,
            b_nom=args.b,
            scenario=sc,
            n_chips=700,
            n_vec_per_chip=args.vec_per_chip,
            vec_len=args.vec_len,
            seed=args.seed + (991 if sc.name == "Nominal" else 992),
        )
        x = np.sort(100.0 * sm["max_abs_p"])
        y = np.linspace(0.0, 1.0, len(x), endpoint=False)
        plt.plot(x, y, lw=2, label=sc.name)

    plt.xlabel("Softmax max |Δp| (%)")
    plt.ylabel("Empirical CDF")
    plt.title("End-to-end softmax probability error under non-idealities")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "nonideal_softmax_cdf.png", dpi=260)

    # --- CSV summary ---
    csv_path = outdir / "nonideality_summary.csv"
    cols = [
        "scenario",
        "block_einf_equiv_rel_median_pct",
        "block_einf_equiv_rel_p90_pct",
        "block_einf_equiv_rel_p95_pct",
        "block_mean_rel_median_pct",
        "softmax_kl_median",
        "softmax_kl_p95",
        "softmax_max_abs_p_median_pct",
        "softmax_max_abs_p_p95_pct",
    ]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    print(f"[done] wrote {outdir / 'nonideal_mc_cdf.png'}")
    print(f"[done] wrote {outdir / 'nonideal_softmax_cdf.png'}")
    print(f"[done] wrote {csv_path}")
    print("\n# SUMMARY")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
