#!/usr/bin/env python3
"""Closed-loop normalization validation for probe-intensity feedback.

Model:
  S(t) = G(t) * P(t) + n(t)
  tau dP/dt = -P + u_delayed
  u = Kp * e + Ki * integral(e),  e = S_ref - S

Outputs:
- feedback_step_response.png
- feedback_stability_map.png
- feedback_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimConfig:
    tau: float = 1.0
    dt: float = 0.002
    t_end: float = 14.0
    P_max: float = 6.0
    noise_sigma: float = 0.0


def simulate_loop(
    *,
    Kp: float,
    Ki: float,
    Td: float,
    cfg: SimConfig,
    G_profile,
    S_ref: float,
):
    n = int(cfg.t_end / cfg.dt)
    t = np.arange(n) * cfg.dt

    delay_steps = max(0, int(round(Td / cfg.dt)))

    P = np.zeros(n)
    S = np.zeros(n)
    e = np.zeros(n)
    u = np.zeros(n)

    # Start from steady state at initial G(0) to isolate disturbance response.
    G0 = float(G_profile(0.0))
    Pk = max(S_ref / max(G0, 1e-9), 0.0)
    integ = Pk / max(Ki, 1e-9)
    u_buf = [Pk] * (delay_steps + 1)

    rng = np.random.default_rng(20260221)

    for k in range(n):
        tk = t[k]
        Gk = float(G_profile(tk))
        noise = rng.normal(0.0, cfg.noise_sigma)

        Sk = Gk * Pk + noise
        ek = S_ref - Sk

        integ += ek * cfg.dt
        uk = Kp * ek + Ki * integ
        uk = float(np.clip(uk, 0.0, cfg.P_max))

        u_buf.append(uk)
        u_del = u_buf.pop(0)

        dP = (-Pk + u_del) / cfg.tau
        Pk = Pk + cfg.dt * dP
        Pk = max(Pk, 0.0)

        P[k] = Pk
        S[k] = Sk
        e[k] = ek
        u[k] = uk

    return t, P, S, e, u


def metrics(t, S, S_ref, settle_band=0.02):
    err = np.abs(S - S_ref)
    overshoot = max(0.0, (np.max(S) - S_ref) / S_ref)

    # settling time: first time after which remains inside band
    band = settle_band * S_ref
    settle_t = np.nan
    for i in range(len(t)):
        if np.all(err[i:] <= band):
            settle_t = t[i]
            break

    tail = S[int(0.8 * len(S)) :]
    ripple = np.std(tail) / S_ref
    return overshoot, settle_t, ripple


def is_stable(t, S, S_ref):
    # practical stability criterion
    if np.any(~np.isfinite(S)):
        return False
    if np.max(S) > 4.0 * S_ref:
        return False
    tail = S[int(0.8 * len(S)) :]
    if np.mean(np.abs(tail - S_ref)) > 0.08 * S_ref:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--t-end", type=float, default=14.0)
    ap.add_argument("--sref", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260221)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(tau=args.tau, dt=args.dt, t_end=args.t_end, P_max=6.0, noise_sigma=0.002)

    # Piecewise G(t): step at 4 tau and 9 tau (representing token/batch change)
    def G_step(t):
        if t < 4.0:
            return 1.0
        elif t < 9.0:
            return 1.25
        return 0.85

    # Three representative controllers
    cases = [
        ("Stable", 0.55, 0.80, 0.20),
        ("Marginal", 0.95, 1.60, 0.45),
        ("Unstable", 1.20, 2.20, 0.75),
    ]

    # Step-response plot
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    summary_rows = []
    for ax, (name, Kp, Ki, Td) in zip(axs, cases):
        t, P, S, e, u = simulate_loop(Kp=Kp, Ki=Ki, Td=Td, cfg=cfg, G_profile=G_step, S_ref=args.sref)
        ov, st, rip = metrics(t, S, args.sref)
        stable = is_stable(t, S, args.sref)

        G_vals = np.array([G_step(tt) for tt in t])

        ax.plot(t, S, lw=1.8, label="Summed output $S(t)$")
        ax.plot(t, np.ones_like(t) * args.sref, "k--", lw=1.1, label="$S_{ref}$")
        ax.plot(t, G_vals / np.max(G_vals), color="tab:gray", lw=1.0, alpha=0.7, label="scaled $G(t)$")
        ax.set_ylabel("Norm. level")
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{name}: Kp={Kp:.2f}, Ki={Ki:.2f}, Td/tau={Td/cfg.tau:.2f}, "
            f"OS={100*ov:.1f}%, Ts={st if np.isfinite(st) else float('nan'):.2f}"
        )

        summary_rows.append(
            {
                "case": name,
                "Kp": Kp,
                "Ki": Ki,
                "Td_over_tau": Td / cfg.tau,
                "overshoot_pct": 100.0 * ov,
                "settling_time_tau": float(st / cfg.tau) if np.isfinite(st) else np.nan,
                "tail_ripple_pct": 100.0 * rip,
                "stable": int(stable),
            }
        )

    axs[-1].set_xlabel("Time (normalized by tau)")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[:2], labels[:2], fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir / "feedback_step_response.png", dpi=260)

    # Stability map over (Kp,Ki) for multiple delays
    Kp_grid = np.linspace(0.1, 1.4, 38)
    Ki_grid = np.linspace(0.1, 2.4, 40)
    delays = [0.0, 0.2, 0.5, 0.8]

    fig2, axes2 = plt.subplots(2, 2, figsize=(8.4, 6.8), sharex=True, sharey=True)
    axes2 = axes2.ravel()

    stable_fraction_rows = []

    for ax, d in zip(axes2, delays):
        M = np.zeros((len(Ki_grid), len(Kp_grid)))
        stable_count = 0
        total = 0
        for i, Ki in enumerate(Ki_grid):
            for j, Kp in enumerate(Kp_grid):
                t, P, S, e, u = simulate_loop(Kp=float(Kp), Ki=float(Ki), Td=d, cfg=cfg, G_profile=G_step, S_ref=args.sref)
                st = is_stable(t, S, args.sref)
                M[i, j] = 1.0 if st else 0.0
                stable_count += int(st)
                total += 1

        frac = stable_count / total
        stable_fraction_rows.append({"Td_over_tau": d / cfg.tau, "stable_fraction": frac})

        ax.imshow(
            M,
            origin="lower",
            extent=[Kp_grid.min(), Kp_grid.max(), Ki_grid.min(), Ki_grid.max()],
            aspect="auto",
            cmap="Greens",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Td/tau={d/cfg.tau:.1f}, stable={100*frac:.1f}%")
        ax.grid(alpha=0.1)

    for ax in axes2[2:]:
        ax.set_xlabel("Kp")
    for ax in (axes2[0], axes2[2]):
        ax.set_ylabel("Ki")

    plt.tight_layout()
    plt.savefig(outdir / "feedback_stability_map.png", dpi=260)

    # CSV summary
    csv_path = outdir / "feedback_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("section,case,Kp,Ki,Td_over_tau,overshoot_pct,settling_time_tau,tail_ripple_pct,stable,stable_fraction\n")
        for r in summary_rows:
            f.write(
                f"step,{r['case']},{r['Kp']},{r['Ki']},{r['Td_over_tau']},{r['overshoot_pct']},{r['settling_time_tau']},{r['tail_ripple_pct']},{r['stable']},\n"
            )
        for r in stable_fraction_rows:
            f.write(
                f"map,delay_scan,,,,{r['Td_over_tau']},,,,{r['stable_fraction']}\n"
            )

    print(f"[done] wrote {outdir / 'feedback_step_response.png'}")
    print(f"[done] wrote {outdir / 'feedback_stability_map.png'}")
    print(f"[done] wrote {csv_path}")
    print("\n# STEP CASES")
    for r in summary_rows:
        print(r)
    print("\n# STABILITY FRACTION")
    for r in stable_fraction_rows:
        print(r)


if __name__ == "__main__":
    main()
