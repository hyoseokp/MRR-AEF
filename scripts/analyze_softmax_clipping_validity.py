#!/usr/bin/env python3
"""Analyze softmax clipping validity on empirical attention rows.

For each attention row, define shifted logits
    u = s - max(s) <= 0,
and clipped logits at threshold t (t <= 0)
    u^(t) = max(u, t).

Let p = softmax(s) and p^(t) = softmax(u^(t)).
We report cumulative softmax error
    E_cum(t) = 0.5 * ||p^(t) - p||_1,
via p50/p90/p95/p99 over all valid attention rows.

Outputs:
- clipping_error_vs_t_per_run.csv
- clipping_error_vs_t_global.csv
- clipping_error_vs_t_quantiles.png
- clipping_threshold_selection.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_URLS = {
    "tiny_shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "pride_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def fetch_text_cached(dataset_name: str, cache_dir: Path) -> str:
    ensure_dir(cache_dir)
    path = cache_dir / f"{dataset_name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    url = DATASET_URLS[dataset_name]
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text

    if dataset_name == "pride_prejudice":
        sm = "*** START OF THE PROJECT GUTENBERG EBOOK"
        em = "*** END OF THE PROJECT GUTENBERG EBOOK"
        if sm in text:
            text = text.split(sm, 1)[1]
        if em in text:
            text = text.split(em, 1)[0]

    path.write_text(text, encoding="utf-8")
    return text


class Hist:
    def __init__(self, vmin: float, vmax: float, nbins: int):
        self.edges = np.linspace(vmin, vmax, nbins + 1, dtype=np.float64)
        self.counts = np.zeros(nbins, dtype=np.int64)

    def add(self, vals: np.ndarray):
        v = np.clip(vals, self.edges[0], np.nextafter(self.edges[-1], self.edges[0]))
        b = np.searchsorted(self.edges, v, side="right") - 1
        np.add.at(self.counts, b, 1)

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.counts)
        if c[-1] <= 0:
            return float("nan")
        target = q * c[-1]
        i = int(np.searchsorted(c, target, side="left"))
        i = min(max(i, 0), len(self.counts) - 1)
        c_prev = 0 if i == 0 else c[i - 1]
        c_curr = c[i]
        l, r = self.edges[i], self.edges[i + 1]
        if c_curr == c_prev:
            return float(l)
        frac = (target - c_prev) / (c_curr - c_prev)
        return float(l + frac * (r - l))


def iter_windows(token_ids, seq_len: int, stride: int, max_windows: int):
    n = len(token_ids)
    done = 0
    for s in range(0, n - seq_len + 1, stride):
        yield token_ids[s : s + seq_len]
        done += 1
        if done >= max_windows:
            break


def parse_thresholds(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No thresholds provided")
    if any(v > 0 for v in vals):
        raise ValueError("All thresholds must be <= 0")
    # Keep user order to preserve sweep direction in reports.
    return vals


def analyze_one(
    model_name: str,
    dataset_name: str,
    text: str,
    out_dir: Path,
    *,
    seq_len: int,
    stride: int,
    max_windows: int,
    min_query_index: int,
    thresholds: np.ndarray,
    err_max: float,
    nbins: int,
    device: str,
):
    print(f"\n[run] {model_name} | {dataset_name}")
    print(f"[mem] before load: {rss_gb():.2f} GB")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = int(1e9)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval().to(device)

    print(f"[mem] after load: {rss_gb():.2f} GB")

    token_ids = tok(text, add_special_tokens=False, return_attention_mask=False, truncation=False)["input_ids"]

    hists = [Hist(vmin=0.0, vmax=err_max, nbins=nbins) for _ in thresholds]
    npz_hists = []

    rows = 0
    windows_done = 0
    eps = 1e-30
    floor_q = np.exp(thresholds).astype(np.float64)[:, None, None]  # [T,1,1]

    with torch.no_grad():
        for w_ids in iter_windows(token_ids, seq_len=seq_len, stride=stride, max_windows=max_windows):
            x = torch.tensor(w_ids, dtype=torch.long, device=device).unsqueeze(0)
            out = model(x, output_attentions=True, use_cache=False)

            for att in out.attentions:
                a = att[0].float().cpu().numpy()  # [H,T,T]
                H, T, _ = a.shape
                for t in range(min_query_index, T):
                    p = np.clip(a[:, t, : t + 1].astype(np.float64), eps, 1.0)  # [H,K]

                    # Recover unnormalized shifted exponentials q = exp(u) via q = p / p_max.
                    p_max = np.max(p, axis=1, keepdims=True)
                    q = p / np.clip(p_max, eps, None)  # [H,K], max=1

                    # Broadcast over thresholds: [Nt,H,K]
                    q_clip = np.maximum(q[None, :, :], floor_q)
                    p_clip = q_clip / np.sum(q_clip, axis=2, keepdims=True)

                    # E_cum(t) = 0.5 * L1 distance, shape [Nt,H]
                    err = 0.5 * np.sum(np.abs(p_clip - p[None, :, :]), axis=2)

                    for i, h in enumerate(hists):
                        h.add(err[i])

                    rows += H

            windows_done += 1
            if windows_done % 20 == 0:
                print(f"  windows={windows_done}/{max_windows} rows={rows} rss={rss_gb():.2f} GB")

            del out, x

    model_slug = slugify(model_name)
    ds_slug = slugify(dataset_name)
    for i, thr in enumerate(thresholds):
        npz_path = out_dir / f"hist_cliperr_t{thr:.2f}__{model_slug}__{ds_slug}.npz"
        np.savez_compressed(npz_path, edges=hists[i].edges, counts=hists[i].counts)
        npz_hists.append(npz_path)

    result_rows = []
    for i, thr in enumerate(thresholds):
        row = {
            "model": model_name,
            "dataset": dataset_name,
            "threshold_t": float(thr),
            "rows": rows,
            "windows": windows_done,
            "seq_len": seq_len,
            "stride": stride,
            "min_query_index": min_query_index,
            "p50": hists[i].quantile(0.50),
            "p90": hists[i].quantile(0.90),
            "p95": hists[i].quantile(0.95),
            "p99": hists[i].quantile(0.99),
            "rss_gb_end": rss_gb(),
            "hist_npz": str(npz_hists[i]),
        }
        result_rows.append(row)

    # cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[done] rows={rows} windows={windows_done} "
        f"p99@t={thresholds[-1]:.2f}:{result_rows[-1]['p99']:.6f} rss={rss_gb():.2f} GB"
    )

    return result_rows


def aggregate_global(per_run_rows: Sequence[dict], thresholds: Sequence[float], err_max: float, nbins: int):
    global_hists = {float(t): Hist(vmin=0.0, vmax=err_max, nbins=nbins) for t in thresholds}

    for r in per_run_rows:
        t = float(r["threshold_t"])
        d = np.load(r["hist_npz"])
        counts = d["counts"]
        global_hists[t].counts += counts

    out = []
    for t in thresholds:
        h = global_hists[float(t)]
        out.append(
            {
                "threshold_t": float(t),
                "rows": int(np.sum(h.counts)),
                "p50": h.quantile(0.50),
                "p90": h.quantile(0.90),
                "p95": h.quantile(0.95),
                "p99": h.quantile(0.99),
            }
        )
    return out


def choose_threshold(global_rows: Sequence[dict], budget: float) -> Tuple[float | None, str]:
    # Pick least negative threshold (closest to 0) satisfying p99 <= budget.
    valid = [r for r in global_rows if float(r["p99"]) <= budget]
    if not valid:
        return None, "No threshold satisfies budget."
    chosen = max(valid, key=lambda r: float(r["threshold_t"]))
    return float(chosen["threshold_t"]), f"Chosen threshold t={chosen['threshold_t']:.2f} with p99={chosen['p99']:.6g}."


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_quantiles(global_rows: Sequence[dict], out_path: Path):
    t = np.array([float(r["threshold_t"]) for r in global_rows], dtype=np.float64)
    p50 = np.array([float(r["p50"]) for r in global_rows], dtype=np.float64)
    p90 = np.array([float(r["p90"]) for r in global_rows], dtype=np.float64)
    p95 = np.array([float(r["p95"]) for r in global_rows], dtype=np.float64)
    p99 = np.array([float(r["p99"]) for r in global_rows], dtype=np.float64)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(t, p50, "o-", lw=1.8, label="p50")
    plt.plot(t, p90, "o-", lw=1.8, label="p90")
    plt.plot(t, p95, "o-", lw=1.8, label="p95")
    plt.plot(t, p99, "o-", lw=2.0, label="p99")
    plt.axhline(1e-3, color="red", ls="--", lw=1.2, label="0.1% budget (1e-3)")
    plt.yscale("log")
    plt.xlabel("Clipping threshold t (u clipped below t)")
    plt.ylabel(r"Cumulative softmax error $E_{cum}(t)$")
    plt.title("Clipping-validity sweep on empirical attention rows")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--cache-dir",
        default="/Users/bot_s/.openclaw/workspace/research/2026-02-19-mrr-exponential/data/logit_corpora",
    )
    ap.add_argument("--models", nargs="+", default=["distilgpt2", "gpt2"])
    ap.add_argument("--datasets", nargs="+", default=["tiny_shakespeare", "pride_prejudice"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=120)
    ap.add_argument("--min-query-index", type=int, default=16)
    ap.add_argument("--thresholds", type=str, default="-14,-13,-12,-11,-10,-9,-8,-7,-6")
    ap.add_argument("--error-budget", type=float, default=1e-3)
    ap.add_argument("--err-max", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=2500)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--max-rss-gb", type=float, default=14.5)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)
    cache_dir = Path(args.cache_dir)
    thresholds = np.array(parse_thresholds(args.thresholds), dtype=np.float64)

    print(f"[config] thresholds={','.join(f'{x:.2f}' for x in thresholds)}")
    print(f"[config] error_budget={args.error_budget}")

    corpora = {}
    for ds in args.datasets:
        corpora[ds] = fetch_text_cached(ds, cache_dir)
        print(f"[dataset] {ds}: chars={len(corpora[ds]):,}")

    per_run_rows: List[dict] = []
    for m in args.models:
        for ds in args.datasets:
            if rss_gb() > args.max_rss_gb:
                raise RuntimeError(f"RSS too high before run: {rss_gb():.2f} GB")
            rows = analyze_one(
                model_name=m,
                dataset_name=ds,
                text=corpora[ds],
                out_dir=out_dir,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.max_windows,
                min_query_index=args.min_query_index,
                thresholds=thresholds,
                err_max=args.err_max,
                nbins=args.nbins,
                device=args.device,
            )
            per_run_rows.extend(rows)

    global_rows = aggregate_global(per_run_rows, thresholds=list(thresholds), err_max=args.err_max, nbins=args.nbins)
    chosen_t, msg = choose_threshold(global_rows, budget=args.error_budget)

    per_run_csv = out_dir / "clipping_error_vs_t_per_run.csv"
    global_csv = out_dir / "clipping_error_vs_t_global.csv"
    fig_png = out_dir / "clipping_error_vs_t_quantiles.png"
    sel_txt = out_dir / "clipping_threshold_selection.txt"

    write_csv(
        per_run_csv,
        per_run_rows,
        fields=[
            "model",
            "dataset",
            "threshold_t",
            "rows",
            "windows",
            "seq_len",
            "stride",
            "min_query_index",
            "p50",
            "p90",
            "p95",
            "p99",
            "rss_gb_end",
            "hist_npz",
        ],
    )

    write_csv(global_csv, global_rows, fields=["threshold_t", "rows", "p50", "p90", "p95", "p99"])

    plot_quantiles(global_rows, fig_png)

    with sel_txt.open("w", encoding="utf-8") as f:
        f.write(f"error_budget={args.error_budget}\n")
        f.write(msg + "\n")
        if chosen_t is not None:
            f.write(f"selected_threshold_t={chosen_t:.2f}\n")
            f.write(f"selected_clipping_magnitude_N={-chosen_t:.2f}\n")

    print(f"[done] per_run_csv={per_run_csv}")
    print(f"[done] global_csv={global_csv}")
    print(f"[done] fig={fig_png}")
    print(f"[done] selection={sel_txt}")
    print(f"[done] {msg}")


if __name__ == "__main__":
    main()
