#!/usr/bin/env python3
"""Empirical L extraction with raw vs effective definitions (16GB-safe).

Definitions per attention row:
- Raw:      L_raw = max(log p) - min(log p)
- Effective: keep top keys until cumulative mass >= alpha (default 0.999),
             L_eff = max(log p_kept) - min(log p_kept)

Outputs:
- L_effective_summary.csv
- L_raw_vs_effective_cdf.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Tuple

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
    return psutil.Process().memory_info().rss / (1024 ** 3)


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
    def __init__(self, lmax: float, nbins: int):
        self.edges = np.linspace(0.0, lmax, nbins + 1, dtype=np.float64)
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

    def cdf_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        c = np.cumsum(self.counts)
        y = c / c[-1] if c[-1] > 0 else np.zeros_like(c, dtype=np.float64)
        x = self.edges[1:]
        return x, y


def iter_windows(token_ids, seq_len: int, stride: int, max_windows: int):
    n = len(token_ids)
    done = 0
    for s in range(0, n - seq_len + 1, stride):
        yield token_ids[s : s + seq_len]
        done += 1
        if done >= max_windows:
            break


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
    lmax: float,
    nbins: int,
    alpha: float,
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

    h_raw = Hist(lmax=lmax, nbins=nbins)
    h_eff = Hist(lmax=lmax, nbins=nbins)

    rows = 0
    eps = 1e-30
    windows_done = 0

    with torch.no_grad():
        for w_ids in iter_windows(token_ids, seq_len=seq_len, stride=stride, max_windows=max_windows):
            x = torch.tensor(w_ids, dtype=torch.long, device=device).unsqueeze(0)
            out = model(x, output_attentions=True, use_cache=False)

            for att in out.attentions:
                a = att[0].float().cpu().numpy()  # [H,T,T]
                H, T, _ = a.shape
                for t in range(min_query_index, T):
                    p = a[:, t, : t + 1]

                    logp = np.log(np.clip(p, eps, 1.0))
                    L_raw = logp.max(axis=1) - logp.min(axis=1)
                    h_raw.add(L_raw)

                    # effective range using top cumulative mass alpha
                    L_eff = np.zeros(H, dtype=np.float64)
                    for h in range(H):
                        ph = p[h]
                        idx = np.argsort(ph)[::-1]
                        ps = ph[idx]
                        cs = np.cumsum(ps)
                        k = int(np.searchsorted(cs, alpha, side="left") + 1)
                        kept = ps[: max(1, k)]
                        L_eff[h] = float(np.log(np.clip(kept.max(), eps, 1.0)) - np.log(np.clip(kept.min(), eps, 1.0)))
                    h_eff.add(L_eff)

                    rows += H

            windows_done += 1
            if windows_done % 20 == 0:
                print(f"  windows={windows_done}/{max_windows} rows={rows} rss={rss_gb():.2f} GB")

            del out, x

    q = [0.50, 0.90, 0.95, 0.99]
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "rows": rows,
        "windows": windows_done,
        "seq_len": seq_len,
        "stride": stride,
        "min_query_index": min_query_index,
        "alpha": alpha,
        "rss_gb_end": rss_gb(),
    }
    for p in q:
        result[f"raw_p{int(p*100)}"] = h_raw.quantile(p)
        result[f"eff_p{int(p*100)}"] = h_eff.quantile(p)

    model_slug = slugify(model_name)
    ds_slug = slugify(dataset_name)
    np.savez_compressed(out_dir / f"hist_raw__{model_slug}__{ds_slug}.npz", edges=h_raw.edges, counts=h_raw.counts)
    np.savez_compressed(out_dir / f"hist_eff__{model_slug}__{ds_slug}.npz", edges=h_eff.edges, counts=h_eff.counts)

    print(
        f"[done] rows={rows} raw_p99={result['raw_p99']:.2f} eff_p99={result['eff_p99']:.2f} rss={result['rss_gb_end']:.2f} GB"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def load_hist(npz_path: Path):
    d = np.load(npz_path)
    return d["edges"], d["counts"]


def make_cdf_plot(results, out_dir: Path):
    plt.figure(figsize=(7.2, 5.0))
    for r in results:
        ms, ds = slugify(r["model"]), slugify(r["dataset"])
        for kind, ls in [("raw", "--"), ("eff", "-")]:
            edges, counts = load_hist(out_dir / f"hist_{kind}__{ms}__{ds}.npz")
            c = np.cumsum(counts)
            y = c / c[-1]
            x = edges[1:]
            label = f"{r['model']}|{r['dataset']} ({kind})"
            plt.plot(x, y, lw=1.7, ls=ls, label=label)
    plt.xlabel("L")
    plt.ylabel("Empirical CDF")
    plt.title("Raw vs effective logit-range distributions")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p = out_dir / "L_raw_vs_effective_cdf.png"
    plt.savefig(p, dpi=240)
    plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache-dir", default="/Users/bot_s/.openclaw/workspace/research/2026-02-19-mrr-exponential/data/logit_corpora")
    ap.add_argument("--models", nargs="+", default=["distilgpt2", "gpt2"])
    ap.add_argument("--datasets", nargs="+", default=["tiny_shakespeare", "pride_prejudice"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=120)
    ap.add_argument("--min-query-index", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.999)
    ap.add_argument("--lmax", type=float, default=120.0)
    ap.add_argument("--nbins", type=int, default=1200)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--max-rss-gb", type=float, default=14.5)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)
    cache_dir = Path(args.cache_dir)

    corpora = {}
    for ds in args.datasets:
        corpora[ds] = fetch_text_cached(ds, cache_dir)
        print(f"[dataset] {ds}: chars={len(corpora[ds]):,}")

    results = []
    for m in args.models:
        for ds in args.datasets:
            if rss_gb() > args.max_rss_gb:
                raise RuntimeError(f"RSS too high before run: {rss_gb():.2f} GB")
            res = analyze_one(
                model_name=m,
                dataset_name=ds,
                text=corpora[ds],
                out_dir=out_dir,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.max_windows,
                min_query_index=args.min_query_index,
                lmax=args.lmax,
                nbins=args.nbins,
                alpha=args.alpha,
                device=args.device,
            )
            results.append(res)

    cdf_png = make_cdf_plot(results, out_dir)

    summary_csv = out_dir / "L_effective_summary.csv"
    fields = [
        "model",
        "dataset",
        "rows",
        "windows",
        "seq_len",
        "stride",
        "min_query_index",
        "alpha",
        "raw_p50",
        "raw_p90",
        "raw_p95",
        "raw_p99",
        "eff_p50",
        "eff_p90",
        "eff_p95",
        "eff_p99",
        "rss_gb_end",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"\n[done] summary={summary_csv}")
    print(f"[done] cdf={cdf_png}")


if __name__ == "__main__":
    main()
