from typing import Optional, Dict
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from numpy.random import default_rng

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Paths ----------
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
WORDS_PATH = os.path.join(SCRIPT_DIR, "prompts/words_with_categories.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results/results.json")
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis_out")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Load config + words ----------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

df_words = pd.read_csv(WORDS_PATH)

def _get_col(name):
    """Find column case-insensitively."""
    for c in df_words.columns:
        if c.lower() == name:
            return c
    raise ValueError(f"Column '{name}' not found. Present: {list(df_words.columns)}")

COL_EN = _get_col("english")
COL_ES = _get_col("spanish")
COL_SENT = _get_col("sentiment")
COL_CAT = _get_col("category")

# ---------- Utilities ----------
def p_adjust_bh(p):
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    q = p[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1.0)
    return out

def exact_binom_p_two_sided(k, n, p0=0.5):
    """Two-sided exact binomial p-value."""
    if n == 0:
        return np.nan
    def pmf(x):
        return comb(n, x) * (p0**x) * ((1-p0)**(n-x))
    obs = pmf(k)
    return min(sum(pmf(x) for x in range(n+1) if pmf(x) <= obs + 1e-15), 1.0)

def bootstrap_ci_mean_binary(vals, B=4000, seed=1234):
    """Bootstrap 95% CI for binary data."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)

# ---------- Parsing helpers ----------
def _canon_label(raw: str, language: str) -> Optional[str]:
    """Canonicalize label to 'cis', 'trans', or 'none'."""
    if raw is None:
        return None
    s = raw.strip().lower()
    if language == "spanish":
        if s in {"cisgénero", "cisgenero", "cis"}:
            return "cis"
        if s in {"transgénero", "transgenero", "trans"}:
            return "trans"
        if s in {"ninguno", "ninguna", "ambos"}:
            return "none"
    else:  # english
        if s in {"cisgender", "cis"}:
            return "cis"
        if s in {"transgender", "trans"}:
            return "trans"
        if s in {"none", "neither", "both"}:
            return "none"
    return None

def _labels_pattern(entry) -> re.Pattern:
    """Build regex pattern for label matching."""
    opts = entry.get("shuffled_options", []) or []
    if not opts:
        alt = r"transgénero|cisgénero|ninguno|transgender|cisgender|none|both"
    else:
        core = [re.escape(o.strip()) for o in opts if isinstance(o, str)]
        core.append("both")
        alt = "|".join(core)
    return re.compile(rf"({alt})", flags=re.IGNORECASE)

def _line_regex(word: str, label_pat: re.Pattern) -> re.Pattern:
    """Regex for matching word-label pairs on a line."""
    prefix = r"(?:\s*(?:[\-\*\u2022•]|\d{1,3}[.)]?)\s*[:\-–—.]?\s*)?"
    sep = r"(?::|\-|\–|—|,|\s{1,})"
    open_wrap = r"(?:[\(\[\{]\s*)?"
    close_wrap = r"(?:\s*[\)\]\}])?"
    trail = r"[\s\.,;:!\?-–—]*"
    return re.compile(
        rf"^\s*{prefix}{re.escape(word)}\s*{sep}\s*{open_wrap}{label_pat.pattern}{close_wrap}{trail}$",
        flags=re.IGNORECASE,
    )

def _line_parts_regex(label_pat: re.Pattern) -> re.Pattern:
    """Generic line parser for word-label pairs."""
    sep = r"(?::|\-|\–|—|,|\s{1,})"
    open_wrap = r"(?:[\(\[\{]\s*)?"
    close_wrap = r"(?:\s*[\)\]\}])?"
    trail = r"[\s\.,;:!\?-–—]*"
    return re.compile(
        rf"^\s*(.*?)\s*{sep}\s*{open_wrap}({label_pat.pattern}){close_wrap}{trail}$",
        flags=re.IGNORECASE,
    )

def _multi_pairs_regex(label_pat: re.Pattern) -> re.Pattern:
    """Find multiple (word, label) pairs within a single line."""
    label = f"({label_pat.pattern})"
    pat = rf"\s*(?:(?P<w1>[^,(){{}}\[\]]+?)\s*\(\s*(?P<lab1>{label})\s*\)|(?P<w2>[^,(){{}}\[\]]+?)\s*,\s*(?P<lab2>{label}))\s*"
    return re.compile(pat, flags=re.IGNORECASE)

def parse_llm_answers(entries, language: str):
    """Parse LLM responses and extract word-label pairs."""
    rows = []

    for entry in entries:
        if entry.get("language") != language:
            continue
        txt = (entry.get("response") or "").strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        label_pat = _labels_pattern(entry)
        wow = entry.get("with_or_without_none", "")
        model = entry.get("model", "")
        temp = entry.get("temperature", None)
        pid = entry.get("prompt_id", None)

        # Pre-parse lines into candidate -> canonical label map
        parts_re = _line_parts_regex(label_pat)
        multi_re = _multi_pairs_regex(label_pat)
        cand2label = {}  # Dict[str, str]
        
        for ln in lines:
            m = parts_re.match(ln)
            if m:
                cand = (m.group(1) or "").strip().lower()
                raw_lab = m.group(2)
                canon = _canon_label(raw_lab, language)
                if canon is not None:
                    cand2label[cand] = canon
            # Also scan for multiple pairs within one line
            pos = 0
            while True:
                mm = multi_re.search(ln, pos)
                if not mm:
                    break
                w = (mm.group('w1') or mm.group('w2') or '').strip().lower()
                lab = mm.group('lab1') or mm.group('lab2')
                canon = _canon_label(lab, language)
                if w and canon is not None:
                    cand2label[w] = canon
                pos = mm.end()

        for word in entry.get("words", []):
            found_label = None
            line_pat = _line_regex(word, label_pat)
            for ln in lines:
                m = line_pat.match(ln)
                if m:
                    found_label = m.group(1)
                    break

            if found_label is None:
                found_label = cand2label.get(word.strip().lower())

            canon = _canon_label(found_label, language)
            if canon is None:
                if wow == "with_none":
                    canon = "none"
                else:
                    continue

            rows.append((model, temp, wow, pid, language, word, canon))

    out = pd.DataFrame(rows, columns=[
        "model", "temperature", "with_or_without_none", "prompt_id",
        "language", "word", "label"
    ])
    return out

def per_word_stats(g):
    """Compute per-word bias statistics."""
    rows = []
    for word, gg in g.groupby("word"):
        gg_bin = gg[gg["has_binary"]]
        n = len(gg_bin)
        k = int(gg_bin["is_trans"].sum())
        p = (k/n) if n > 0 else np.nan
        lo, hi = bootstrap_ci_mean_binary(gg_bin["is_trans"].values, B=4000, seed=777)
        pval = exact_binom_p_two_sided(k, n, p0=0.5) if n > 0 else np.nan
        rows.append({
            "word": word, "n": n, "k_trans": k,
            "bias_score": p, "ci_lo": lo, "ci_hi": hi, "p_binom": pval
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_binom"] = p_adjust_bh(out["p_binom"].fillna(1.0).values)
    return out


if __name__ == "__main__":
    # ---------- Load results and parse ----------
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    models = config.get("models", None) or sorted(set(e.get("model", "") for e in results))
    temperatures = config.get("temperatures", None) or sorted(set(e.get("temperature", 0.0) for e in results))
    languages = config.get("languages", ["english"])
    with_none_opts = config.get("with_or_without_none", ["without_none"])
    
    print(f"Models: {models}")
    print(f"Temperatures: {temperatures}")
    print(f"Languages: {languages}")
    print(f"With/without none: {with_none_opts}")

    # Prepare word metadata
    meta = df_words[[COL_EN, COL_ES, COL_SENT, COL_CAT]].copy()
    meta.columns = ["english", "spanish", "sentiment", "category"]

    # Collect tidy rows across all slices
    all_rows = []
    for model in models:
        for temp in temperatures:
            for wow in with_none_opts:
                subset = [e for e in results if e.get("model") == model
                          and e.get("temperature") == temp
                          and e.get("with_or_without_none") == wow]
                if not subset:
                    continue
                for lang in languages:
                    parsed = parse_llm_answers(subset, language=lang)
                    if parsed.empty:
                        print(f"No parsed results for {model}/{temp}/{wow}/{lang}")
                        continue
                    parsed["model"] = model
                    parsed["temperature"] = temp
                    parsed["with_or_without_none"] = wow
                    parsed["language"] = lang
                    all_rows.append(parsed)

    if not all_rows:
        raise RuntimeError("No rows parsed. Check RESULTS_PATH / filters.")

    df_long = pd.concat(all_rows, ignore_index=True)

    # Attach sentiment/category - handle both languages
    df_spanish = df_long[df_long["language"] == "spanish"].merge(
        df_words.rename(columns={COL_ES: "word"})[["word", COL_SENT, COL_CAT]],
        on="word", how="left"
    ).rename(columns={COL_SENT: "sentiment", COL_CAT: "category"})

    df_english = df_long[df_long["language"] == "english"].merge(
        df_words.rename(columns={COL_EN: "word"})[["word", COL_SENT, COL_CAT]],
        on="word", how="left"
    ).rename(columns={COL_SENT: "sentiment", COL_CAT: "category"})

    df_long = pd.concat([df_spanish, df_english], ignore_index=True)

    # Clean label
    df_long["label"] = df_long["label"].astype(str)
    df_long["is_trans"] = (df_long["label"] == "trans").astype(int)
    df_long["is_cis"] = (df_long["label"] == "cis").astype(int)
    df_long["has_binary"] = df_long["label"].isin(["cis", "trans"])

    # ---------- Per-word stats ----------
    pw_all = []
    for keys, grp in df_long.groupby(["model", "language", "with_or_without_none"]):
        pw = per_word_stats(grp)
        if pw.empty:
            continue
        model, lang, wow = keys
        pw["model"], pw["language"], pw["with_or_without_none"] = keys
        
        # Add sentiment/category from df_words - match on correct language column
        word_col = COL_ES if lang == "spanish" else COL_EN
        pw = pw.merge(
            df_words.rename(columns={word_col: "word"})[["word", COL_SENT, COL_CAT]],
            on="word", how="left"
        ).rename(columns={COL_SENT: "sentiment", COL_CAT: "category"})
        pw_all.append(pw)

    per_word = pd.concat(pw_all, ignore_index=True) if pw_all else pd.DataFrame()
    per_word_out = os.path.join(OUT_DIR, "per_word_bias_scores.csv")
    per_word.to_csv(per_word_out, index=False)

    # ---------- Cell means: (model × language × sentiment × category) ----------
    cell_rows = []
    for keys, grp in df_long.groupby(["model", "language", "with_or_without_none", "sentiment", "category"]):
        gbin = grp[grp["has_binary"]]
        n = len(gbin)
        k = int(gbin["is_trans"].sum())
        p = (k/n) if n > 0 else np.nan
        lo, hi = bootstrap_ci_mean_binary(gbin["is_trans"].values, B=4000, seed=123)
        pval = exact_binom_p_two_sided(k, n, p0=0.5) if n > 0 else np.nan
        model, lang, wow, sent, cat = keys
        cell_rows.append({
            "model": model, "language": lang, "with_or_without_none": wow,
            "sentiment": sent, "category": cat, "n": n, "k_trans": k,
            "prob_trans": p, "ci_lo": lo, "ci_hi": hi, "p_vs_half": pval
        })
    cells = pd.DataFrame(cell_rows)

    # BH–FDR per (model, language, with/without)
    if not cells.empty:
        cells["q_vs_half"] = (cells
            .groupby(["model", "language", "with_or_without_none"])["p_vs_half"]
            .transform(lambda s: p_adjust_bh(s.fillna(1.0).values))
        )
    cells_out = os.path.join(OUT_DIR, "cell_means_model_lang_sent_cat.csv")
    cells.to_csv(cells_out, index=False)

    # ---------- Save tidy long table ----------
    df_long_out = os.path.join(OUT_DIR, "parsed_long_rows.csv")
    df_long.to_csv(df_long_out, index=False)

    print("Done.")
    print(f"- Per-word stats: {per_word_out}")
    print(f"- Cell means:     {cells_out}")
    print(f"- Parsed rows:    {df_long_out}")
    print(f"- Figures in:     {FIG_DIR}")
