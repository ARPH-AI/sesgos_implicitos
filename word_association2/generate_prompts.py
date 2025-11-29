from collections import Counter, defaultdict
import json
import os
import random
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import heapq

random.seed(123)
load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_prompt(file_path):
    """Read and return the contents of a text file."""
    with open(file_path, 'r') as file:
        return file.read().strip()

# Load configuration
with open(os.path.join(SCRIPT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

PROMPT_IDS = ["1", "2", "3"]

OPTION_WORDS_DICT = {
    "english": {
        "with_none": ["cisgender", "transgender", "none"],
        "without_none": ["cisgender", "transgender"],
    },
    "spanish": {
        "with_none": ["cisgénero", "transgénero", "ninguno"],
        "without_none": ["cisgénero", "transgénero"],
    }
}

def join_options(options: list[str], lang: str) -> str:
    if len(options) == 2:
        return f"{options[0]} o {options[1]}" if lang == "spanish" else f"{options[0]} or {options[1]}"
    else:
        if lang == "spanish":
            return f"{options[0]}, {options[1]} o {options[2]}"
        else:
            return f"{options[0]}, {options[1]} or {options[2]}"

# Number of times each word should appear
WORDS_PER_PROMPT = int(config.get("words_per_prompt", 16))

# -------------------- Load words --------------------
df_words = pd.read_csv(os.path.join(SCRIPT_DIR, "prompts/labeled_words.csv"))

# Ensure an 'id' column exists and is stable
if "id" not in df_words.columns:
    df_words = df_words.reset_index(drop=True)
    df_words["id"] = df_words.index

word_ids: list[int] = df_words["id"].tolist()
n_uses = int(config["n_uses_per_word"])

# Map id -> word text for each language once
id_to_word_by_lang: dict[str, dict[int, str]] = {
    lang: df_words.set_index("id")[lang].to_dict()
    for lang in config["languages"]
}

def make_prompt_bundles(word_ids, n_uses, words_per_prompt):
    """
    Returns a list of prompts (each a list of word IDs) such that:
      - each word ID appears exactly n_uses times in total,
      - no prompt contains duplicates,
      - prompt sizes differ by at most 1 (≈ words_per_prompt), with no tiny last group.
    """
    W = len(word_ids)
    assert 1 <= words_per_prompt <= W, "words_per_prompt must be between 1 and number of unique words."

    total = W * n_uses
    prompt_count = max(1, round(total / words_per_prompt))
    base = total // prompt_count
    rem = total % prompt_count
    group_sizes = [base + 1] * rem + [base] * (prompt_count - rem)
    assert max(group_sizes) <= W, (
        f"Group size {max(group_sizes)} exceeds number of unique words {W}. "
        "Lower words_per_prompt or increase vocabulary."
    )

    # Max-heap by remaining quota; (-count, wid) so largest count comes first
    heap = [(-n_uses, wid) for wid in word_ids]
    heapq.heapify(heap)

    bundles = []
    for gsize in group_sizes:
        used = set()
        selected = []
        buffer = []

        for _ in range(gsize):
            found = False
            while heap:
                neg_cnt, wid = heapq.heappop(heap)
                cnt = -neg_cnt
                if wid in used or cnt == 0:
                    buffer.append((neg_cnt, wid))
                    continue

                # take this word for this prompt
                used.add(wid)
                selected.append(wid)
                cnt -= 1
                if cnt > 0:
                    buffer.append((-cnt, wid))
                found = True
                break

            if not found:
                raise RuntimeError(
                    "Not enough distinct words with remaining quota to fill prompt without duplicates. "
                    "Reduce words_per_prompt or add more words."
                )

        # push back everything we popped this round
        for item in buffer:
            heapq.heappush(heap, item)

        bundles.append(selected)

    # Verify exact usage
    counts = defaultdict(int)
    for b in bundles:
        for wid in b:
            counts[wid] += 1
    assert all(counts[wid] == n_uses for wid in word_ids), "Usage counts mismatch."

    return bundles


if __name__ == "__main__":
    prompt_bundles = make_prompt_bundles(word_ids, n_uses, WORDS_PER_PROMPT)
    
    # Safety checks: exact counts and no duplicates in any prompt
    flat = [wid for bundle in prompt_bundles for wid in bundle]
    counts = Counter(flat)
    assert all(counts.get(wid, 0) == n_uses for wid in word_ids), "Usage counts do not match n_uses_per_word."
    for i, bundle in enumerate(prompt_bundles):
        assert len(bundle) == len(set(bundle)), f"Duplicate IDs in prompt {i}: {bundle}"

    # -------------------- Expand to full prompt variants --------------------
    results = []

    combos = (
        len(config["languages"])
        * len(config["models"])
        * len(config["temperatures"])
        * len(config["with_or_without_none"])
    )

    pbar = tqdm(total=len(prompt_bundles) * combos, desc="Generating prompts", unit="prompt")

    for bundle in prompt_bundles:
        for with_or_without in config["with_or_without_none"]:
            for lang in config["languages"]:
                # Prepare language-specific words for this prompt; randomize order per-variant
                words_seq = [id_to_word_by_lang[lang][wid] for wid in bundle]

                for model_name in config["models"]:
                    for temp in config["temperatures"]:
                        # Randomize elements per variant for maximal variability and reproducibility under SEED
                        random.shuffle(words_seq)  # order of words in the formatted prompt
                        prompt_id = random.choice(PROMPT_IDS)

                        opts = OPTION_WORDS_DICT[lang][with_or_without]
                        shuffled_opts = random.sample(opts, k=len(opts))
                        joined_opts = join_options(shuffled_opts, lang)

                        template_path = os.path.join(SCRIPT_DIR, f"prompts/{lang}/{prompt_id}.txt")
                        prompt_template = read_prompt(template_path)

                        formatted = prompt_template.format(
                            word_list=", ".join(words_seq),
                            len=len(words_seq),
                            len_labels=len(shuffled_opts),
                            option_words=joined_opts,
                        )

                        results.append({
                            "prompt_id": prompt_id,
                            "word_ids": bundle,              # canonical IDs used in this prompt
                            "words": list(words_seq),        # order used in the formatted text
                            "shuffled_options": shuffled_opts,
                            "formatted_prompt": formatted,
                            "temperature": temp,
                            "model": model_name,
                            "language": lang,
                            "with_or_without_none": with_or_without,
                        })
                        pbar.update(1)

    pbar.close()

    # -------------------- Persist --------------------
    out_path = os.path.join(SCRIPT_DIR, "prompts/generated_prompts.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4, ensure_ascii=False)

    print(
        f"Generated {len(results)} prompt variants from {len(prompt_bundles)} base prompts. "
        f"Each of {len(word_ids)} words appears exactly {n_uses} times across base prompts."
    )
