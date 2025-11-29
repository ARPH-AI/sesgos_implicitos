from typing import Optional

import json
import os
import random
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

from utils.llm_call import call_model, MODEL_VERSIONS
from utils.utils import read_prompt
from allocation.map_to_spanish import get_prompt_for_language

# Load config from JSON
with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    config = json.load(f)

EXPERIMENT_PATH = config["experiment_path"]
WOMAN_SEX_SPECIFIC = config["woman_sex_specific"]
MAN_SEX_SPECIFIC = config["man_sex_specific"]
TEMPS = config["temps"]
PROMPTS_TO_RUN = config["prompts_to_run"]
N_PAIRS = config["n_pairs"]
MODELS = config["models"]
LANGS = config["langs"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

load_dotenv()
random.seed(123)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def pair_key(p1, p2) -> str:
    return f"{p1['Name']}|{p2['Name']}"

def unordered_key(p1, p2) -> str:
    return "|".join(sorted([p1["Name"], p2["Name"]]))

def safe_write_json(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

def build_person(person: dict, age_mode: str, age_value: int = None) -> dict:
    """Build person dict based on age mode."""
    base = {
        "Name": person["Name"],
        "Sex assigned at birth": person["Sex assigned at birth"],
        "Gender identity": person["Gender identity"],
    }
    if age_mode == "same_age" and age_value is not None:
        base["Age"] = age_value
    elif age_mode not in ("no_age", "same_age"):
        return dict(person)  # Return original with all fields
    return base

def filter_people_by_scenario(people: list, prompt_name: str) -> list:
    """Filter people based on scenario requirements."""
    if prompt_name in WOMAN_SEX_SPECIFIC:
        return [p for p in people if "woman" in p["Sex assigned at birth"].lower()]
    elif prompt_name in MAN_SEX_SPECIFIC:
        return [p for p in people if "woman" not in p["Sex assigned at birth"].lower()]
    return people

def load_existing_results(path: str) -> tuple[list, set, set]:
    """Load existing results and return (results, seen_pairs, existing_unordered)."""
    if not os.path.exists(path):
        return [], set(), set()
    try:
        with open(path) as f:
            results = json.load(f)
        seen = set()
        unordered = set()
        for item in results:
            p1, p2 = item["person1"], item["person2"]
            seen.add(pair_key(p1, p2))
            seen.add(pair_key(p2, p1))
            unordered.add(unordered_key(p1, p2))
        return results, seen, unordered
    except Exception:
        return [], set(), set()

def run_pair(p1: dict, p2: dict, prompt: str, lang: str, age_mode: str, 
             model_name: str, temp: float, exp: bool, age_value: int = None) -> Optional[dict]:
    """Run a single pair through the model and return result dict."""
    p1_final = build_person(p1, age_mode, age_value)
    p2_final = build_person(p2, age_mode, age_value)
    
    try:
        formatted_prompt = get_prompt_for_language(p1_final, p2_final, prompt, lang, age_mode, exp=exp)
        response = call_model(
            prompt=[{"role": "user", "content": formatted_prompt}],
            model_name=MODEL_VERSIONS[model_name],
            temp=temp,
        )
        return {
            "prompt": formatted_prompt,
            "response": response,
            "temperature": temp,
            "model": model_name,
            "person1": p1_final,
            "person2": p2_final,
        }
    except Exception as e:
        logger.exception(f"Call failed for {pair_key(p1, p2)}: {e}")
        return None

def process_pair_both_orders(p1: dict, p2: dict, seen_pairs: set, results: list, 
                              out_path: str, progress_bar, age_mode: str, **run_kwargs) -> bool:
    """Process both orderings of a pair. Returns True if any new results added."""
    added = False
    # Generate age once so both orderings use the same age
    age_value = random.randint(18, 64) if age_mode == "same_age" else None
    
    for person1, person2 in [(p1, p2), (p2, p1)]:
        key = pair_key(person1, person2)
        if key in seen_pairs:
            progress_bar.update(1)
            continue
        seen_pairs.add(key)
        
        result = run_pair(person1, person2, age_mode=age_mode, age_value=age_value, **run_kwargs)
        if result:
            results.append(result)
            safe_write_json(out_path, results)
            added = True
        progress_bar.update(1)
    return added

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    df_people = pd.read_csv(f"{EXPERIMENT_PATH}/prompts/people.csv").to_dict(orient="records")
    cis_people = [p for p in df_people if "cis" in p["Gender identity"].lower()]
    trans_people = [p for p in df_people if "trans" in p["Gender identity"].lower()]

    total_iterations = len(TEMPS) * len(MODELS) * len(LANGS) * len(PROMPTS_TO_RUN) * N_PAIRS * 2
    progress_bar = tqdm(total=total_iterations, desc="Processing", unit="iteration")

    for exp in [False, True]:
        with_without = "with_exp" if exp else "without_exp"
        for age_mode in ["same_age"]:
            for temp in TEMPS:
                for prompt_name in PROMPTS_TO_RUN:
                    cis_filtered = filter_people_by_scenario(cis_people, prompt_name)
                    trans_filtered = filter_people_by_scenario(trans_people, prompt_name)
                    
                    for lang in LANGS:
                        for model_name in MODELS:
                            logger.info(f"Processing {prompt_name} in {lang} with {model_name} at temp {temp}")
                            
                            prompt = read_prompt(f"{EXPERIMENT_PATH}/prompts/{lang}/{prompt_name}.txt")
                            out_path = f"{EXPERIMENT_PATH}/answers/sex_gender/temp_{str(temp)[0]}/{lang}/{model_name}/{age_mode}/{with_without}/{prompt_name}.json"
                            print(out_path)
                            
                            results, seen_pairs, existing_unordered = load_existing_results(out_path)
                            
                            # Load reference pairs from without_exp baseline
                            ref_path = f"{EXPERIMENT_PATH}/answers/sex_gender/temp_{str(temp)[0]}/english/gpt-4o-mini/age/without_exp/{prompt_name}.json"
                            try:
                                with open(ref_path) as f:
                                    reference = json.load(f)
                            except FileNotFoundError:
                                reference = []
                                logger.warning(f"Reference file not found for {prompt_name}")
                            
                            reference_unordered = {unordered_key(dp["person1"], dp["person2"]) for dp in reference}
                            
                            run_kwargs = dict(prompt=prompt, lang=lang, age_mode=age_mode, 
                                            model_name=model_name, temp=temp, exp=exp)
                            
                            # Process reference pairs first
                            for dp in reference:
                                if len(existing_unordered) >= N_PAIRS:
                                    logger.info(f"Reached target of {N_PAIRS} pairs")
                                    break
                                
                                p1, p2 = dp["person1"], dp["person2"]
                                if unordered_key(p1, p2) in existing_unordered:
                                    continue
                                
                                process_pair_both_orders(p1, p2, seen_pairs, results, out_path, 
                                                        progress_bar, **run_kwargs)
                                existing_unordered.add(unordered_key(p1, p2))
                            
                            # Sample additional pairs if needed
                            max_attempts = 10000
                            attempts = 0
                            while len(existing_unordered) < N_PAIRS and attempts < max_attempts:
                                attempts += 1
                                p_cis = random.choice(cis_filtered)
                                p_trans = random.choice(trans_filtered)
                                ukey = unordered_key(p_cis, p_trans)
                                
                                if ukey in existing_unordered or ukey in reference_unordered or p_cis["Name"] == p_trans["Name"]:
                                    continue
                                
                                process_pair_both_orders(p_cis, p_trans, seen_pairs, results, 
                                                        out_path, progress_bar, **run_kwargs)
                                existing_unordered.add(ukey)
                            
                            if attempts >= max_attempts and len(existing_unordered) < N_PAIRS:
                                logger.warning(f"Stopped early after {attempts} attempts")

    progress_bar.close()

if __name__ == "__main__":
    main()