import json
import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import random
import google.generativeai as genai

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.llm_call import MODEL_VERSIONS, call_model

random.seed(123)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

with open(os.path.join(SCRIPT_DIR, 'config.json'), 'r') as file:
    config = json.load(file)

results = []

df_words = pd.read_csv(os.path.join(SCRIPT_DIR, "prompts/labeled_words.csv"))

OPTION_WORDS_DICT = {
    "english": {
        "with_none": "cisgender, transgender or none",
        "without_none": "cisgender or transgender",
    },
    "spanish": {
        "with_none": "cisgénero, transgénero o ninguno",
        "without_none": "cisgénero o transgénero",
    }
}

df_words["id"] = range(len(df_words))
positive_ids = df_words[df_words.sentiment == "positive"]["id"].tolist()
negative_ids = df_words[df_words.sentiment == "negative"]["id"].tolist()
neutral_ids = df_words[df_words.sentiment == "neutral"]["id"].tolist()

results_path = os.path.join(SCRIPT_DIR, "results/results.json")
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = []

done = {
    (
        r["formatted_prompt"],
        r["model"],
        r["with_or_without_none"],
    )
    for r in results
}

# Load generated prompts
generated_prompts_path = os.path.join(SCRIPT_DIR, "prompts/generated_prompts.json")
with open(generated_prompts_path, 'r') as f:
    generated_prompts = json.load(f)

progress_bar = tqdm(total=(len(generated_prompts) * len(config["models"])), desc="Processing", unit="iteration")

for model in config["models"]:
    print(f"Starting model: {model}")
    for entry in generated_prompts:
        prompt_id = entry["prompt_id"]
        words = entry["words"]
        lang = entry["language"]
        temp = entry["temperature"]
        model_name = model
        with_or_without = entry["with_or_without_none"]
        shuffled_options = entry["shuffled_options"]
        
        formatted_prompt = entry["formatted_prompt"]
        sig = (
            formatted_prompt,
            model_name,
            with_or_without,
        )

        if sig in done:
            # already done: skip but advance progress
            progress_bar.update(1)
            continue

        try:
            response = call_model(
                prompt=[
                    {
                        "role": "user", 
                        "content": formatted_prompt
                    }
                ], 
                model_name=MODEL_VERSIONS[model_name], 
                temp=temp
            )
            print("ok")
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60/15) 
            print("sleep")
            continue
                        
        progress_bar.update(1)
                    
        result_item = {
            "prompt_id": prompt_id,
            "words": words,
            "formatted_prompt": formatted_prompt,
            "response": response,
            "temperature": temp,
            "model": model_name,
            "language": lang,
            "with_or_without_none": with_or_without,
            "shuffled_options": shuffled_options
        }
        done.add(sig)
        results.append(result_item)

        with open(results_path, "w") as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)

progress_bar.close()
