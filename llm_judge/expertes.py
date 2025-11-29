import json
from tqdm import tqdm
from utils.llm_call import call_model, MODEL_VERSIONS
from utils.utils import read_prompt
from dotenv import load_dotenv
import pandas as pd
import random
import os
import logging

# Load config from JSON
with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    config = json.load(f)

LLM_JUDGE_PATH = config["llm_judge_path"]
INPUT_CSV = config["input_csv"]
TEMP = config["temp"]
LANGS = config["langs"]
JUDGE_MODEL = config["judge_model"]
RUBRIC_FILTER = config["rubric_filter"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.propagate = False

load_dotenv()
random.seed(123)

results = []

df_people = pd.read_csv(INPUT_CSV).to_dict(orient="records")


def safe_write_json(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

def format_prompt(rubric:str, og_prompt: str, response: str) -> str:
    base_prompt = read_prompt(f"{LLM_JUDGE_PATH}/prompts/llm_judge_desempeno.txt")
    formatted_prompt = base_prompt.format(
        rubric=rubric,
        prompt=og_prompt,
        response=response,
    )
    return formatted_prompt

rubrics = []
rubrics_folder = f"{LLM_JUDGE_PATH}/prompts/rubrics"
for filename in os.listdir(rubrics_folder):
    if RUBRIC_FILTER and RUBRIC_FILTER not in filename:
        continue
    if filename.endswith(".txt"):
        with open(os.path.join(rubrics_folder, filename), "r") as f:
            content = f.read()
        rubrics.append((filename, content))

total_iterations = len(df_people) * len(rubrics)
progress_bar = tqdm(total=total_iterations, desc="Processing", unit="iteration")


out_path = f"{LLM_JUDGE_PATH}/answers_{JUDGE_MODEL}.json"
    
existing_pairs = set()

if os.path.exists(out_path):
    try:
        with open(out_path, "r") as f:
            existing_results = json.load(f)
        for item in existing_results:
            existing_pairs.add((item["og_prompt"], item["model"], item["rubric"]))
        results = existing_results
    except Exception:
        results = []
else:
    results = []

for item in df_people:
    for filename, rubric in rubrics:
        question = item["question"]
        model_name = item["model"]
        model_answer = item["answer"]
        

        if (question, model_name, filename) in existing_pairs:
            progress_bar.update(1)
            continue

        formatted_prompt = format_prompt(rubric, question, model_answer)

        response = call_model(
            prompt=[{"role": "user", "content": formatted_prompt}],
            model_name=MODEL_VERSIONS[JUDGE_MODEL],
            temp=TEMP,
            reasoning_effort="high",
        )

        existing_pairs.add((question, model_name, filename))


        result_item = {
            "og_prompt": question,
            "og_response": model_answer,
            "judge_prompt": formatted_prompt,
            "judge_response": response,
            "model": model_name,
            "rubric": filename,
            "rubric_definition": rubric,
        }
        results.append(result_item)
        safe_write_json(out_path, results)
        progress_bar.update(1)


progress_bar.close()

