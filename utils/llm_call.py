import google.generativeai as genai
from openai import OpenAI
import os
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user
from replicate import Client as ReplicateClient


load_dotenv()

MODEL_VERSIONS = {
    "o3": "o3-2025-04-16",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-001",
    "grok-4": "grok-4-0709",
    "grok-3": "grok-3",
    "grok-3-mini": "grok-3-mini",
}

def call_model(
    prompt: list[str],
    model_name: str,
    temp: float = 0.0,
    i: int = 0,
    reasoning_effort: str = "none",
    response_model = None,
):
    if "gemini" in model_name:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(model_name=model_name)
        generation_config = genai.GenerationConfig(temperature=temp)
        full_response = gemini_model.generate_content(
            contents=prompt[0]["content"],
            generation_config=generation_config
        )
        return full_response.text

    elif "gpt" in model_name or "o3" in model_name:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if response_model:
            response = client.chat.completions.parse(
                model=model_name,
                messages=prompt,
                response_format=response_model,
            )
            return response.choices[0].message.parsed

        if "o3" in model_name:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                reasoning_effort=reasoning_effort,
            )
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=temp,
            )


        return completion.choices[0].message.content
    elif "grok" in model_name:
        client = Client(api_key=os.getenv("XAI_API_KEY"))

        chat = client.chat.create(model=model_name, temperature=0)
        chat.append(user(prompt[0]["content"]))

        response = chat.sample()
        return response.content
    elif "meta/meta-llama-3-70b-instruct" == model_name:
        client = ReplicateClient(api_token=os.getenv("REPLICATE_API_KEY"))  
        input = {
            "prompt": prompt[0]["content"],
            "max_tokens": 1024
        }

        output = client.run(
            model_name,
            input=input,
        )

        return "".join(output)    
    else:
        raise ValueError(f"Model {model_name} not supported.")