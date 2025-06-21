# ask_llm_direct.py
import requests

def ask_llm(prompt, token):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta" 
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 512
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    # Make it robust
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "error" in result:
        return f"Error from LLM: {result['error']}"
    else:
        return str(result)
