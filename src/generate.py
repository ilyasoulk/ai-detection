from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import random
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_prompt(truncated_text):
    messages = [
        {"role": "system", "content": (
            "You are an article completion assistant. You MUST ALWAYS format your response with [ARTICLE] and [/ARTICLE] tokens. "
            "Your completion should be natural and coherent with the given text."
            "Your completion should be detailled enough to be a daily news article"
        )},
        {"role": "user", "content": (
            "Complete the following article. Your response MUST:\n"
            "1. Start with [ARTICLE]\n"
            "2. End with [/ARTICLE]\n"
            "3. Include the given text at the start\n"
            "4. Detail the article as much as needed\n"
            "5. Complete it naturally\n\n"
            "Here's the text to complete:\n"
            "[ARTICLE]\n"
            f"{truncated_text}\n"
            "[/ARTICLE]"
        )},
        {"role": "assistant", "content": (
            "I understand that I must:\n"
            "- Keep the [ARTICLE] and [/ARTICLE] tokens\n"
            "- Include the original text\n"
            "- Detail the article as much as needed\n"
            "- Complete it naturally\n"
            "I will now provide the completion."
        )},
        {"role": "user", "content": "Please provide the completion now:"}
    ]
    return messages

def parse_article_content(text):
    start_token = "[ARTICLE]"
    end_token = "[/ARTICLE]"
    
    start_index = text.find(start_token) + len(start_token)
    end_index = text.find(end_token)
    
    if start_index == -1 or end_index == -1:
        return ""
        
    return text[start_index:end_index].strip()


def send_openai(message, model="gpt-3.5-turbo-0125", max_tokens=800, temperature=1):
    response = client.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )
    return response.choices[0].message.content


def random_truncate(text, min_n=30, max_n=50):
    n = random.randint(min_n, max_n)
    if n >= len(text):
        return text
    return text[:n]

def process_article(article):
    prefix = random_truncate(article)
    prompt = generate_prompt(prefix)
    response = send_openai(prompt)
    return {
        "ai": parse_article_content(response),
        "human": article
    }

def generate(dataset_path="abisee/cnn_dailymail", filepath="data/synthetic_reporter.json"):
    ds = load_dataset(dataset_path, '3.0.0', split='train')
    articles = ds["article"]

    # create file if does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_article, article): article for article in articles}
        for future in tqdm(as_completed(futures), total=len(articles), desc="Generating the dataset..."):
            try:
                sample = future.result()
                data.append(sample)
            except Exception as e:
                print(f"Error processing article: {e}")

            # Save periodically
            if len(data) % 10 == 0:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)

    # Final save
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=openai_key)
    generate()
