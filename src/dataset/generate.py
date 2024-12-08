from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
import random
import os
import json
import yaml


def generate_prompts(article: str, nb_chars: int):
    with open(config["prompt_path"], "r") as f:
        messages = json.load(f)

    assert messages[0]["role"] == "system"

    for message in messages:
        if "content" in message and isinstance(message["content"], str):
            message["content"] = message["content"].format(
                article=article, nb_chars=nb_chars
            )
            if config["prepend_system_role_to_user"]:
                message["content"] = messages[0]["content"] + "\n" + message["content"]
    if config["prepend_system_role_to_user"]:
        return messages[1:]
    return messages


def random_split(text) -> tuple[str, str, int]:
    min_n = len(text) // 4
    max_n = len(text) // 2
    n = random.randint(min_n, max_n)
    if n >= len(text):
        return text
    prefix = text[:n]
    suffix = text[n:]
    return prefix, suffix


def process_articles(articles):
    prefixes, suffixes = zip(*[random_split(article) for article in articles])
    messages = [
        generate_prompts(prefix, len(suffix))
        for prefix, suffix in zip(prefixes, suffixes)
    ]
    answers = generate_batch_answer(messages)
    return [
        {"ai": answer, "human": suffix} for answer, suffix in zip(answers, suffixes)
    ]


def get_dataloader(dataset_path: str, batch_size: int = 32):
    ds = load_dataset(path=dataset_path, name="3.0.0", split="train")
    ds.set_format(type="torch", columns=["article"])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader


def generate_dataset():
    dataloader = get_dataloader(
        dataset_path=config["dataset_path"], batch_size=config["batch_size"]
    )
    datas = []
    for batch in dataloader:
        articles = batch["article"]
        ai_human_articles = process_articles(articles)
        datas.extend(ai_human_articles)
        if config["one_batch"]:
            break
    directory = f'{config["directory"]}/{config["model"]}'
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/synthetic_reporter.json", "w") as f:
        json.dump(datas, f)
    with open(f"{directory}/config.yaml", "w") as f:
        yaml.dump(config, f)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_and_load_config():
    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="abisee/cnn_dailymail",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--directory", type=str, default="data", help="Directory to save outputs"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/complete_articles.json",
        help="Path to the prompt file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset_path"] = args.dataset_path
    config["directory"] = args.directory
    config["prompt_path"] = args.prompt_path

    return config


if __name__ == "__main__":
    config = parse_and_load_config()
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

    llm = LLM(model=config["model"])
    sampling_params = SamplingParams(**config["sampling_params"])

    def generate_batch_answer(
        messages: list[list[ChatCompletionMessageParam]],
    ):
        res = llm.chat(messages=messages, sampling_params=sampling_params)
        return [res.outputs[-1].text for res in res]

    generate_dataset()
