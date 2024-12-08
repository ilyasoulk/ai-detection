from datasets import Dataset
from huggingface_hub import login, HfApi
import yaml
import argparse
import json


def push_to_hub(
    data_path: str,
    token: str = None,
    hf_id: str = None,
    author: str = None,
    private: bool = False,
):
    login(token)

    with open(f"{data_path}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"{data_path}/synthetic_reporter.json", "r") as f:
        data = json.load(f)

    repo_id = f"{hf_id}/ai-vs-human-{config['model'].replace('/', '-')}"

    dataset = Dataset.from_list(data)
    dataset.info.description = "AI vs Human dataset on the CNN Daily mails"
    dataset.info.license = "MIT"
    dataset.info.homepage = f"https://huggingface.co/datasets/{repo_id}"
    dataset.info.citation = f"""
    @misc{{synthetic_reporter}},
        author = {{{author}}},
        title = {{Synthetic Reporter Dataset}},
        year = {2024},
        publisher = {{Hugging Face}}
    """

    # Push to hub with README
    readme_content = f"""---
license: mit
task_categories:
- text-classification
- text-generation
language:
- en
pretty_name: AI vs Human CNN Daily News
size_categories:
- 1K<n<10K
---
# AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)

## Dataset Description
This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
Each article was randomly truncated between 25% and 50% of its length. 
The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.

## Data Fields
- 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
- 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.

## Model and Sampling Parameters
The model used to generate the AI completions was {config["model"]}.

The sampling parameters used were:
{config["sampling_params"]}

## License
MIT License
    """
    dataset.push_to_hub(repo_id, private=private)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the dataset folder containing synthetic_reporter.json and config.yaml",
    )
    parser.add_argument("--author", type=str, help="Author of the dataset")
    parser.add_argument(
        "--private", action="store_true", help="Set the repository to private"
    )
    parser.add_argument("--hf_id", type=str, help="Hugging Face ID")
    parser.add_argument("--token", type=str, help="Hugging Face token")

    args = parser.parse_args()

    if not args.data_path:
        args.data_path = input("Enter the path to the dataset folder: ")
    if not args.author:
        args.author = input("Enter the author of the dataset: ")
    if not args.hf_id:
        args.hf_id = input("Enter the Hugging Face ID: ")
    if not args.token:
        args.token = input("Enter the Hugging Face token: ")

    push_to_hub(
        data_path=args.data_path,
        author=args.author,
        private=args.private,
        token=args.token,
        hf_id=args.hf_id,
    )
