# This file contains the evaluation for finetuned pretrained models
from transformers import pipeline
from datasets import load_dataset
from utils.utils import split_dataset_random, transform_paired_dataset, evaluate
from tqdm import tqdm
import argparse


def pred_test_set(model_id, dataset_path):
    pipe = pipeline("text-classification", model=model_id)
    ds = load_dataset(dataset_path, split="train")

    ds = split_dataset_random(ds)

    test_set = ds["validation"]
    test_set = transform_paired_dataset(test_set)

    preds = []
    labels = []
    skip = 0
    for data, label in tqdm(zip(test_set["text"], test_set["class"])):
        try:
            pred = pipe(data)
            preds.append(1 if pred[0]["label"] == "Fake" else 0)
            labels.append(label)
        except:
            skip += 1

    return preds, labels


if __name__ == "__main__":
    model_id = "openai-community/roberta-base-openai-detector"
    parser = argparse.ArgumentParser(description="Load config for dataset processing")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset or huggingface id",
    )

    args = parser.parse_args()

    predictions, gt = pred_test_set(model_id, args.dataset_path)
    evaluate(gt, predictions)
