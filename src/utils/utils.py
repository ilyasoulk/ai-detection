def split_dataset_random(dataset, seed=42):
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    return {"train": split["train"], "validation": split["test"]}


def transform_paired_dataset(dataset):
    """
    Transform a dataset with 'ai' and 'human' columns into a format with 'text' and 'class' columns
    Returns a dataset with twice as many rows, where:
    - 'text' contains all texts (both AI and human)
    - 'class' contains 1 for AI-generated text and 0 for human-written text
    """
    texts = []
    labels = []

    # Add AI texts with label 1
    texts.extend(dataset["ai"])
    labels.extend([1] * len(dataset["ai"]))

    # Add human texts with label 0
    texts.extend(dataset["human"])
    labels.extend([0] * len(dataset["human"]))

    return {"text": texts, "class": labels}
