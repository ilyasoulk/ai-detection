from datasets import load_dataset

def count_avg_words(dataset_path):
    ds = load_dataset(dataset_path)

    word_count = 0
    for corpus in ds["train"]:
        txt = corpus["text"]
        list_text = txt.split(' ')
        word_count += len(list_text)

    ds_size = len(ds["train"])
    return word_count / ds_size


if __name__ == "__main__":
    ai_dataset_path = "HuggingFaceTB/cosmopedia-100k"
    human_dataset_path = "imdb"
    embedding_path = "multi-qa-mpnet-base-dot-v1"
    ai_avg_words = count_avg_words(ai_dataset_path)
    human_avg_words = count_avg_words(human_dataset_path)
    print(f"Average word per sample for the dataset: {ai_dataset_path} = {ai_avg_words}")
    print(f"Average word per sample for the dataset: {human_dataset_path} = {human_avg_words}")
