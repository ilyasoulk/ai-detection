generate_SmolLM-360M_dataset:
	python3 ./src/dataset/generate.py --config ./configs/SmolLM2-360M-Instruct.yaml

generate_SmolLM2-1.7B_dataset:
	python3 ./src/dataset/generate.py --config ./configs/SmolLM2-1.7B-Instruct.yaml

generate_Phi-3.5_dataset:
	python3 ./src/dataset/generate.py --config ./configs/Phi-3.5-mini-instruct.yaml

generate_Qwen2.5-1.5B_dataset:
	python3 ./src/dataset/generate.py --config ./configs/Qwen2.5-1.5B-Instruct.yaml

generate_Llama-3.2-1B_dataset:
	python3 ./src/dataset/generate.py --config ./configs/Llama-3.2-1B-Instruct.yaml

generate_gemma-2-2b-it_dataset:
	python3 ./src/dataset/generate.py --config ./configs/gemma-2-2b-it.yaml

generate_Llama-3.1-8B_dataset:
	python3 ./src/dataset/generate.py --config ./configs/Llama-3.1-8B-Instruct.yaml

push_dataset_to_hf:
	python3 ./src/dataset/push_hf.py


DATASETS=\
    ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct-CNN\
    zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct\
    zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct \
    zcamz/ai-vs-human-google-gemma-2-2b-it \
    zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct

SUBSET_SIZE=512

EMBEDDING_MODEL=google-bert/bert-base-uncased

compute_all_phd:
	@mkdir -p ./logs/phd/zcamz
	@mkdir -p ./logs/phd/ilyasoulk
	@for dataset in $(DATASETS); do \
		python ./src/intrinsic_dim.py \
			--dataset_path=$$dataset \
			--embedding_model=$(EMBEDDING_MODEL) \
			--intrinsic_dim_method="phd" > ./logs/phd/$$dataset.log 2>&1 \
			--subset_size=$(SUBSET_SIZE); \
	done

compute_all_pca:
	@mkdir -p ./logs/pca/zcamz
	@mkdir -p ./logs/pca/ilyasoulk
	@for dataset in $(DATASETS); do \
		python ./src/intrinsic_dim.py \
			--dataset_path=$$dataset \
			--embedding_model=$(EMBEDDING_MODEL) \
			--intrinsic_dim_method="pca" > ./logs/pca/$$dataset.log 2>&1 \
			--subset_size=$(SUBSET_SIZE); \
	done

compute_all_phd_pca: compute_all_phd compute_all_pca


