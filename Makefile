generate_SmolLM-360M_dataset:
	python3 /home/yanel/epita/ai-detection/src/dataset/generate.py --config ./configs/SmolLM2-360M-Instruct.yaml

generate_SmolLM2-1.7B_dataset:
	python3 /home/yanel/epita/ai-detection/src/dataset/generate.py --config ./configs/SmolLM2-1.7B-Instruct.yaml

generate_Phi-3.5_dataset:
	python3 /home/yanel/epita/ai-detection/src/dataset/generate.py --config ./configs/Phi-3.5-mini-instruct.yaml

generate_Qwen2.5-1.5B_dataset:
	python3 /home/yanel/epita/ai-detection/src/dataset/generate.py --config ./configs/Qwen2.5-1.5B-Instruct.yaml

push_dataset_to_hf:
	python3 /home/yanel/epita/ai-detection/src/dataset/push_hf.py
