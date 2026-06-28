inference:
	mkdir -p logs
	docker run --rm --runtime=nvidia \
		-v ~/projects/vllm-qwen:/app \
		-v ~/hf-cache:/root/.cache/huggingface \
		-e HF_HOME=/root/.cache/huggingface \
		-p 8000:8000 -w /app dustynv/vllm:r36.4.0 python3 main.py \
		2>&1 | tee logs/vllm_$(shell date +%Y%m%d_%H%M%S).log
