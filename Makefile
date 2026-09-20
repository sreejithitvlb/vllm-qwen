inference:
	mkdir -p logs ~/hf-cache
	docker build -t vllm-qwen .
	docker run --rm --runtime=nvidia --shm-size=8g \
		-v ~/hf-cache:/workspace/.cache/huggingface \
		-v $(CURDIR)/config:/app/config \
		-p 8000:8000 \
		vllm-qwen \
		2>&1 | tee logs/vllm_$(shell date +%Y%m%d_%H%M%S).log
