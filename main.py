import os
import signal
import subprocess
import sys

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen2.5-7b")
PORT = os.environ.get("PORT", "8000")
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.83")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
MAX_NUM_SEQS = os.environ.get("MAX_NUM_SEQS", "32")


def main():
    cmd = [
        "vllm", "serve", MODEL,
        "--quantization", "awq_marlin",
        "--dtype", "float16",
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-model-len", MAX_MODEL_LEN,
        "--tensor-parallel-size", "1",
        "--max-num-seqs", MAX_NUM_SEQS,
        "--swap-space", "0",
        "--host", "0.0.0.0",
        "--port", PORT,
        "--served-model-name", SERVED_NAME,
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--max-log-len", "100",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)

    def _shutdown(sig, _frame):
        print(f"\nReceived signal {sig}, shutting down…", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
