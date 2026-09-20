import os
import signal
import subprocess
import sys
from pathlib import Path
import yaml


def load_configs(config_dir: str = "config") -> tuple[dict, dict]:
    cfg_path = Path(config_dir)
    vllm_config: dict = {}
    frontend_config: dict = {}

    if cfg_path.exists():
        for yaml_file in sorted(cfg_path.glob("*.yaml")):
            with open(yaml_file) as f:
                file_config = yaml.safe_load(f) or {}

                if yaml_file.name == "frontend.yaml":
                    frontend_config.update(file_config)
                else:
                    vllm_config.update(file_config)

    # Apply environment variable overrides
    for config in [vllm_config, frontend_config]:
        for key, value in list(config.items()):
            env_key = key.upper()
            if env_key in os.environ:
                if isinstance(value, bool):
                    config[key] = os.environ[env_key].lower() == "true"
                elif isinstance(value, int):
                    config[key] = int(os.environ[env_key])
                else:
                    config[key] = os.environ[env_key]

    # Merge frontend config into vllm_config, excluding informational fields
    exclude_keys = {"api_base_url", "api_timeout"}
    for key, value in frontend_config.items():
        if key not in exclude_keys:
            vllm_config[key] = value

    return vllm_config, frontend_config


def build_vllm_cmd(config: dict) -> list[str]:
    cmd = ["vllm", "serve", config.pop("model")]

    for key, value in config.items():
        if value is None:
            continue

        cli_key = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{cli_key}")
        elif isinstance(value, list):
            for item in value:
                cmd.extend([f"--{cli_key}", str(item)])
        else:
            cmd.extend([f"--{cli_key}", str(value)])

    return cmd


def main():
    vllm_config, frontend_config = load_configs()
    cmd = build_vllm_cmd(vllm_config)

    if frontend_config:
        print(f"Frontend config loaded: {frontend_config}", flush=True)

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
