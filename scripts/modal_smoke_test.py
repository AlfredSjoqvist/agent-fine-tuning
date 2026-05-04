"""Smoke test for Modal setup.

Confirms:
  - Container image builds with torch
  - GPU is visible to the container
  - Persistent volume `224project-data` mounts and accepts writes

Run from project root:
    modal run scripts/modal_smoke_test.py

Expected wall-clock: ~30s first run (image build), ~10s subsequent.
Expected cost on T4: well under $0.01.
"""

import modal

app = modal.App("cs224r-smoke-test")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1")

volume = modal.Volume.from_name("224project-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/output": volume},
    timeout=300,
)
def smoke_test() -> dict:
    import datetime
    import subprocess

    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["compute_capability"] = f"{props.major}.{props.minor}"

    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    touch_path = "/output/smoke_test_touch.txt"
    with open(touch_path, "w") as f:
        f.write(f"smoke test ok at {datetime.datetime.utcnow().isoformat()}Z\n")
    print(f"wrote {touch_path}")
    volume.commit()
    print("volume committed")
    return info


@app.local_entrypoint()
def main():
    result = smoke_test.remote()
    print("\nsmoke test result:", result)
