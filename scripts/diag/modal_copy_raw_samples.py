"""Copy a few raw ALFWorld game files from the Modal image to the persistent
volume so you can browse them via the Modal UI.

Picks 3 train games (Pick&Place, Heat, Clean) + 1 val game.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_copy_raw_samples.py
"""

import modal

app = modal.App("cs224r-copy-raw-samples")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .pip_install("alfworld[full]", "textworld", "pyyaml")
    .env({"ALFWORLD_DATA": "/root/alfworld_data"})
    .run_commands(
        "alfworld-download -f",
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


@app.function(image=image, volumes={"/output": volume}, timeout=300)
def copy_samples() -> dict:
    import os
    import shutil

    DATA = "/root/alfworld_data/json_2.1.1"
    OUT = "/output/raw_samples"
    os.makedirs(OUT, exist_ok=True)

    found = {"pick_and_place": None, "pick_heat": None, "pick_clean": None, "val": None}

    # Walk train/ for one of each task type
    for root, dirs, files in os.walk(f"{DATA}/train"):
        if "traj_data.json" not in files:
            continue
        bucket = None
        if "pick_and_place_simple" in root and not found["pick_and_place"]:
            bucket = "pick_and_place"
        elif "pick_heat_then_place" in root and not found["pick_heat"]:
            bucket = "pick_heat"
        elif "pick_clean_then_place" in root and not found["pick_clean"]:
            bucket = "pick_clean"
        if bucket and not found[bucket]:
            found[bucket] = root
        if all(found[k] for k in ("pick_and_place", "pick_heat", "pick_clean")):
            break

    # Pick one val game (from valid_unseen)
    for root, dirs, files in os.walk(f"{DATA}/valid_unseen"):
        if "traj_data.json" in files:
            found["val"] = root
            break

    print(f"Found sample game folders:")
    summary = {}
    for bucket, src in found.items():
        if not src:
            continue
        rel = os.path.relpath(src, DATA)
        dst = f"{OUT}/{bucket}__{rel.replace('/', '__')}"
        os.makedirs(dst, exist_ok=True)
        # Copy the two key files
        for fname in ["traj_data.json", "game.tw-pddl"]:
            src_path = os.path.join(src, fname)
            if os.path.exists(src_path):
                size = os.path.getsize(src_path)
                shutil.copy2(src_path, os.path.join(dst, fname))
                summary[f"{bucket}/{fname}"] = f"{size:,} bytes"
        print(f"  {bucket}: {rel}")

    volume.commit()
    return {"out_dir": OUT, "files": summary}


@app.local_entrypoint()
def main():
    print(copy_samples.remote())
