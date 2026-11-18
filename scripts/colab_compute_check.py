"""Run a bounded, game-free HKRL validation on a Colab GPU.

Upload benchmark_compute.py to /content/benchmark_compute.py first.
This script installs only into the temporary runtime and creates synthetic data.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import torch

REVISION = "44a9dd8d311510e7e29e35e69e38c6ad2fd204d4"
ROOT = Path("/content/hkrl-validation")
OUT = ROOT / "runs/validation"


def run(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, cwd=ROOT, check=True, timeout=600, **kwargs)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires the requested GPU")
    print("GPU", torch.cuda.get_device_name(0), "TORCH", torch.__version__, flush=True)
    if not ROOT.exists():
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/myouo/hk_Rl.git", str(ROOT)],
            check=True,
            timeout=120,
        )
    run(["git", "checkout", "--detach", REVISION])
    OUT.mkdir(parents=True, exist_ok=True)
    hardware = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "upstream_revision": REVISION,
        "game_tested": False,
    }
    (OUT / "colab-hardware.json").write_text(json.dumps(hardware, indent=2))
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "./python[dev]"])
    archive = OUT / "flatc.zip"
    urllib.request.urlretrieve(
        "https://github.com/google/flatbuffers/releases/download/v25.2.10/"
        "Linux.flatc.binary.clang%2B%2B-18.zip",
        archive,
    )
    with zipfile.ZipFile(archive) as z:
        data = z.read(next(name for name in z.namelist() if Path(name).name == "flatc"))
    flatc = OUT / "flatc"
    flatc.write_bytes(data)
    flatc.chmod(0o755)
    run([str(flatc), "--python", "-o", "python/hkrl/schema", "schema/hkrl.fbs"])
    shutil.copy2("/content/benchmark_compute.py", ROOT / "scripts/benchmark_compute.py")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CUDA_VISIBLE_DEVICES="",
    )
    with (OUT / "linux-tests.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "-m",
                "not integration",
                "--maxfail=8",
                "--junitxml=../runs/validation/linux-tests.xml",
            ],
            check=False,
            cwd=ROOT / "python",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    print("LINUX_TEST_EXIT", result.returncode, flush=True)
    run(
        [
            sys.executable,
            "scripts/benchmark_compute.py",
            "--device",
            "cuda",
            "--output",
            "runs/validation/compute_cuda.json",
        ]
    )
    # Use one worker / one batch, disable compilation for this bounded acceptance run.
    import yaml

    cfg = {
        "defaults": [str(ROOT / "configs/train/remote_learner.yaml")],
        "learner": {
            "device": "cuda",
            "batches_per_update": 1,
            "compile_mode": "off",
            "amp_dtype": "auto",
        },
    }
    config = OUT / "colab_smoke.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run(
        [
            sys.executable,
            "scripts/run_phase8_smoke.py",
            "--config",
            str(config),
            "--tasks",
            "configs/tasks/mantis_lords.yaml",
            "--num-workers",
            "1",
            "--synthetic-train-update",
            "--work-dir",
            "runs/validation/colab-smoke",
            "--output",
            "runs/validation/colab-smoke.json",
        ]
    )
    (OUT / "colab-status.json").write_text(
        json.dumps(
            {
                "completed": True,
                "linux_test_exit": result.returncode,
                "synthetic_only": True,
            }
        )
    )
    with tarfile.open("/content/hkrl-validation-results.tar.gz", "w:gz") as tar:
        for p in OUT.rglob("*"):
            if p.is_file() and p.name not in {"flatc", "flatc.zip"}:
                tar.add(p, arcname=str(p.relative_to(OUT)))
    print("COMPLETE /content/hkrl-validation-results.tar.gz", flush=True)


if __name__ == "__main__":
    main()
