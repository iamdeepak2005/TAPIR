"""
Phase 0: Download llama.cpp binary and GGUF model files.
- Resumes partial downloads with curl -C -
- Queries Hugging Face API for exact file sizes and LFS sha256
- Verifies checksums and logs to data/raw/download_manifest.json
"""
import os
import sys
import json
import hashlib
import shutil
import datetime
import subprocess
import zipfile
import urllib.request

BASE = r"c:\Users\deepa\Research-TAPIR\tapir"
BIN_DIR = os.path.join(BASE, "bin")
MODEL_DIR = os.path.join(BASE, "models")
RAW_DIR = os.path.join(BASE, "data", "raw")

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

LLAMA_CPP_BUILD = "b12399"
LLAMA_CPP_SEMVER = "v1.0.0"
LLAMA_CPP_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}/llama-{LLAMA_CPP_BUILD}-bin-win-cpu-x64.zip"

HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILES = {
    "Q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
    "Q5_K_M": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
    "Q4_K_M": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_hf_metadata(repo_id):
    """Fetch model sibling info from Hugging Face API."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "TAPIR-Research-Pipeline"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            siblings = data.get("siblings", [])
            meta = {}
            for s in siblings:
                rfilename = s.get("rfilename", "")
                lfs = s.get("lfs", {})
                size = lfs.get("size", s.get("size", None))
                oid = lfs.get("oid", None)
                meta[rfilename] = {"size": size, "sha256": oid}
            return meta
    except Exception as e:
        print(f"Warning: Could not fetch HF metadata via API: {e}")
        return {}


def download_with_resume(url, dest, expected_size=None, description="file"):
    print(f"\nDownloading/Checking {description}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    if os.path.exists(dest) and expected_size is not None:
        cur_size = os.path.getsize(dest)
        if cur_size == expected_size:
            print(f"  File completely matches expected size ({cur_size / (1024*1024):.1f} MB). Skipping download.")
            return True
        elif cur_size > expected_size:
            print(f"  File is larger ({cur_size}) than expected ({expected_size}). Removing and redownloading.")
            os.remove(dest)
        else:
            print(f"  Partial download detected ({cur_size / (1024*1024):.1f} / {expected_size / (1024*1024):.1f} MB). Resuming...")

    # curl with resume support: -C -
    cmd = ["curl.exe", "-L", "-C", "-", "-o", dest, "--progress-bar", "--fail", url]
    try:
        res = subprocess.run(cmd, timeout=900)
        if res.returncode != 0:
            print(f"  curl failed with exit code {res.returncode}")
            return False
        return os.path.exists(dest) and os.path.getsize(dest) > 1000
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def main():
    print("=" * 60)
    print("TAPIR Phase 0: Download Dependencies & Models")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print("=" * 60)

    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "downloads": {},
        "status": "in_progress",
    }

    usage = shutil.disk_usage("C:\\")
    free_gb = usage.free / (1024**3)
    print(f"Disk space: {free_gb:.1f} GB free")
    manifest["disk_free_gb_before"] = round(free_gb, 2)

    # 1. llama.cpp check/install
    print("\n--- llama.cpp ---")
    extract_dir = os.path.join(BIN_DIR, "llama-cpp")
    cli_path = os.path.join(extract_dir, "llama-cli.exe")
    bench_path = os.path.join(extract_dir, "llama-bench.exe")
    zip_path = os.path.join(BIN_DIR, f"llama-{LLAMA_CPP_BUILD}-bin-win-cpu-x64.zip")

    if os.path.exists(cli_path) and os.path.exists(bench_path):
        print(f"  llama.cpp binaries already extracted at {extract_dir}")
        zip_sha = sha256_file(zip_path) if os.path.exists(zip_path) else "pre-existing"
        manifest["downloads"]["llama_cpp"] = {
            "build": LLAMA_CPP_BUILD,
            "semver": LLAMA_CPP_SEMVER,
            "url": LLAMA_CPP_URL,
            "file": zip_path,
            "size_mb": round(os.path.getsize(zip_path) / (1024**2), 1) if os.path.exists(zip_path) else 0,
            "sha256": zip_sha,
            "cli_path": cli_path,
            "bench_path": bench_path,
            "status": "downloaded",
        }
    else:
        success = download_with_resume(LLAMA_CPP_URL, zip_path, description=f"llama.cpp {LLAMA_CPP_BUILD}")
        if success and os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            manifest["downloads"]["llama_cpp"] = {
                "build": LLAMA_CPP_BUILD,
                "semver": LLAMA_CPP_SEMVER,
                "url": LLAMA_CPP_URL,
                "file": zip_path,
                "size_mb": round(os.path.getsize(zip_path) / (1024**2), 1),
                "sha256": sha256_file(zip_path),
                "cli_path": cli_path,
                "bench_path": bench_path,
                "status": "downloaded",
            }
        else:
            manifest["downloads"]["llama_cpp"] = {"status": "failed"}

    # 2. Fetch HF repo metadata
    print("\n--- Querying Hugging Face for model metadata ---")
    hf_meta = get_hf_metadata(HF_REPO)
    for fname, info in hf_meta.items():
        if fname in GGUF_FILES.values():
            print(f"  {fname}: size={info.get('size')} bytes, sha256={info.get('sha256')}")

    # 3. Download GGUF Models
    print("\n--- Downloading GGUF Models ---")
    for quant, filename in GGUF_FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
        expected_size = hf_meta.get(filename, {}).get("size")
        expected_sha = hf_meta.get(filename, {}).get("sha256")

        success = download_with_resume(url, dest, expected_size=expected_size, description=f"{quant} ({filename})")
        if success and os.path.exists(dest):
            cur_size = os.path.getsize(dest)
            if expected_size and cur_size != expected_size:
                print(f"  Warning: {quant} size mismatch ({cur_size} vs {expected_size})")
            print(f"  Calculating SHA256 for {filename}...")
            actual_sha = sha256_file(dest)
            print(f"  Actual SHA256: {actual_sha}")
            if expected_sha and actual_sha != expected_sha:
                print(f"  WARNING: SHA256 mismatch with Hugging Face LFS OID ({expected_sha})")

            manifest["downloads"][f"gguf_{quant}"] = {
                "model": "Qwen2.5-1.5B-Instruct",
                "quantization": quant,
                "repo": HF_REPO,
                "filename": filename,
                "local_path": dest,
                "size_bytes": cur_size,
                "size_mb": round(cur_size / (1024**2), 1),
                "sha256": actual_sha,
                "expected_sha256": expected_sha,
                "status": "downloaded",
            }
        else:
            manifest["downloads"][f"gguf_{quant}"] = {
                "quantization": quant,
                "status": "failed",
            }

    usage = shutil.disk_usage("C:\\")
    free_after = usage.free / (1024**3)
    manifest["disk_free_gb_after"] = round(free_after, 2)
    manifest["disk_used_gb"] = round(manifest["disk_free_gb_before"] - free_after, 2)

    all_ok = all(d.get("status") == "downloaded" for d in manifest["downloads"].values())
    manifest["status"] = "complete" if all_ok else "partial"

    manifest_path = os.path.join(RAW_DIR, "download_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Manifest written: {manifest_path}")
    print(f"Overall status: {manifest['status']}")
    print(f"Disk space used: {manifest['disk_used_gb']} GB (Free remaining: {manifest['disk_free_gb_after']} GB)")
    print("=" * 60)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
