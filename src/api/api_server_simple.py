import os
import sys
import requests
import subprocess
import time

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
# We will use the Hub's CDN to download files directly to bypass HF login/cli issues
HF_BASE_URL = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def download_file(url, dest_path):
    """Downloads a file with progress bar."""
    print(f"Downloading {os.path.basename(dest_path)}...")
    
    # Stream download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            downloaded += len(chunk)
            
            # Simple progress bar
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            sys.stdout.write(f"\r{percent:.1f}% [{downloaded}/{total_size}]")
            sys.stdout.flush()
    
    print(f"\n✅ Downloaded: {dest_path}")

def ensure_model_exists():
    """
    Checks if model files exist. If not, downloads them manually.
    """
    # Construct expected local paths (HuggingFace cache structure)
    # Pattern: ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/commit_hash/
    # Simplified: We will just download to a known folder and let vLLM handle it, 
    # but vLLM expects HF cache structure. 
    # STRATEGY: Use `huggingface-cli download` with --local-dir if possible, 
    # but if that fails, we fallback to manual wget/gdown.
    
    print("Checking model cache...")
    
    # Check if standard HF CLI works first
    try:
        # Attempt to download using CLI (most reliable structure)
        # We pass --local-dir to force a local folder structure if cache is messy
        check_cmd = f"huggingface-cli download {MODEL_ID} --local-dir /tmp/model_cache --local-dir-use-symlinks False"
        result = subprocess.run(check_cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Model found or successfully downloaded via CLI.")
        return "/tmp/model_cache" # Return path for vLLM
    except subprocess.CalledProcessError:
        print("⚠️ CLI download failed. Attempting Manual HTTP Download...")
        
        # Fallback: Manual Download (Simplified for Mistral-7B)
        # Note: This requires knowing exact filenames. Mistral-7B uses safetensors.
        local_dir = "/tmp/mistral_manual"
        os.makedirs(local_dir, exist_ok=True)
        
        files_to_download = [
            "config.json",
            "tokenizer.model",
            "tokenizer.json",
            "generation_config.json"
        ]
        
        # Download config and tokenizer files
        for f in files_to_download:
            url = f"{HF_BASE_URL}/{f}"
            dest = os.path.join(local_dir, f)
            if not os.path.exists(dest):
                download_file(url, dest)
        
        # Note: We skip downloading the 15GB weights here in this script 
        # because it will timeout in this simple python script context if connection is slow.
        # We rely on vLLM's auto-download which handles resume better.
        # However, we ensured the small config files are there, which often triggers the rest.
        
        print("✅ Config files downloaded. vLLM will handle weights.")
        return local_dir

# --- Main Execution ---

def main():
    print("=== Compliance Guard vLLM Launcher ===")
    print("1. Ensuring Model Download...")
    model_path = ensure_model_exists()
    
    print(f"\n2. Starting vLLM Server...")
    print(f"   Model Path: {model_path}")
    print(f"   Loading LoRA Adapter: /home/ubuntu/compliance-guard-gpu/models/checkpoints")
    
    # Construct command
    # We pass the local path explicitly if we downloaded it, else just the name
    # Note: If we used the manual fallback, we point to the folder.
    # If we want to use the specific LoRA, we must use the base model name or path.
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "mistralai/Mistral-7B-Instruct-v0.1", # vLLM handles download resume better than simple script
        "--enable-lora",
        "--lora-modules", "compliance-trained=/home/ubuntu/compliance-guard-gpu/models/checkpoints",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    # Execute
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
