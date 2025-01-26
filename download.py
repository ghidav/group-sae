import json
import os
from huggingface_hub import snapshot_download
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--act_fn", type=str, required=True)
parser.add_argument("--layer", type=str, required=True)
parser.add_argument("--token", type=str, required=True)
args = parser.parse_args()

os.environ["HF_TOKEN"] = args.token

# Define the repository ID and target folder
repo_id = "mech-interp/baselines-jr-target-l0-pythia-160m-deduped"
subfolder = "layers.9"
local_save_path = "saes/pythia-160pm-deduped/jr/baseline/9"

# Download the entire repo snapshot temporarily
repo_path = snapshot_download(repo_id=repo_id, allow_patterns=f"{subfolder}/*")

# Copy only the desired subfolder to the local directory
subfolder_path = os.path.join(repo_path, subfolder)
if os.path.exists(subfolder_path):
    shutil.copytree(subfolder_path, local_save_path, dirs_exist_ok=True)
    print(f"Folder '{subfolder}' downloaded to '{local_save_path}'.")
else:
    print(f"Subfolder '{subfolder}' not found in the repository.")