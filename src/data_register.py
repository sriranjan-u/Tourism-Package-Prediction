from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

print("Start with initializing API Client")

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Sriranjan/Tourism-Package-Pred"
repo_type = "dataset"   

#data_folder = "/content/Tourism-Package-Pred/data"

# Local file path (from  project)
local_file_path = "data/raw/tourism.csv"

# Desired path INSIDE Hugging Face dataset repo
path_in_repo = "data/raw/tourism.csv"

print(f"Uploading {local_file_path} to HF dataset repo at {path_in_repo}")

api.upload_file(
    path_or_fileobj=local_file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Add tourism.csv under data/raw"
)

print("Data successfully uploaded to Hugging Face Dataset")
