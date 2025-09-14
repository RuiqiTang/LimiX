import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from utils.utils import  download_datset, download_model

model_file = download_model(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", save_path="./cache")