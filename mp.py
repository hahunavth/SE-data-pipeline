from multiprocessing import Lock

import torch


gpu_lock = Lock()
hf_api_lock = Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
