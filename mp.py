import multiprocessing
import torch


gpu_lock = multiprocessing.BoundedSemaphore(3)
hf_api_lock = multiprocessing.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
