import os, random, json, time
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class Timer:
    def __init__(self): self.t0=time.time()
    def lap(self, msg=""): 
        t=time.time()-self.t0
        print(f"{msg} {t:.1f}s"); return t

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def json_dump(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
