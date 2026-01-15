import os
import sys

# Set the environment variable
os.environ["HOG_DIR"] = "/data/dataset/HOGraspNet"

sys.path.append(".")
from scripts.HOG_dataloader import HOGDataset


setup = "s5"
split = "train"
db_path = os.path.join(os.environ["HOG_DIR"], "data")
dataloader = HOGDataset(setup, split, db_path)

a = 1
