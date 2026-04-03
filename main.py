import os
import shutil
from train import train
from utils import load_config
# from train_MOS import train

args = load_config("./config.yaml")
os.makedirs(args.outdir, exist_ok=True)
shutil.copy("./config.yaml", args.outdir)
train(args)
# inference(args)