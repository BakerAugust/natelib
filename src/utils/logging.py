"""
Useful logging snippet from
https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L09/code/mlp-softmax-pyscripts/train_model.py
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logpath = "training.log"
logger.addHandler(logging.FileHandler(logpath, "a"))
print = logger.info
