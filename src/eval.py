import glob
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
import logging
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from opt import get_options
from utils import (clear_and_create_folder, Logger)
from models.nerf import NeRFSystem
class Evaluator(object):
    pass