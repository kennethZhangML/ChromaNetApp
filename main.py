import streamlit as st

import torchvision
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot 
from PIL import Image

from colorizationTools import *

st.title("Colorization Tool")

model = ColorizationAutoencoder()
model.load_state_dict(torch.load('landscapeAE.pt'))



