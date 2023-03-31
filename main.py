import streamlit as st

import torchvision
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from colorizationTools import *
import warnings
warnings.filterwarnings('ignore')


st.title("Colorization AE Tool")

AEmodel = ColorizationAutoencoder()
AEmodel.load_state_dict(torch.load('landscapeAE.pt'))


def create_colorized_pred(model, img_path):
    convert_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize((160, 160)),
                                         transforms.Grayscale()])
    converted_img = convert_tensor(img_path)
    print(converted_img.shape)

    pred = model.forward(converted_img.float().view(1, 1, 160, 160))
    model_pred = torch.cat((converted_img.view(1, 160, 160), pred[0].cpu()), dim = 0)
    model_pred_inv = model_pred.permute(1, 2, 0) * torch.tensor([100, 255, 255]) - torch.tensor([0, 128, 128])
    rgb_pred = lab2rgb(model_pred_inv.detach().numpy())
    return rgb_pred

uploaded_img = st.file_uploader("Upload an Image ['jpg', 'jpeg', 'png']")

if uploaded_img is not None:
    uploaded_img = Image.open(uploaded_img)
    pred = create_colorized_pred(AEmodel, uploaded_img)

    fig = plt.figure(figsize = (10, 5))
    plt.subplot(221)
    st.image(uploaded_img)
    plt.title("Grayscale Image")
    plt.subplot(222)
    st.image(pred, use_column_width = True)
    plt.title("AE Colorized Image")
    plt.show()

