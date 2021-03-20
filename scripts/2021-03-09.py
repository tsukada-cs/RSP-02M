#%%
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

#%%
filename = "../sampleimage/子衛星LED点灯.jpg"
img = Image.open(filename)
img = np.array(img.convert("L"))

plt.rcParams["figure.dpi"] = 144
#%%
# 元画像のヒストグラム
def dray_histgram(img):
    fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={"hspace":0.35})
    ax[0].hist(img.ravel(), 256, color="gray")
    ax[0].axvline(np.mean(img), c="C0", label="average")
    ax[0].axvline(np.mean(img)*2, c="C1", label="average*2")
    ax[0].axvline(np.mean(img)*3, c="C2", label="average*3")
    ax[0].set(title="Histgram of gray image", xlabel="Brightness", ylabel="Counts")
    ax[0].legend()
    ax[1].hist(img.ravel(), 256, color="gray")
    ax[1].axvline(np.mean(img), c="C0", label="average")
    ax[1].axvline(np.mean(img)*2, c="C1", label="average*2")
    ax[1].axvline(np.mean(img)*3, c="C2", label="average*3")
    ax[1].set(xlabel="Brightness", ylabel="Counts (log)")
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.savefig("../outputs/Histgram.png", bbox_inches="tight", pad_inches=0.1)
# dray_histgram(img)
# %%
# 単純な平均値による閾値
def mean_threshold(img):
    fig, ax = plt.subplots(2, 2, figsize=(8,7))
    ax[0,0].imshow(img, cmap="gray")
    ax[0,0].set(title="Original image (grayscale)")
    ax[0,1].imshow(np.where(img > np.mean(img), 1, 0), cmap="gray")
    ax[0,1].set(title="Binarized with average")
    ax[1,0].imshow(np.where(img > np.mean(img)*2, 1, 0), cmap="gray")
    ax[1,0].set(title="Binarized with average * 2")
    ax[1,1].imshow(np.where(img > np.mean(img)*3, 1, 0), cmap="gray")
    ax[1,1].set(title="Binarized with average * 3")
    fig.savefig("../outputs/mean_threshold.png", bbox_inches="tight", pad_inches=0.1)
# mean_threshold(img)