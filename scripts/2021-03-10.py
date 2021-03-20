#%%
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 144

#%%
filename = "../sampleimage/子衛星LED点灯.jpg"
img = Image.open(filename)
img = np.array(img.convert("L"))

#%%
# 元画像のヒストグラム
def draw_histgram(img):
    std = img.std()
    fig, ax = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={"hspace":0.15})
    ax[0].hist(img.ravel(), 256, color="gray")
    ax[0].axvline(np.mean(img), c="C0", label="m")
    ax[0].axvline(np.mean(img)+std, c="C1", label="m + σ")
    ax[0].axvline(np.mean(img)+2*std, c="C2", label="m + 2σ")
    ax[0].axvline(np.mean(img)+3*std, c="C3", label="m + 3σ")
    ax[0].axvline(np.mean(img)+4*std, c="C4", label="m + 4σ")
    ax[0].set(title="Histgram of grayscale image", ylabel="Counts")
    ax[0].legend()
    ax[1].hist(img.ravel(), 256, color="gray")
    ax[1].axvline(np.mean(img), c="C0", label="m")
    ax[1].axvline(np.mean(img)+std, c="C1", label="m + σ")
    ax[1].axvline(np.mean(img)+2*std, c="C2", label="m + 2σ")
    ax[1].axvline(np.mean(img)+3*std, c="C3", label="m + 3σ")
    ax[1].axvline(np.mean(img)+4*std, c="C4", label="m + 4σ")
    ax[1].set(xlabel="Brightness", ylabel="Counts (log)")
    ax[1].set_yscale('log')
    fig.savefig("../outputs/Histgram_std.png", bbox_inches="tight", pad_inches=0.1)
draw_histgram(img)
# %%
# 単純な平均値による閾値
def std_threshold(img, out="std_threshold.png"):
    std = img.std()
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("(a) Original image (grayscale)", loc="left")
    ax[1].imshow(np.where(img > np.mean(img)+2*std, 1, 0), cmap="gray")
    ax[1].set_title("(b) Binarized with m + 2σ", loc="left")
    ax[2].imshow(np.where(img > np.mean(img)+3*std, 1, 0), cmap="gray")
    ax[2].set_title("(c) Binarized with m + 3σ", loc="left")
    ax[3].imshow(np.where(img > np.mean(img)+4*std, 1, 0), cmap="gray")
    ax[3].set_title("(d) Binarized with m + 4σ", loc="left")
    fig.savefig("../outputs/"+out, bbox_inches="tight", pad_inches=0.1)
std_threshold(img, out="std_threshold_lighton.png")
# %%
# 点灯画像と消灯画像の比較
filename = "../sampleimage/子衛星LED点灯.jpg"
img1 = Image.open(filename)
img1 = np.array(img1.convert("L"))

filename2 = "../sampleimage/子衛星LED消灯.jpg"
img2 = Image.open(filename2)
img2 = np.array(img2.convert("L"))

def draw_histgram_2image(img1, img2):
    fig, ax = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={"hspace":0.15})
    ax[0].hist(img1.ravel(), 256, color="red", alpha=0.3, label="Light on")
    ax[0].hist(img2.ravel(), 256, color="blue", alpha=0.3, label="Light off")
    ax[0].set(title="Histgram of grayscale image [light on/off]", ylabel="Counts")
    ax[0].legend()
    ax[1].hist(img1.ravel(), 256, color="red", alpha=0.3, label="Light on")
    ax[1].hist(img2.ravel(), 256, color="blue", alpha=0.3, label="Light off")
    ax[1].set(ylabel="Counts (log)")
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.savefig("../outputs/Histgram_light_onoff.png", bbox_inches="tight", pad_inches=0.1)
draw_histgram_2image(img1, img2)
# %%
std_threshold(img2, out="std_threshold_lightoff.png")
# %%