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
img = np.array(img.convert("RGB")) # R, G, B の順番

red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]
#%%
# RGB図
def draw_RGB2d(red, green, blue):
    fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={"wspace":0.01})
    ax[0].imshow(red, vmin=0, vmax=255, cmap="Reds")
    ax[0].set(title="(a) Red")
    ax[1].imshow(green, vmin=0, vmax=255, cmap="Greens")
    ax[1].set(title="(b) Green")
    ax[2].imshow(blue, vmin=0, vmax=255, cmap="Blues")
    ax[2].set(title="(c) Blue")
    for _ax in ax:
        _ax.axis("off")
    fig.savefig("../outputs/RGB2d.png", bbox_inches="tight", pad_inches=0.1)
draw_RGB2d(red, green, blue)

#%%
# 差分RGB図
def draw_RGB2d_diff(red, green, blue):
    fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={"wspace":0.01})
    ax[0].imshow(red-blue, vmin=0, vmax=255, cmap="Reds")
    ax[0].set(title="(a) Red - Blue")
    ax[1].imshow(red-green, vmin=0, vmax=255, cmap="Reds")
    ax[1].set(title="(b) Red - Green")
    ax[2].imshow(red-(blue+green)/2, vmin=0, vmax=255, cmap="Reds")
    ax[2].set(title="(c) Red - (Blue + Green)/2")
    for _ax in ax:
        _ax.axis("off")
    fig.savefig("../outputs/RGB2d_diff.png", bbox_inches="tight", pad_inches=0.1)
draw_RGB2d_diff(red, green, blue)

#%%
# RGB別ヒストグラム
def draw_RGB_histgram(red, green, blue, log=False):
    fig, ax = plt.subplots(3, 1, figsize=(6,6), gridspec_kw={"hspace":0.2})
    ax[0].hist(red.ravel(), 256, color="r", alpha=0.9)
    ax[0].set(ylabel="Counts")
    ax[1].hist(green.ravel(), 256, color="g", alpha=0.9)
    ax[1].set(xlabel="Brightness", ylabel="Counts")
    ax[2].hist(blue.ravel(), 256, color="b", alpha=0.9)
    ax[2].set(xlabel="Brightness", ylabel="Counts")
    for _ax in ax:
        _ax.grid(ls="--", alpha=0.4, lw=0.5)
        if log:
            _ax.set_yscale('log')
            _ax.set(ylim=(0, 1e5))
        else:
            _ax.set(ylim=(0, 70000))
    figname = "../outputs/Histgram_RGB"
    if log:
        figname += "_log"
    fig.savefig(figname + ".png", bbox_inches="tight", pad_inches=0.1)
draw_RGB_histgram(red, green, blue, log=True)


# %%
# 差分RGB図
def draw_RGB2d_diff_masked(red, green, blue, thresh):
    red[red < thresh] = 0
    green[green < thresh] = 0
    blue[blue < thresh] = 0
    fig, ax = plt.subplots(1, 3, figsize=(15,5), gridspec_kw={"wspace":0.01})
    ax[0].imshow(red-blue, vmin=0, vmax=255, cmap="Reds")
    ax[0].set(title="(a) Red - Blue")
    ax[1].imshow(red-green, vmin=0, vmax=255, cmap="Reds")
    ax[1].set(title="(b) Red - Green")
    ax[2].imshow(red-(blue+green)/2, vmin=0, vmax=255, cmap="Reds")
    ax[2].set(title="(c) Red - (Blue + Green)/2")
    for _ax in ax:
        _ax.axis("off")
    fig.savefig("../outputs/RGB2d_diff_masked.png", bbox_inches="tight", pad_inches=0.1)
draw_RGB2d_diff_masked(red, green, blue, thresh=img.mean()+img.std()*4)
# %%
# 閾値マスキング → ラベリング → 差分 → 差分 <= 0 を含むラベルを除去

filename = "../sampleimage/子衛星LED点灯.jpg"
img = Image.open(filename)
rgb = np.array(img.convert("RGB")) # R, G, B の順番
red, green, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
diff = red - blue
diff[diff < 0] = 0

gray = np.array(img.convert("L"))
thresh = gray.mean() + gray.std() * 4

mask = np.where(gray >= thresh, 1, 0)

lbl, nlbl = ndi.label(mask)

out_px = 5 # 差分０の画素数がこれ以上ならアウト
for l in range(1, nlbl+1):
    total_diff0_px = (diff[lbl==l] <= 0).sum()
    if total_diff0_px >= out_px:
        diff[lbl==l] = 0
        print(f"label {l} is out (total_diff0_px={total_diff0_px})")

_, n = ndi.label(diff*mask>0)
print(f"number of label = {n}")

fig, ax = plt.subplots(figsize=(7,7))
ax.imshow((diff > 0)*mask, cmap="gray")
ax.set_title(f"Binalized (mask_thresh: {round(thresh,1)}, diff0_thresh: {out_px}px, nlabel: {n})", loc="left")
fig.savefig("../outputs/Binalized_v1.png", bbox_inches="tight", pad_inches=0.1)

# %%
# %%
