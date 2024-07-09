import matplotlib.pyplot as plt
import torch
import yaml
import sys

sys.path.append("..")
from coresi.image import Image


with open("../config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

images = ["sens-mc-poly-parallel.pth"]
image = Image(len(config["E0"]), config["volume"])
image.values = torch.load(images[0], map_location=torch.device("cpu"))
print(image.values.shape)
volume = image.values


fig, axs = plt.subplots(2, 1, figsize=(10, 10))

profile_h = volume[:, :, volume.shape[-2] // 2, volume.shape[-1] // 2].squeeze()
profile_v = volume[:, volume.shape[-3] // 2, volume.shape[-2] // 2, :].squeeze()
for e in range(profile_h.shape[0]):
    axs[0].plot(profile_h[e], label=f"{config['E0'][e]} keV")
    axs[0].set_xticks(list(range(0, profile_h.shape[-1], 5)))
    axs[0].set_xticklabels(
        list(range((-profile_h.shape[-1] // 2) + 1, profile_h.shape[-1] // 2 + 1, 5))
    )

for e in range(profile_v.shape[0]):
    axs[1].plot(profile_v[e], label=f"{config['E0'][e]} keV")
    axs[1].set_xticks(list(range(0, profile_v.shape[-1], 5)))
    axs[1].set_xticklabels(
        list(range((-profile_v.shape[-1] // 2) + 1, profile_v.shape[-1] // 2 + 1, 5))
    )

for ax in axs.ravel():
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_title("1 d profile for " + images[0])
plt.savefig("1d_profile_" + images[0] + ".png", dpi=300)
plt.show()
