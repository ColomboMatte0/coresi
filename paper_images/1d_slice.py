# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import torch
import yaml
import sys

sys.path.append("..")
from coresi.image import Image


with open("../config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

images = ["sens-mc-poly-parallel.pth", "sens-mc-364-angular.pth"]
# images = ["sens-solid-attn.pth"]


fig, axs = plt.subplots(2, 1, figsize=(10, 10))

for image_name in images:
    image = Image(len(config["E0"]), config["volume"])
    image.values = torch.load(image_name, map_location=torch.device("cpu"))
    volume = image.values
    volume = volume / volume.sum(dim=[1, 2, 3], keepdim=True)
    profile_h = volume[:, :, volume.shape[-2] // 2, volume.shape[-1] // 2]
    profile_v = volume[:, volume.shape[-3] // 2, volume.shape[-2] // 2, :]
    if "poly" in image_name:
        profile_h = profile_h.squeeze()
        profile_v = profile_v.squeeze()
        for e in range(profile_h.shape[0]):
            axs[0].plot(
                torch.linspace(
                    -config["volume"]["volume_dimensions"][0] / 2,
                    config["volume"]["volume_dimensions"][0] / 2,
                    config["volume"]["n_voxels"][0],
                ),
                profile_h[e],
                label=f"{config['E0'][e]} keV",
            )

            axs[1].plot(
                torch.linspace(
                    -config["volume"]["volume_dimensions"][2] / 2,
                    config["volume"]["volume_dimensions"][2] / 2,
                    config["volume"]["n_voxels"][2],
                ),
                profile_v[e],
                label=f"{config['E0'][e]} keV",
            )
    else:
        axs[0].plot(
            torch.linspace(
                -config["volume"]["volume_dimensions"][0] / 2,
                config["volume"]["volume_dimensions"][0] / 2,
                config["volume"]["n_voxels"][0],
            ),
            profile_h[0],
            label=image_name.split(".")[0].replace("_", " "),
        )
        axs[1].plot(
            torch.linspace(
                -config["volume"]["volume_dimensions"][2] / 2,
                config["volume"]["volume_dimensions"][2] / 2,
                config["volume"]["n_voxels"][2],
            ),
            profile_v[0],
            label=image_name.split(".")[0].replace("_", " "),
        )


for ax in axs.ravel():
    ax.legend(loc="upper right")
plt.savefig("1d_profile_" + images[0] + ".png", dpi=300)
plt.show()
