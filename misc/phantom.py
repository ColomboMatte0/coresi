import cv2
import sys

sys.path.append("../coresi")
from coresi.image import Image
import yaml
import torch
import matplotlib.pyplot as plt

with open("config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

phantom = Image(1, config["volume"])
phantom.values[0, :, :, config["volume"]["n_voxels"][2] // 2] = torch.tensor(
    cv2.circle(
        phantom.values[0, :, :, config["volume"]["n_voxels"][2] // 2].clone().numpy(),
        (config["volume"]["n_voxels"][0] // 2, config["volume"]["n_voxels"][1] // 2),
        5,
        (1, 1, 1),
        -1,
    )
)
phantom.display_z(slice=config["volume"]["n_voxels"][2] // 2)
plt.show()
phantom.display_y(slice=config["volume"]["n_voxels"][1] // 2)
plt.show()
torch.save(phantom.values, "phantom.pth")
