# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:26:00 2024

@author: Voichita Maxim
"""

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import torch
import yaml
import sys

from coresi.camera import setup_cameras
from coresi.image import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

# open the configuration file to get dimensions
try:
    with open("config_angular.yaml", "r") as fh:
        config = yaml.safe_load(fh)
except IOError as e:
    print(f"Failed to open the configuration file: {e}")
    sys.exit(1)

# load the phantom
source = Image(0, config["volume"])
source.values = torch.load("./checkpoints_simu364_PMB/phantom.pth", weights_only=True)
# load the coordinates of the simulated points
points_z = torch.load("./checkpoints_simu364_PMB/points.pth", map_location=torch.device("cpu"))

# plot the phantom and the histogram of the simulated points
fig, axs = plt.subplots(1, 2)
# axs[0].scatter(
#     [point[0] for point in points_z], [point[1] for point in points_z]
# )
plot_dims = [
    [
        source.center[0] - source.dim_in_cm[0] / 2,
        source.center[0] + source.dim_in_cm[0] / 2,
    ],
    [
        source.center[1] - source.dim_in_cm[1] / 2,
        source.center[1] + source.dim_in_cm[1] / 2,
    ],
]
hist = axs[1].hist2d([point[0] for point in points_z], [point[1] for point in points_z], 
              bins=81, range = plot_dims)
colorbar = fig.colorbar(hist[3], ax=axs[1], pad=0.02, orientation='vertical')  # hist[3] est le mappable
# axs[0].set_xlim(*plot_dims[0])
# axs[0].set_ylim(*plot_dims[1])
source.display_z(ax=axs[0], fig=fig, slice=0)
plt.show()

# compute the theoretical distribution of the Compton angles 
cameras = setup_cameras(config["cameras"])
camera = cameras[0]
energy = config["E0"][0]
angles = torch.arange(0, 181, 1).deg2rad()
KN = torch.tensor(
       [
            camera.get_compton_diff_xsection_dtheta(energy, torch.cos(angle))
            for angle in angles
       ]
)
fig = plt.figure(figsize=(10,4))
plt.plot(angles.rad2deg(), KN/sum(KN)*37000.0, label = "Klein-Nishina")
# load the realization of Compton angles
betas = torch.load("./checkpoints_simu364_PMB/betas.pth", weights_only=True)
plt.hist(betas, 90, density=False, histtype='step', align = 'mid', facecolor='g',
               alpha=0.75, label = "measured Compton angles")
plt.legend(loc="upper right")
# fig.savefig("KN_beta.png", dpi=fig.dpi, bbox_inches='tight')
plt.show()