import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gpuoptional import array_module
import numpy as np

from point import Point


class Image:
    """docstring for Image"""

    def __init__(self, config: dict):
        super(Image, self).__init__()
        self.dim_in_voxels = Point(*config["n_voxels"])
        self.dim_in_cm = Point(*config["volume_dimensions"])
        self.voxel_size = self.dim_in_cm / self.dim_in_voxels
        self.center = Point(*config["volume_centre"])
        # Bottom left corner
        self.corner = self.center - (self.dim_in_cm / 2)

        self.xp = array_module()
        # Contains the actual values of the image
        self.values = self.xp.zeros(
            (self.dim_in_voxels.x, self.dim_in_voxels.y, self.dim_in_voxels.z)
        )
        self.values[self.dim_in_voxels.x // 2, 0, 0] = 1
        self.values[0, self.dim_in_voxels.x // 2, 0] = 1

    def display_x(self, slice: int = 0, **params):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[slice, :, :].T,
            origin="lower",
            # TODO documentj extent and fix centering
            extent=[
                self.center.y - self.dim_in_cm.y / 2,
                self.center.y + self.dim_in_cm.y / 2,
                self.center.z - self.dim_in_cm.z / 2,
                self.center.z + self.dim_in_cm.z / 2,
            ],
            **params,
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def display_y(self, slice: int = 0, **params):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[:, slice, :].T,
            origin="lower",
            extent=[
                self.center.x - self.dim_in_cm.x / 2,
                self.center.x + self.dim_in_cm.x / 2,
                self.center.z - self.dim_in_cm.z / 2,
                self.center.z + self.dim_in_cm.z / 2,
            ],
            **params,
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def display_z(self, slice: int = 0, **params):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[:, :, slice].T,
            origin="lower",
            extent=[
                self.center.x - self.dim_in_cm.x / 2,
                self.center.x + self.dim_in_cm.x / 2,
                self.center.y - self.dim_in_cm.y / 2,
                self.center.y + self.dim_in_cm.y / 2,
            ],
            **params,
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def read_file(self, file_name: str) -> None:
        self.values = self.xp.fromfile(file_name, dtype="double").reshape(
            self.dim_in_voxels.x, self.dim_in_voxels.y, self.dim_in_voxels.z
        )
