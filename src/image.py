import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from point import Point

torch.set_grad_enabled(False)


class Image:
    """docstring for Image"""

    def __init__(self, n_energies: int, config: dict, init="zeros"):
        super(Image, self).__init__()
        self.dim_in_voxels = Point(*config["n_voxels"])
        self.dim_in_cm = Point(*config["volume_dimensions"])
        self.voxel_size = self.dim_in_cm / self.dim_in_voxels
        self.center = Point(*config["volume_centre"])
        # Bottom left corner
        self.corner = self.center - (self.dim_in_cm / 2)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Contains the actual values of the image
        if init == "zeros":
            self.values = torch.zeros(
                n_energies,
                self.dim_in_voxels.x,
                self.dim_in_voxels.y,
                self.dim_in_voxels.z,
                device=device,
            )
        if init == "ones":
            self.values = torch.ones(
                n_energies,
                self.dim_in_voxels.x,
                self.dim_in_voxels.y,
                self.dim_in_voxels.z,
                device=device,
            )

    def display_x(self, slice: int = 0, **params):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[slice, :, :].T.cpu(),
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
        ax.set_title("First slice of the X axis view")
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def display_y(self, slice: int = 0, **params):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[:, slice, :].T.cpu(),
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
        ax.set_title("First slice of the Y axis view")
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def display_z(self, energy: int = 0, slice: int = 0, title: str = ""):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[energy, :, :, slice].T.cpu(),
            origin="lower",
            extent=[
                self.center.x - self.dim_in_cm.x / 2,
                self.center.x + self.dim_in_cm.x / 2,
                self.center.y - self.dim_in_cm.y / 2,
                self.center.y + self.dim_in_cm.y / 2,
            ],
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        ax.set_title("First slice of the Z axis view" + title)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def read_file(self, file_name: str) -> None:
        self.values = self.xp.fromfile(file_name, dtype="double").reshape(
            self.dim_in_voxels.x, self.dim_in_voxels.y, self.dim_in_voxels.z
        )
