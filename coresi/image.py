import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from coresi.point import Point

torch.set_grad_enabled(False)

plt.set_loglevel("info")

from matplotlib.patches import Circle


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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_energies = n_energies

        # Contains the actual values of the image
        if init == "zeros":
            self.values = torch.zeros(
                self.n_energies,
                self.dim_in_voxels.x,
                self.dim_in_voxels.y,
                self.dim_in_voxels.z,
                device=self.device,
            )
        if init == "ones":
            self.values = torch.ones(
                self.n_energies,
                self.dim_in_voxels.x,
                self.dim_in_voxels.y,
                self.dim_in_voxels.z,
                device=self.device,
            )

    def set_to_zeros(self):
        self.values = torch.zeros(
            self.n_energies,
            self.dim_in_voxels.x,
            self.dim_in_voxels.y,
            self.dim_in_voxels.z,
            device=self.device,
        )

    def display_x(self, energy: int = 0, slice: int = 0, title: str = ""):
        fig, ax = plt.subplots()
        print(self.values.shape)
        mappable = ax.imshow(
            self.values[energy, slice, :, :].T.cpu(),
            origin="lower",
            # TODO documentj extent and fix centering
            extent=[
                self.center.y - self.dim_in_cm.y / 2,
                self.center.y + self.dim_in_cm.y / 2,
                self.center.z - self.dim_in_cm.z / 2,
                self.center.z + self.dim_in_cm.z / 2,
            ],
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        ax.set_title("First slice of the X axis view")
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

    def display_y(self, energy: int = 0, slice: int = 0, title: str = ""):
        fig, ax = plt.subplots()
        mappable = ax.imshow(
            self.values[energy, :, slice, :].T.cpu(),
            origin="lower",
            extent=[
                self.center.x - self.dim_in_cm.x / 2,
                self.center.x + self.dim_in_cm.x / 2,
                self.center.z - self.dim_in_cm.z / 2,
                self.center.z + self.dim_in_cm.z / 2,
            ],
        )
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        ax.set_title("First slice of the Y axis view")
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

        # def display_z(self, point, energy: int = 0, slice: int = 0, title: str = ""):

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
        # ax.add_patch(Circle((point[0], point[1]), radius=0.1, color="red"))
        ax.set_title("First slice of the Z axis view" + title)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        # plt.show()

    def save_all(
        self,
        config_name: str,
        config: dict,
        cpp: bool = False,
        commit: str = "00000000",
    ):
        if len(config["E0"]) > 1:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            for idx, ax in enumerate(axs.flatten()):
                mappable = ax.imshow(
                    self.values[idx, :, :, 0].T.cpu(),
                    origin="lower",
                    extent=[
                        self.center.x - self.dim_in_cm.x / 2,
                        self.center.x + self.dim_in_cm.x / 2,
                        self.center.y - self.dim_in_cm.y / 2,
                        self.center.y + self.dim_in_cm.y / 2,
                    ],
                )
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                ax.set_title(f"{config['E0'][idx]} keV")
                fig.colorbar(mappable, cax=cax, orientation="vertical")
        else:
            fig, ax = plt.subplots()
            mappable = ax.imshow(
                self.values[0, :, :, 0].T.cpu(),
                origin="lower",
                extent=[
                    self.center.x - self.dim_in_cm.x / 2,
                    self.center.x + self.dim_in_cm.x / 2,
                    self.center.y - self.dim_in_cm.y / 2,
                    self.center.y + self.dim_in_cm.y / 2,
                ],
            )
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            ax.set_title(f"{config['E0'][0]} keV")
            fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.suptitle(
            f"{'C++' if cpp else 'version ' + commit} Sensitivity: {config['lm_mlem']['sensitivity_model'] if config['lm_mlem']['sensitivity'] else 'False'}, Algorithm: {config['lm_mlem']['cone_thickness']}\nModel: {config['lm_mlem']['model']}, Iterations: {int(config['lm_mlem']['last_iter'] - config['lm_mlem']['first_iter'])}, n_events: {config['n_events']}, dual denoising: {str(config['lm_mlem']['tv'])}"
        )
        fig.tight_layout()
        fig.savefig(config_name)

    def read_file(self, file_name: str) -> None:
        self.values = self.xp.fromfile(file_name, dtype="double").reshape(
            self.dim_in_voxels.x, self.dim_in_voxels.y, self.dim_in_voxels.z
        )
