# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from coresi.point import Point

_ = torch.set_grad_enabled(False)

plt.set_loglevel("info")


class Image:
    """docstring for Image"""

    def __init__(self, n_energies: int, config: dict, init: str = "zeros", device: torch.device = None):
        super(Image, self).__init__()
        self.dim_in_voxels = Point(*config["n_voxels"])
        self.dim_in_cm = Point(*config["volume_dimensions"])
        self.voxel_size = self.dim_in_cm / self.dim_in_voxels
        self.center = Point(*config["volume_centre"])
        # Bottom left corner
        self.corner = self.center - (self.dim_in_cm / 2)

        # Use provided device or select automatically: CUDA > MPS > CPU
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.n_energies = n_energies
        
        # Create cylindrical mask (inscribed circle in XY plane)
        self.mask = self.set_mask()

        # Contains the actual values of the image
        if init == "zeros":
            self.values = torch.zeros(
                self.n_energies,
                int(self.dim_in_voxels.x),
                int(self.dim_in_voxels.y),
                int(self.dim_in_voxels.z),
                device=self.device,
            )
        if init == "ones":
            self.values = torch.ones(
                self.n_energies,
                int(self.dim_in_voxels.x),
                int(self.dim_in_voxels.y),
                int(self.dim_in_voxels.z),
                device=self.device,
            )

    def set_mask(self):
        """
        Create a binary cylindrical mask with inscribed circle in XY plane.
        The mask is True inside the cylinder (inscribed circle), False outside.
        The cylinder extends along the entire Z axis.
        
        Returns:
            torch.Tensor: Boolean mask of shape (nx, ny, nz) with dtype=torch.bool
        """
        nx = int(self.dim_in_voxels.x)
        ny = int(self.dim_in_voxels.y)
        nz = int(self.dim_in_voxels.z)
        
        # Use the same coordinate system as create_mesh_axes to ensure alignment
        # Points are centered at the middle of each voxel
        x = torch.linspace(
            self.corner.x + (self.voxel_size.x / 2),
            self.corner.x + self.dim_in_cm.x - (self.voxel_size.x / 2),
            nx,
            device=self.device
        )
        y = torch.linspace(
            self.corner.y + (self.voxel_size.y / 2),
            self.corner.y + self.dim_in_cm.y - (self.voxel_size.y / 2),
            ny,
            device=self.device
        )
        
        # Create meshgrid
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Radius of inscribed circle (smaller of the two dimensions)
        radius = min(self.dim_in_cm.x, self.dim_in_cm.y) / 2
        
        # Calculate distance from center for each point in XY plane
        distance = torch.sqrt((X - self.center.x)**2 + (Y - self.center.y)**2)
        
        # Create 2D circular mask as boolean (True inside circle, False outside)
        mask_2d = distance <= radius
        
        # Expand to 3D cylinder (same mask for all Z slices)
        mask_3d = mask_2d.unsqueeze(2).expand(nx, ny, nz).contiguous()
        
        return mask_3d


    def set_to_zeros(self):
        self.values = torch.zeros(
            self.n_energies,
            int(self.dim_in_voxels.x),
            int(self.dim_in_voxels.y),
            int(self.dim_in_voxels.z),
            device=self.device,
        )

    def display_x(
        self, energy: int = 0, slice: int = 0, title: str = "", ax=None, fig=None
    ):
        plt.rcParams.update({'font.size': 16})
        if ax is None:
            fig, ax = plt.subplots()
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
        ax.set_title(f"slice {str(slice)} of the x axis view " + title)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        # fig.savefig("Simu364_x_40_it_20.png", dpi=fig.dpi, bbox_inches='tight')
        plt.show()

    def display_y(
        self, energy: int = 0, slice: int = 0, title: str = "", ax=None, fig=None
    ):
        if ax is None:
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
        ax.set_title(f"slice {str(slice)} of the Y axis view " + title)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        plt.show()

        # def display_z(self, point, energy: int = 0, slice: int = 0, title: str = ""):

    def display_z(
        self, energy: int = 0, slice: int = 0, title: str = "", ax=None, fig=None
    ):
        plt.rcParams.update({'font.size': 16})
        if ax is None:
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
        ax.set_title(f"slice {str(slice)}  of the z axis view " + title)
        fig.colorbar(mappable, cax=cax, orientation="vertical")
        fig.tight_layout()
        # fig.savefig("Simu364_z_20_it_20.png", dpi=fig.dpi, bbox_inches='tight')
        plt.show()

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
            f"{'C++' if cpp else 'version ' + commit} Sensitivity: {config['lm_mlem']['sensitivity_model'] if config['lm_mlem']['sensitivity'] else 'False'}, Algorithm: {config['lm_mlem']['cone_thickness']}\nModel: {config['lm_mlem']['model']}, Iterations: {int(config['lm_mlem']['last_iter'] - config['lm_mlem']['first_iter'] + 1 )}, n_events: {config['n_events']}, dual denoising: {str(config['lm_mlem']['tv'])}"
        )
        fig.tight_layout()
        fig.savefig(config_name)
