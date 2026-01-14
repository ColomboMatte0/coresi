# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import os
import sys
from logging import getLogger
from math import pi

import numpy as np
import torch

from coresi.single_layer_camera import SingleLayerCamera as Camera
from coresi.Events import Events
from coresi.image import Image
from coresi.interpolation import torch_1d_interp

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)

class SM_Model(object):
    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        E_0: float,
        device: torch.device,
    ):
        super(SM_Model, self).__init__()
        self.cone_thickness = config_mlem["cone_thickness"]
        self.mu_total = cameras[0].get_total_diff_xsection(E_0) * cameras[0].density

        self.compute_theta_j = True
        if config_mlem["model"] == "cos0rho0":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False
            self.do_nothing = True

            def model(kbl_j):
                return kbl_j

        elif config_mlem["model"] == "cos0rho2":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False
            self.do_nothing = False

            def model(inv_rho_j):
                return inv_rho_j*inv_rho_j

        elif config_mlem["model"] == "cos1rho2":
            self.do_nothing = False
            def model(cos_theta_j, inv_rho_j):
                return abs(cos_theta_j) * inv_rho_j * inv_rho_j

        else:
            logger.fatal(
                f"Model {config_mlem['model']} is not supported, either use cos0rho0, cos0rho2 or cos1rho2"
            )
            sys.exit(1)

        self.model = model
        self.cameras = cameras
        self.device = device
        
        self.m_e = torch.tensor(
            511, dtype=torch.float, device=self.device
        )  # electron mass in EV

        self.inv_sqrt_2pi = torch.tensor(
            1.0 / np.sqrt(2.0 * np.pi), 
            dtype=torch.float, 
            device=self.device
        )
        self.E_0 = E_0

        self.config_volume = config_volume
        self.line = Image(self.config_volume, device=self.device)

        if self.cone_thickness == "parallel":
            self.sigma_beta = (
                self.line.voxel_size.norm2() * config_mlem["width_factor"] / 2
            )
            # Skip the Gaussian above n_sigma * Gaussian std
            self.limit_sigma = self.sigma_beta * config_mlem["n_sigma"]
            self.SM_line = self.SM_parallel_thickness
            # self.SM_line = self.SM_parallel_thickness
        elif self.cone_thickness == "doppler":
            self.limit_arm =  config_mlem["maximum_cone_thickness"]
            self.SM_line = self.SM_arm

        logger.info(f"Using algorithm {self.SM_line.__name__}")

        x, y, z = SM_Model.create_mesh_axes(
            [
                self.line.corner.x + (self.line.voxel_size.x / 2),
                self.line.corner.x
                + self.line.dim_in_cm.x
                - (self.line.voxel_size.x / 2),
            ],
            self.line.dim_in_voxels.x,
            [
                self.line.corner.y + (self.line.voxel_size.y / 2),
                self.line.corner.y
                + self.line.dim_in_cm.y
                - (self.line.voxel_size.y / 2),
            ],
            self.line.dim_in_voxels.y,
            [
                self.line.corner.z + (self.line.voxel_size.z / 2),
                self.line.corner.z
                + self.line.dim_in_cm.z
                - (self.line.voxel_size.z / 2),
            ],
            self.line.dim_in_voxels.z,
        )

        # Used to go through the volume
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, sparse=True, indexing="ij")
        self.xx = torch.from_numpy(self.xx).to(dtype=torch.float32, device=self.device)
        self.yy = torch.from_numpy(self.yy).to(dtype=torch.float32, device=self.device)
        self.zz = torch.from_numpy(self.zz).to(dtype=torch.float32, device=self.device)

    def SM_arm(
        self, events: Events, indices: torch.Tensor
    ) -> torch.Tensor:

        beta = events.beta[indices]              # [batch, 1, 1, 1]
        E0 = events.E0[indices]                  # [batch, 1, 1, 1]
        inv_sigma_arm = events.inv_sigma_total[indices]  # [batch, 1, 1, 1]
        inv_sigma_arm_sq = events.inv_sigma_total_sq[indices]  # [batch, 1, 1, 1]
        V1 = events.V1[indices]  # [batch, 3, 1, 1, 1]
        axis = events.axis[indices]  # [batch, 3, 1, 1, 1]
        camera_Oz = events.camera_Oz[indices]  # [batch, 3, 1, 1, 1]
        depth_z = events.depth_z[indices]  # [batch, 1, 1, 1]

        dx = self.xx - V1[:,0]  # [batch, nx, 1, 1]
        dy = self.yy - V1[:,1]  # [batch, 1, ny, 1]
        dz = self.zz - V1[:,2]  # [batch, 1, 1, nz]

        rho_j_sq = torch.addcmul(torch.addcmul(dx * dx, dy, dy), dz, dz)
        inv_rho_j = torch.rsqrt(rho_j_sq) 

        cos_delta_j = torch.addcmul(
            torch.addcmul(axis[:,0] * dx, axis[:,1], dy), 
            axis[:,2], dz
        ) * inv_rho_j
        cos_delta_j.clamp_(-1.0, 1.0)  # In-place clamp

        line_values = torch.abs(torch.arccos(cos_delta_j) - beta)
        inner_mask = line_values <= self.limit_arm

        line_values = self.inv_sqrt_2pi * inv_sigma_arm * torch.exp(-0.5 * line_values * line_values * inv_sigma_arm_sq)

        KN = torch.reciprocal(torch.addcmul(
            torch.ones_like(E0),
            E0 * torch.reciprocal(self.m_e),
            1.0 - cos_delta_j
        ))
        KN_sq = KN * KN
        KN = torch.addcmul(
            KN * (KN_sq + 1.0),
            KN_sq,
            cos_delta_j * cos_delta_j - 1.0
        )
        if self.compute_theta_j:
            cos_theta_j = torch.addcmul(
                torch.addcmul(camera_Oz[:,0] * dx, camera_Oz[:,1], dy),
                camera_Oz[:,2], dz* inv_rho_j
            ) 
            
            attn = torch.exp(-self.mu_total * (depth_z / cos_theta_j)) 
            
            # # Average over events axis (only counting valid voxels)
            # mask_expanded = self.line.mask.unsqueeze(0).expand_as(cos_theta_j)  # [batch, nx, ny, nz]
            # attn_masked = torch.where(mask_expanded, attn, torch.zeros_like(attn))
            # mean_attn_volume = attn_masked.sum(dim=0) / mask_expanded.sum(dim=0).clamp(min=1)  # [nx, ny, nz]
            
            # # Save the averaged attenuation volume
            # torch.save(mean_attn_volume.cpu(), 'sensitivity/attn_mean_volume.pth')
            # print(f"Saved mean attenuation volume with shape {mean_attn_volume.shape}")
            
            line_values *= (
                self.model(cos_theta_j, inv_rho_j) * 
                attn
            )
            # line_values *= self.model(cos_theta_j, inv_rho_j)
        elif not self.do_nothing:
            self.line.values *= self.model(inv_rho_j)

        return line_values * KN * inner_mask.float()

    def SM_parallel_thickness(
        self, events: Events, indices: torch.Tensor
    ) -> torch.Tensor:
        """docstring for SM_parallel_thickness"""

        beta = events.beta[indices]              # [batch, 1, 1, 1]
        E0 = events.E0[indices]                  # [batch, 1, 1, 1]
        inv_sigma_arm = events.inv_sigma_total[indices]  # [batch, 1, 1, 1]
        inv_sigma_arm_sq = events.inv_sigma_total_sq[indices]  # [batch, 1, 1, 1]
        V1 = events.V1[indices]  # [batch, 3, 1, 1, 1]
        axis = events.axis[indices]  # [batch, 3, 1, 1, 1]
        camera_Oz = events.camera_Oz[indices]  # [batch, 3, 1, 1, 1]

        dx = self.xx - V1[:,0]  # [batch, nx, 1, 1]
        dy = self.yy - V1[:,1]  # [batch, 1, ny, 1]
        dz = self.zz - V1[:,2]  # [batch, 1, 1, nz]

        rho_j_sq = torch.addcmul(torch.addcmul(dx * dx, dy, dy), dz, dz)
        inv_rho_j = torch.rsqrt(rho_j_sq) 

        cos_delta_j = torch.addcmul(
            torch.addcmul(axis[:,0] * dx, axis[:,1], dy), 
            axis[:,2], dz
        ) * inv_rho_j
        cos_delta_j.clamp_(-1.0, 1.0)  # In-place clamp

        line_values = torch.sqrt(rho_j_sq) * torch.abs(torch.sin(beta - torch.acos(cos_delta_j) ))
        inner_mask = line_values <= self.limit_sigma
  
        # Apply the Gaussian
        line_values = torch.exp(
            -(line_values * line_values) * 0.5 / (self.sigma_beta*self.sigma_beta)
        )

        KN = torch.reciprocal(torch.addcmul(
            torch.ones_like(E0),
            E0 * torch.reciprocal(self.m_e),
            1.0 - cos_delta_j
        ))
        KN_sq = KN * KN
        KN = torch.addcmul(
            KN * (KN_sq + 1.0),
            KN_sq,
            cos_delta_j * cos_delta_j - 1.0
        )

        if self.compute_theta_j:
            cos_theta_j = torch.addcmul(
                torch.addcmul(camera_Oz[:,0] * dx, camera_Oz[:,1], dy),
                camera_Oz[:,2], dz
            ) * inv_rho_j
            line_values *= self.model(cos_theta_j, inv_rho_j)
        elif not self.do_nothing:
            line_values *= self.model(inv_rho_j)

        return line_values * KN * inner_mask.float()

    @staticmethod
    def create_mesh_axes(
        x_range: tuple,
        x_steps: int,
        y_range: tuple,
        y_steps: int,
        z_range: tuple,
        z_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample points along each volume dimension. use voxel size to center
        # the points on the voxels
        return (
            torch.linspace(*x_range, x_steps),
            torch.linspace(*y_range, y_steps),
            torch.linspace(*z_range, z_steps),
        )
