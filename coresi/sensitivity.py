# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT


from logging import getLogger

import torch
import numpy as np

from coresi.single_layer_camera import SingleLayerCamera as Camera
from coresi.image import Image

_ = torch.set_grad_enabled(False)

logger = getLogger("CORESI")

def solid_angle(
    cameras: list[Camera],
    volume_config: dict,
    E_0: float,
    sens_device: torch.device,
    sub_N: list[int],
    use_attn: bool,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    sensitivity_vol = Image(volume_config,sens_device)
    hx = cameras[0].dim.x / sub_N[0]
    hy = cameras[0].dim.y / sub_N[1]
    hz = cameras[0].dim.z / sub_N[2]
    a = -cameras[0].dim.x / 2
    c = -cameras[0].dim.y / 2

    # Compute the grid ONCE per camera
    m = torch.arange(sub_N[0], device=sens_device, dtype=torch.double)
    n = torch.arange(sub_N[1], device=sens_device, dtype=torch.double)
    l = torch.arange(sub_N[2], device=sens_device, dtype=torch.double) 
    mm, nn, ll = torch.meshgrid(m, n, l, indexing='ij')  # shape (sub_Nx, sub_Ny, sub_Nz)
    x_grid = (a + (mm + 0.5) * hx).unsqueeze(-1)  # shape (sub_Nx, sub_Ny, sub_Nz)
    y_grid = (c + (nn + 0.5) * hy).unsqueeze(-1)  # shape (sub_Nx, sub_Ny, sub_Nz)
    z_grid = (-(ll + 0.5) * hz).unsqueeze(-1)  # shape (sub_Nx, sub_Ny, sub_Nz)

    points_world = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)  # shape: (N_voxels, 3)
    exp_total = 1
    exp_compton = 1 

    for camera in cameras:

        # Put the volume in the coordinate system of the camera
        camera_rotation = torch.tensor(np.array([camera.Ox, camera.Oy, camera.Oz]), dtype=torch.double).T
        points = torch.tensordot(points_world - camera.centre, camera_rotation, dims=1).to(sens_device)

        points_x = points[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, Npoints)
        points_y = points[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, Npoints)
        points_z = points[:, 2].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, Npoints)

        # Compute sq for all grid and all points at once
        sq = (
            (x_grid - points_x) ** 2 +
            (y_grid - points_y) ** 2 +
            (z_grid - points_z) ** 2
        )  # (sub_Nx, sub_Ny, sub_Nz, Npoints)

        D = torch.abs(z_grid - points_z) # (sub_Nx, sub_Ny, sub_Nz, Npoints) 

        if  use_attn:

            depth = torch.abs(z_grid)
            mu_compton = camera.get_incoherent_diff_xsection(E_0) * camera.density
            mu_total = camera.get_total_diff_xsection(E_0) * camera.density

            logger.info(f"mu_compton = {mu_compton:.6f} cm-1")
            logger.info(f"mu_total = {mu_total:.6f} cm-1")

            # mask_flat = sensitivity_vol.mask.flatten() 
            # A_full = D / torch.sqrt(sq)  # shape: (sub_Nx, sub_Ny, sub_Nz, Npoints)
            # A = A_full[:, :, 1, mask_flat]  # shape: (sub_Nx, sub_Ny, sub_Nz, N_valid_points)

            # logger.info(f"A range: [{A.min().item():.4f}, {A.max().item():.4f}]")

            exp_total = torch.exp(-mu_total * depth * torch.sqrt(sq) / D)   
            logger.info(f"exp_total - min: {exp_total.min().item():.6f}, max: {exp_total.max().item():.6f}, mean: {exp_total.mean().item():.6f}")
            # exp_compton = 1 - torch.exp(-mu_compton * hz * torch.sqrt(sq) / D)

        rect = ((D * torch.pow(sq, -1.5)) * exp_total * exp_compton).sum(dim=(0, 1, 2))  # sum over m, n, and l


        rect = rect.reshape(sensitivity_vol.values.shape)
        sensitivity_vol.values += rect
    
    sensitivity_vol.values[~sensitivity_vol.mask] = 0.0
    voldim = volume_config["n_voxels"][0] * volume_config["n_voxels"][1] * volume_config["n_voxels"][2]
    sensitivity_vol.values = (sensitivity_vol.values / torch.linalg.norm(sensitivity_vol.values)) *voldim

    sensitivity_vol.values[~sensitivity_vol.mask] = 1.0
    return sensitivity_vol.values
