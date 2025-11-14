# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import random
from logging import getLogger
from typing import Callable

import torch

# from coresi.camera import Camera, generate_random_angle
from coresi.single_layer_camera import SingleLayerCamera as Camera
from coresi.event import Event
from coresi.image import Image
from coresi.point import Point
from coresi.utils import generate_random_point, generate_weighted_random_point

_ = torch.set_grad_enabled(False)

logger = getLogger("CORESI")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def block(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n_subdivisions: int = 2,
):
    """The sensitivity is the solid angle of scatterer bounding box
    
    Computes sensitivity by sampling detector at multiple points using hierarchical subdivision.
    
    Args:
        cameras: List of camera objects
        volume_config: Dictionary with volume configuration
        x, y, z: Coordinate tensors for voxel centers
        n_subdivisions: Number of recursive subdivisions by 4
            - 0: Detector center only (1 point)
            - 1: 4 quadrant centers (4 points)
            - 2: 16 sub-quadrant centers (16 points)
            - 3: 64 sub-sub-quadrant centers (64 points)
            - etc. Total points = 4^n (or 1 if n=0)
    """
    # cameras = [cameras[0]]
    # As computed here the sensitivity does not depend on the energy
    sensitivity_vol = Image(1, volume_config)
    # Put the volume in the coordinate system of the camera
    for camera in cameras:
        center_points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx], y[idx2], z[idx3])
                        for idx in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.centre
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double).T,
            dims=1,
        ).to(device)

        # Special case: n_subdivisions = 0 means use only detector center
        if n_subdivisions == 0:
            D = torch.abs(center_points[:, 2])
            sq = torch.sqrt(
                torch.pow(center_points[:, 0], 2) +
                torch.pow(center_points[:, 1], 2) +
                torch.pow(D, 2)
            )
            sensitivity_vol.values += (
                D / sq**3 * camera.dim.x * camera.dim.y
            ).reshape(sensitivity_vol.values.shape)
            logger.info(f"Computed sensitivity with detector center only (1 point per detector)")
        else:
            # Compute for the finest subdivision level only
            level = n_subdivisions
            n_points = 4 ** level  # Number of points at finest level
            
            # Generate all possible angle combinations for this level
            # We need 'level' nested loops, each iterating 4 times (angles at 45°, 135°, 225°, 315°)
            # Put angle tensor on the same device/dtype as center_points to avoid device-mismatch
            base_angles = torch.tensor(
                [torch.pi / 2 * (0.5 + i) for i in range(4)],
                device=device,
                dtype=center_points.dtype,
            )
            
            # Create all combinations using meshgrid
            # For level 1: 4 angles -> 4 points
            # For level 2: 4x4 = 16 angle combinations -> 16 points
            # For level 3: 4x4x4 = 64 angle combinations -> 64 points, etc.
            angle_grids = torch.meshgrid(*([base_angles] * level), indexing='ij')
            angles_per_subdivision = torch.stack([grid.flatten() for grid in angle_grids], dim=1)  # [n_points, level]
            
            # Compute cumulative shifts from all subdivision levels
            # Ensure shifts are allocated on the same device/dtype as center_points
            shifts_x = torch.zeros(n_points, device=device, dtype=center_points.dtype)
            shifts_y = torch.zeros(n_points, device=device, dtype=center_points.dtype)
            
            for sub_level in range(level):
                # Distance factor: sqrt(2) / 2^(sub_level+2)
                distance_factor = torch.sqrt(torch.tensor(2.0)) / (2 ** (sub_level + 2))
                shifts_x += distance_factor * camera.dim.x * torch.cos(angles_per_subdivision[:, sub_level])
                shifts_y += distance_factor * camera.dim.y * torch.sin(angles_per_subdivision[:, sub_level])
            
            # Expand center_points to handle all shifts at this level
            center_points_expanded = center_points.unsqueeze(0).expand(n_points, -1, -1)
            
            # Apply shifts
            points_shifted = center_points_expanded.clone()
            points_shifted[:, :, 0] += shifts_x.unsqueeze(1)
            points_shifted[:, :, 1] += shifts_y.unsqueeze(1)
            
            # Compute distances for all points at this level
            D_all = torch.abs(points_shifted[:, :, 2])  # shape: [n_points, n_voxels]
            sq_all = torch.sqrt(
                torch.pow(points_shifted[:, :, 0], 2) +
                torch.pow(points_shifted[:, :, 1], 2) +
                torch.pow(D_all, 2)
            )  # shape: [n_points, n_voxels]
            
            # Sum contributions from all points at finest level and normalize by number of points
            contribution = (D_all / sq_all**3 * camera.dim.x * camera.dim.y).sum(dim=0) / n_points  # shape: [n_voxels]
            sensitivity_vol.values += contribution.reshape(sensitivity_vol.values.shape)
            
            total_points = 4 ** n_subdivisions
            logger.info(f"Computed sensitivity with {n_subdivisions} subdivision levels = {total_points} sampling points per detector (averaged)")

    return sensitivity_vol.values

# def attenuation_exp(
#     cameras: list[Camera],
#     volume_config: dict,
#     energies: list[float],
# 	x: torch.Tensor,
# 	y: torch.Tensor,
# 	z: torch.Tensor,
# ):
#     sensitivity_vol = Image(len(energies), volume_config)
#     cameras = [cameras[0]]
    
#     for camera in cameras:
#         b, d = camera.sca_layers[0].dim.x / 2, camera.sca_layers[0].dim.y / 2
#         a, c = -b, -d
#         Nx = 100
#         Ny = 100
#         hx = (b - a) / Nx
#         hy = (d - c) / Ny
#         # Put the volume in the coordinate system of the camera
#         points = torch.tensordot(
#             (
#                 torch.tensor(
#                     [
#                         Point(x[idx1], y[idx2], z[idx3])
#                         for idx1 in range(len(x))
#                         for idx2 in range(len(y))
#                         for idx3 in range(len(z))
#                     ]
#                 )
#                 - camera.origin
#             ),
#             torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
#             dims=1,
#         ).to(device)
#         for idx_energy, energy in enumerate(energies):
#             # Only compute sensitivity for sca layers, assume probabilities for 
#             # hits in absorber the same. Also assume first hit in sca. To be improved ...
#             for idx_layer, layer in enumerate(camera.sca_layers):
#             #for idx_layer, layer in enumerate(camera.sca_layers + camera.abs_layers):
#                 # Compton linear attenuation coefficient (incoherent scattering)
#                 mu_Compton = camera.get_incoherent_diff_xsection(
#                            energy, layer.detector_type
#                            ) * camera.sca_density
#                 # total linear attenuation coefficient
#                 mu_total = camera.get_total_diff_xsection(
#                            energy, layer.detector_type
#                            ) * camera.sca_density
#                 # 3rd dimension is z
#                 D = torch.abs(points[:, 2] - layer.center.z)
#                 rect = 0.0
#                 for m in range(0, Nx):
#                     for n in range(0, Ny):
#                         sq = (
#                             (
#                                 (a + (m + 0.5) * hx - points[:, 0])
#                                 * (a + (m + 0.5) * hx - points[:, 0])
#                             )
#                             + (
#                                 (c + (n + 0.5) * hy - points[:, 1])
#                                 * (c + (n + 0.5) * hy - points[:, 1])
#                             )
#                             + D**2
#                         )
#                         # rect
#                         # += 1.0/sq*exp(-mu*L*(layer-1)*sqrt(sq)/D);
#                         rect += (
#                             D
#                             * torch.pow(sq, -1.5)
#                             * torch.exp(
#                                 -mu_total
#                                 * camera.sca_layers[0].dim.z
#                                 * idx_layer
#                                 * torch.sqrt(sq)
#                                 / D
#                             )
#                             * (
#                                 1
#                                 - torch.exp(
#                                     -mu_Compton * camera.sca_layers[0].dim.z * torch.sqrt(sq) / D
#                                 )
#                             )
#                         )
#                 sensitivity_vol.values[idx_energy] += (
#                     rect.reshape(sensitivity_vol.values[idx_energy].shape) * hx * hy
#                 )
#     return sensitivity_vol.values


# def lyon_4D(
#     cameras: list[Camera],
#     volume_config: dict,
#     energies: list[float],
#     SM_line: Callable[[Event, bool], Image],
#     mc_samples: int = 1,
# ):
#     """Compute a system matrix by computing the probability of a random gammas
#     reaching the cameras detectos"""
#     sensitivity_vol = Image(len(energies), volume_config)
#     angles = torch.arange(0, 181, 1).deg2rad()
#     cdf_compton = cameras[0].cdf_compton_diff_xsection_dtheta(energies, angles)
#     for camera in cameras:
#         for idx_energy, energy in enumerate(energies):
#             valid_events = 0
#             while valid_events < mc_samples:
#                 sca = random.choice(camera.sca_layers)
#                 absorber = random.choice(camera.abs_layers)
#                 x1 = generate_random_point(sca.dim, sca.center, 1)[0]
#                 x2 = generate_random_point(absorber.dim, absorber.center, 1)[0]
#                 beta = generate_random_angle(cdf_compton, angles, idx_energy)
#                 E_gamma = energy / (1 + (energy / 511.0) * (1 - torch.cos(beta)))
#                 # The Event class does the conversion to centimeters
#                 x1 = x1 * 10
#                 x2 = x2 * 10
#                 # Forge an event with the GATE line's format
#                 line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
#                 try:
#                     event = Event(
#                         0,
#                         line,
#                         energies,
#                         Point(*volume_config["volume_centre"]),
#                         Point(*volume_config["volume_dimensions"]),
#                     )
#                     event.set_camera_index([camera])
#                 except ValueError as e:
#                     logger.debug(f"Skipping event {line.strip()} REASON: {e}")
#                     continue
#                 try:
#                     result = SM_line(event, True).values
#                     # Because we want a sensitivity for a given energy, ensure
#                     # that the SM line for the given energy is non-zero
#                     if result[idx_energy].any():
#                         sensitivity_vol.values[idx_energy] = (
#                             sensitivity_vol.values[idx_energy] + result[idx_energy]
#                         )
#                     else:
#                         raise ValueError(f"Got zeros for energy {energy} keV")
#                 except ValueError as e:
#                     logger.debug(f"Skipping forged event {line.strip()} REASON: {e}")
#                     continue
#                 valid_events += 1

#     # We only do the sum here rather than the average because it's unlikely we
#     # get high values as the chance the cone goes perfectly through the
#     # voxel is low
#     return sensitivity_vol.values


# def sm_like(
#     cameras: list[Camera],
#     volume_config: dict,
#     energies: list[float],
#     SM_line: Callable[[Event, bool], Image],
#     mc_samples: int = 1,
# ):
#     """Compute a system matrix by computing the probability of a random gammas
#     reaching the cameras detectos"""
#     sensitivity_vol = Image(len(energies), volume_config)
#     angles = torch.arange(0, 181, 1).deg2rad()
#     # cdf_compton = cameras[0].cdf_compton_diff_xsection(energies, angles)
#     # volume = Image(len(energies), volume_config, init="ones")
#     for camera in cameras:
#         for idx_energy, energy in enumerate(energies):
#             valid_events = 0
#             while valid_events < mc_samples:
#                 # x0, k = generate_weighted_random_point(volume, 1)
#                 x0 = generate_random_point(sensitivity_vol.dim_in_cm, sensitivity_vol.center, 1)[0]
#                 sca = random.choice(camera.sca_layers)
#                 absorber = random.choice(camera.abs_layers)
#                 x1 = generate_random_point(sca.dim, sca.center, 1)[0]
#                 x2 = generate_random_point(absorber.dim, absorber.center, 1)[0]
#                 r1 = x1 - x0
#                 r2 = x2 - x1
#                 cosbeta = (r1 * r2).sum(axis=0) / (
#                     torch.linalg.norm(r1, 2, axis=0) * torch.linalg.norm(r2, 2, axis=0)
#                 )
#                 E_gamma = energy / (1 + (energy / 511.0) * (1 - cosbeta))
#                 # The Event class does the conversion to centimeters
#                 x1 = x1 * 10
#                 x2 = x2 * 10
#                 # Forge an event with the GATE line's format
#                 line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
#                 try:
#                     event = Event(
#                         0,
#                         line,
#                         energies,
#                         Point(*volume_config["volume_centre"]),
#                         Point(*volume_config["volume_dimensions"]),
#                     )
#                     event.set_camera_index([camera])
#                 except ValueError as e:
#                     logger.debug(f"Skipping event {line.strip()} REASON: {e}")
#                     continue
#                 try:
#                     result = SM_line(event, True).values
#                     # Because we want a sensitivity for a given energy, ensure
#                     # that the SM line for the given energy is non-zero
#                     if result[idx_energy].any():
#                         sensitivity_vol.values[idx_energy] = (
#                             sensitivity_vol.values[idx_energy] + result[idx_energy]
#                         )
#                     else:
#                         raise ValueError(f"Got zeros for energy {energy} keV")
#                 except ValueError as e:
#                     logger.debug(f"Skipping forged event {line.strip()} REASON: {e}")
#                     continue
#                 valid_events += 1

#     # We only do the sum here rather than the average because it's unlikely we
#     # get high values as the chance the cone goes perfectly through the
#     # voxel is low
#     return sensitivity_vol.values
