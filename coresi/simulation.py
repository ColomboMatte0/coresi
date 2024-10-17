# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import random
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch

from coresi.camera import Camera, generate_random_angle
from coresi.image import Image
from coresi.utils import generate_random_point, generate_weighted_random_point

logger = getLogger("CORESI")


def simulate(
    source_filename: str,
    config: dict,
    cameras: list[Camera],
    n_events: int,
    energy: float,
    n_v2_samples: int = 500,
    angle_threshold: float = 2,
    visualize_generated_source: bool = False,
) -> str:
    """
    angle_threshold is in degrees
    """
    source = Image(0, config["volume"])
    source.values = torch.load(source_filename, weights_only=True)
    if source.values.shape[0] != 1:
        raise ValueError(
            "The source should be for one energy, expected dimensions are [1, W, H, D]"
        )
    simulated_events: list[str] = []
    n_forged_events = 0
    n_valid_forged_events = 0
    angles = torch.arange(0, 181, 1).deg2rad()
    points = []
    slice = []
    betas = []
    logger.info(f"Doing a simulation for {str(n_events)} events")
    while n_valid_forged_events < n_events:
        n_forged_events += 1
        #x0 is in the xyplane, k is the slice number
        x0, k = generate_weighted_random_point(source, 1) 
        camera = random.choice(cameras)
        cdf_compton = camera.cdf_compton_diff_xsection_dtheta([energy], angles)
        sca = random.choice(camera.sca_layers)
        x1 = generate_random_point(sca.dim, sca.center, 1)[0]
        r1 = x1 - x0
        # The simulation is mono-energetic so idx_energy is 0
        beta = generate_random_angle(cdf_compton, angles, 0)
        v2_candidates = []
        # It's hard to compute the exact x2 that corresponds to the randomly chosen
        # angle. So draw candidates and pick one that loosely match the angle
        for _ in range(n_v2_samples):
            absorber = random.choice(camera.abs_layers)
            x2 = generate_random_point(absorber.dim, absorber.center, 1)[0]
            r2 = x2 - x1
            candidate_beta = torch.acos(
                (r1 * r2).sum(axis=0)
                / (torch.linalg.norm(r1, 2, axis=0) * torch.linalg.norm(r2, 2, axis=0))
            )
            # Compute the error between the chosen angle and angle computed for
            # a v2 candidate
            v2_candidates.append([abs(candidate_beta - beta), x2])
        # Sort by the error e.g. the difference between the randomly chosen
        # angle and the angle computed for the randomly chosen x2
        best_try = min(v2_candidates, key=lambda v2: v2[0])
        # If the best candidate is within a tolerance
        if best_try[0].item() <= torch.tensor(angle_threshold).deg2rad():
            n_valid_forged_events += 1
            points.append(x0)
            slice.append(k)
            betas.append(beta.rad2deg())
            x2 = best_try[1]
            # Gate output used to be in mm. Moreover, the Event class expects mm
            # and does the conversion, so multiply by 10 to account for that
            x1 = x1 * 10
            x2 = x2 * 10
            E_gamma = energy / (1 + (energy / 511.0) * (1 - torch.cos(beta)))
            line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
            simulated_events.append(line)
    if visualize_generated_source:
        visualize_source_points(points, slice, source)
    torch.save(betas, "betas.pth")
    logger.info("Simulation done")
    logger.debug(
        f"V2 success rate = {str(n_valid_forged_events * 100 / n_forged_events)}%"
    )
    return "\n".join(simulated_events)


def visualize_source_points(points, slice, source):
    points = torch.tensor(np.array(points))
    torch.save(points, "points.pth")
    slice = torch.tensor(slice)
    torch.save(slice, "slice.pth")
    # for z in range(source.dim_in_voxels[-1]):
    for z in range(0, 1):
        mask = torch.isin(slice, z)
        points_z = points[mask]

        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(
            [point[0] for point in points_z], [point[1] for point in points_z]
        )
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
        axs[0].set_xlim(*plot_dims[0])
        axs[0].set_ylim(*plot_dims[1])
        ##  axs[0].grid()
        # axs[0].set_xticks(
        #     np.linspace(*plot_dims[0], (source.dim_in_voxels[0] + 1) // 2)
        # )
        # axs[0].set_yticks(
        #     np.linspace(*plot_dims[1], (source.dim_in_voxels[1] + 1) // 2)
        # )
        source.display_z(ax=axs[1], fig=fig, slice=z)
        plt.show()
