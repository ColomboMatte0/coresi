import random
from logging import getLogger

import torch

from coresi.camera import Camera
from coresi.event import Event
from coresi.image import Image
from coresi.interpolation import torch_1d_interp
from coresi.point import Point

torch.set_grad_enabled(False)

logger = getLogger("CORESI")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def block(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    """The sensitivity is the solid angle of scatterer bounding box"""
    cameras = [cameras[0]]
    # As computed here the sensitivity does not depend on the energy
    sensitivity_vol = Image(1, volume_config)
    for camera in cameras:
        points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx], y[idx2], z[idx3])
                        for idx in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.origin
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
            dims=1,
        ).to(device)
        # 3rd dimension is z
        D = torch.abs(points[:, 2] - camera.sca_centre.z)
        sq = torch.sqrt(
            # 1st dimension is x
            torch.pow(points[:, 0] - camera.sca_centre.x, 2)
            # 2nd dimension is y
            + torch.pow(points[:, 1] - camera.sca_centre.y, 2)
            + torch.pow(D, 2)
        )
        sensitivity_vol.values += (
            D / sq**3 * camera.sca_layers[0].dim.x * camera.sca_layers[0].dim.y
        ).reshape(sensitivity_vol.values.shape)

    return sensitivity_vol.values


def attenuation_exp(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    # TODO: hard coded in C++, does that correspond to Silicium attenuation?
    mu=0.2038,
):
    sensitivity_vol = Image(1, volume_config)
    cameras = [cameras[0]]
    for camera in cameras:
        # As computed here the sensitivity does not depend on the energy
        b, d = camera.sca_layers[0].dim.x / 2, camera.sca_layers[0].dim.y / 2
        a, c = -b, -d
        Nx = 100
        Ny = 100
        hx = (b - a) / Nx
        hy = (d - c) / Ny
        points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx], y[idx2], z[idx3])
                        for idx in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.origin
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
            dims=1,
        ).to(device)
        for idx_layer, layer in enumerate(camera.sca_layers + camera.abs_layers):
            # 3rd dimension is z
            D = torch.abs(points[:, 2] - layer.center.z)
            trap = 0.0
            for m in range(1, Nx):
                for n in range(1, Ny):
                    sq = (
                        (
                            (a + (m + 0.5) * hx - points[:, 0])
                            * (a + (m + 0.5) * hx - points[:, 0])
                        )
                        + (
                            (c + (n + 0.5) * hy - points[:, 1])
                            * (c + (n + 0.5) * hy - points[:, 1])
                        )
                        + D**2
                    )
                    # trap
                    # += 1.0/sq*exp(-mu*L*(layer-1)*sqrt(sq)/D);
                    trap += (
                        D
                        * torch.pow(sq, -1.5)
                        * torch.exp(
                            -mu
                            * camera.sca_layers[0].dim.z
                            * (idx_layer - 1)
                            * torch.sqrt(sq)
                            / D
                        )
                        * (
                            1
                            - torch.exp(
                                -mu * camera.sca_layers[0].dim.z * torch.sqrt(sq) / D
                            )
                        )
                    )
            sensitivity_vol.values += (
                trap.reshape(sensitivity_vol.values.shape) * hx * hy
            )
    return sensitivity_vol.values


def valencia_4D(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    energies: list[float],
    SM_line: callable,
    # mc_samples: int = int(1e4),
    # mc_samples: int = int(2e1),
    mc_samples: int = 1,
):
    """Compute a system matrix by computing the probability of a random gammas
    reaching the cameras detectos"""
    cameras = [cameras[0]]
    sensitivity_vol = Image(len(energies), volume_config)
    for camera in cameras:
        points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx], y[idx2], z[idx3])
                        for idx in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.origin
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
            dims=1,
        ).to(device)
        for idx_layer, layer in enumerate(camera.sca_layers):
            # TODO: Use spectral version of the algorithm to sensitivity for all
            # energies at once? Choose an energy at random?
            for idx_energy, energy in enumerate(energies):
                for _ in range(mc_samples):
                    x1 = generate_random_point(layer.dim, layer.center, points.shape[0])
                    x2 = generate_random_point(
                        camera.abs_layers[0].dim,
                        camera.abs_layers[0].center,
                        points.shape[0],
                    )
                    r1 = x1 - points
                    r2 = x2 - x1
                    cosbeta = (r1 * r2).sum(axis=1) / (
                        torch.linalg.norm(r1, 2, axis=1)
                        * torch.linalg.norm(r2, 2, axis=1)
                    )
                    E_gamma = energy / (1 + (energy / 511.0) * (1 - cosbeta))
                    print("e_gamma", E_gamma)
                    # The Event class does the convertion to centimeters
                    x1 = x1 * 10
                    x2 = x2 * 10
                    for idx in range(len(E_gamma)):
                        # Forge an event with the GATE line's format
                        line = f"2\t1\t{x1[idx][0]}\t{x1[idx][1]}\t{x1[idx][2]}\t{energy - E_gamma[idx]}\t2\t{x2[idx][0]}\t{x2[idx][1]}\t{x2[idx][2]}\t{E_gamma[idx]}\t3\t0\t0\t0\t0"
                        try:
                            event = Event(
                                0,
                                line,
                                energies,
                                Point(*volume_config["volume_centre"]),
                                Point(*volume_config["volume_dimensions"]),
                            )
                            event.set_camera_index([camera])
                        except ValueError as e:
                            logger.debug(f"Skipping event {line.strip()} REASON: {e}")
                            continue
                        try:
                            sensitivity_vol.values[idx_energy] += (
                                SM_line(0, event, energy != -1).values
                            )[idx_energy]
                        except ValueError as e:
                            logger.debug(
                                f"Skipping forged event {line.strip()} REASON: {e}"
                            )
                            continue

    return sensitivity_vol.values / mc_samples, points


def lyon_4D(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    energies: list[float],
    SM_line: callable,
    mc_samples: int = 1,
):
    """Compute a system matrix by computing the probability of a random gammas
    reaching the cameras detectos"""
    sensitivity_vol = Image(len(energies), volume_config)
    angles = torch.arange(0, 181, 1).deg2rad()
    cdf_compton = cameras[0].cdf_compton_diff_xsection(energies, angles)
    for camera in cameras:
        for idx_energy, energy in enumerate(energies):
            valid_events = 0
            while valid_events < mc_samples:
                sca = random.choice(camera.sca_layers)
                abs = random.choice(camera.abs_layers)
                x1 = generate_random_point(sca.dim, sca.center, 1)[0]
                x2 = generate_random_point(abs.dim, abs.center, 1)[0]
                beta = generate_random_angle(cdf_compton, angles, idx_energy)
                E_gamma = energy / (1 + (energy / 511.0) * (1 - torch.cos(beta)))
                # The Event class does the conversion to centimeters
                x1 = x1 * 10
                x2 = x2 * 10
                # Forge an event with the GATE line's format
                line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
                try:
                    event = Event(
                        0,
                        line,
                        energies,
                        Point(*volume_config["volume_centre"]),
                        Point(*volume_config["volume_dimensions"]),
                    )
                    event.set_camera_index([camera])
                except ValueError as e:
                    logger.debug(f"Skipping event {line.strip()} REASON: {e}")
                    continue
                try:
                    result = SM_line(0, event).values
                    # Because we want a sensitivity for a given energy, ensure
                    # that the SM line for the given energy is non-zero
                    if result[idx_energy].any():
                        sensitivity_vol.values = sensitivity_vol.values + result
                    else:
                        raise ValueError(f"Got zeros for energy {energy} keV")
                except ValueError as e:
                    logger.debug(f"Skipping forged event {line.strip()} REASON: {e}")
                    continue
                valid_events += 1

    # We only do the sum here rather than the average because it's unlikely we
    # get high values as the chance the cone goes perfectly through the
    # voxel is low
    return sensitivity_vol.values


def generate_random_angle(cdf_KN: torch.Tensor, angles: torch.Tensor, energy_idx: int):
    """docstring for generate_random_angle"""
    x = torch.distributions.uniform.Uniform(0, 1).sample((1,))
    return torch_1d_interp(x[0], cdf_KN[energy_idx], angles)


def generate_random_point(dim: Point, center: Point, n_points: int) -> torch.Tensor:
    """Generate random points in a volume"""
    return torch.distributions.uniform.Uniform(
        torch.tensor(
            [center.x - dim.x / 2, center.y - dim.y / 2, center.z - dim.z / 2]
        ),
        torch.tensor(
            [center.x + dim.x / 2, center.y + dim.y / 2, center.z + dim.z / 2]
        ),
    ).sample((n_points,))
