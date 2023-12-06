from logging import getLogger

import numpy as np
import torch

from camera import Camera
from event import Event
from image import Image
from point import Point

torch.set_grad_enabled(False)

logger = getLogger("CORESI")


def block(
    camera: Camera,
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    """The sensitivity is the solid angle of scatterer bounding box"""
    # As computed here the sensitivity does not depend on the energy
    sensitivity_vol = Image(1, volume_config)
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
    )
    # 3rd dimension is z
    D = torch.abs(points[:, 2] - camera.sca_centre.z)
    sq = torch.sqrt(
        # 1st dimension is x
        torch.pow(points[:, 0] - camera.sca_centre.x, 2)
        # 2nd dimension is y
        + torch.pow(points[:, 1] - camera.sca_centre.y, 2)
        + torch.pow(D, 2)
    )
    return (
        D / sq**3 * camera.sca_layers[0].dim.x * camera.sca_layers[0].dim.y
    ).reshape(sensitivity_vol.values.shape)


def attenuation_exp(
    camera: Camera,
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    # TODO: hard coded in C++, does that correspond to Silicium attenuation?
    mu=0.2038,
):
    # As computed here the sensitivity does not depend on the energy
    sensitivity_vol = Image(1, volume_config)
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
    )
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
    camera: Camera,
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    energies: list[float],
    SM_line: callable,
    # mc_samples: int = int(1e4),
    mc_samples: int = int(2e1),
    point_samples: int = 0.5,
):
    """Compute a system matrix by computing the probability of a random gammas
    reaching the cameras detectos"""
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
    )
    if point_samples > 0:
        indices = torch.randperm(len(points))[:point_samples]
        points = points[indices]
    sensitivity_vol = Image(len(energies), volume_config)
    for idx_layer, layer in enumerate(camera.sca_layers):
        # TODO: Use spectral version of the algorithm to sensitivity for all
        # energies at once?
        for idx_energy, energy in enumerate(energies):
            for _ in range(mc_samples):
                x1 = generate_random_point(layer.dim, layer.center)
                x2 = generate_random_point(
                    camera.abs_layers[0].dim, camera.abs_layers[0].center
                )
                r1 = x1 - points.numpy()
                r2 = x2 - x1
                cosbeta = r1.dot(r2) / (
                    np.linalg.norm(r1, 2, axis=1) * np.linalg.norm(r2, 2)
                )
                E_gamma = energy / (1 + (energy / 511.0) * (1 - cosbeta))
                # The Event class does the convertion to centimeters
                x1 = x1 * 10
                x2 = x2 * 10
                for idx in range(len(E_gamma)):
                    # Forge an event with the GATE line's format
                    line = f"2\t1\t{x1.x}\t{x1.y}\t{x1.z}\t{energy - E_gamma[idx]}\t2\t{x2.x}\t{x2.y}\t{x2.z}\t{E_gamma[idx]}\t3\t0\t0\t0"
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
                            SM_line(0, event, energy != -1).values / mc_samples
                        )[idx_energy]
                    except ValueError as e:
                        logger.debug(
                            f"Skipping forged event {line.strip()} REASON: {e}"
                        )
                        continue
    return sensitivity_vol.values


def generate_random_point(dim: Point, center: Point) -> Point:
    """Generate a random point in a volume"""
    return Point(
        *torch.distributions.uniform.Uniform(
            torch.tensor(
                [center.x - dim.x / 2, center.y - dim.y / 2, center.z - dim.z / 2]
            ),
            torch.tensor(
                [center.x + dim.x / 2, center.y + dim.y / 2, center.z + dim.z / 2]
            ),
        ).sample()
    )
