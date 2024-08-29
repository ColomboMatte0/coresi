import torch
from torch.utils.data import WeightedRandomSampler

from coresi.image import Image
from coresi.point import Point


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


def generate_weighted_random_point(volume: Image, n_points: int = 1) -> torch.Tensor:
    """Generate random points in a volume"""
    sample = list(
        WeightedRandomSampler(volume.values.reshape(-1), n_points, replacement=True)
    )
    # The first coordinate is the energy but this function is currently used for
    # simulation with monoenergy source so the energy is always the first
    # therefore we don't need it
    _, i, j, k = [
        int(idx)
        for idx in torch.unravel_index(torch.tensor(sample), volume.values.shape)
    ]
    sampled_point = (
        volume.corner + Point(i, j, k) * volume.voxel_size + volume.voxel_size / 2
    )

    sampled_point = generate_random_point(volume.voxel_size, sampled_point, 1).squeeze(
        0
    )
    return sampled_point, k
