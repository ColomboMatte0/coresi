# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import torch


def torch_1d_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: float | None = None,
    right: float | None = None,
) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.
    taken from https://github.com/pytorch/pytorch/issues/1552#issuecomment-1563150850

    Remove once pytorch interopolation is merged

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: 1d sequence of floats. x-coordinates. Must be increasing
        fp: 1d sequence of floats. y-coordinates. Must be same length as xp
        left: Value to return for x < xp[0], default is fp[0]
        right: Value to return for x > xp[-1], default is fp[-1]

    Returns:
        The interpolated values, same shape as x.
    """
    if left is None:
        left = fp[0]

    if right is None:
        right = fp[-1]

    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)

    answer = torch.where(
        x < xp[0],
        left,
        (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )
    answer = torch.where(x > xp[-1], right, answer)
    return answer


def interpolate(volume: torch.Tensor, new_size: list) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        # Unsqueeze to add a batch dimension
        volume.unsqueeze(0),
        size=new_size,
        mode="trilinear",
    )
