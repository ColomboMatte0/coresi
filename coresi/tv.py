import torch
from math import sqrt

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:54:52 2020

Iterative algorithms for the reconstruction

@author: Louise Friot--Giroux and adapted to PyTorch by Vincent Lequertier
"""


def torch_gradient(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of a

    Parameters
    ----------
    a : torch.array
        data

    Returns
    -------
    grad : torch.array
           gradient(a)
    """
    dim = a.dim()
    grad = torch.zeros_like(a, dtype=torch.float32).repeat(
        dim, *[1 for _ in range(dim)]
    )
    for d in range(dim):
        grad[d] = torch.diff(
            a, 1, axis=d, append=a.select(dim=d, index=-1).unsqueeze(d)
        )
    return grad


def torch_gradient_div(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of a

    Parameters
    ----------
    a : torch.array
        data

    Returns
    -------
    grad : torch.array
           gradient(a)
    """
    dim = a.dim()
    grad = torch.zeros_like(a, dtype=torch.float32).repeat(
        dim, *[1 for _ in range(dim)]
    )
    for d in range(dim):
        grad[d] = torch.diff(
            a, 1, axis=d, prepend=a.select(dim=d, index=-1).unsqueeze(d)
        )
    return grad


def torch_divergence(u):
    """
    Compute divergence of u

    Parameters
    ----------
    u : torch.array
        data

    Returns
    -------
    torch.array
        divergence(u)
    """
    dim = u.shape[0]
    return torch.stack([torch_gradient_div(u[d])[d] for d in range(dim)]).sum(dim=0)


def torch_module(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the module

    Parameters
    ----------
    q : torch.array
        dimension of q : (2,Nx,Ny)

    Return
    ------
    m : torch.array
        (m)_ij = ((q1)_ij**2 + (q2)_ik**2))**(1/2))
        dimension of m : (Nx,Ny)
    """
    return torch.sqrt(torch.sum(torch.pow(q, 2), axis=0))


def torch_TV(x: torch.Tensor) -> float:
    """
    Compute TV norm of x

    Parameters
    ----------
    x : torch.array

    Returns
    -------
    res : float()
          TV norm of x
    """
    # x = crop(x,10)
    grad = torch_gradient(x)
    res = torch_module(grad)
    return float(torch.sum(res))


def TV_denoise(
    f: torch.Tensor,
    alpha: float = 12,
    iters: int = 3,
    tol: float = 0.01,
    tau: float = 0.248,
) -> torch.Tensor:
    """
    Denoise the image f with Chambolle's projection algorithm. used for gaussian
    noise

    Parameters
    ----------
    f : array
        noised image
    alpha : float
            trade-off parameter (recommanded : 12)
    iters : int
        number max of iterations
    tol : float
          tolerance difference between two successives iterations
    tau : float
          time step, default = 0.248 (2D), 0.125 (3D)

    Return
    ------
    u : array
        denoised image
    """
    dim = f.dim()
    if dim == 3:
        tau = 1 / 8
    p = torch.zeros_like(f, dtype=torch.float32).repeat(dim, *[1 for _ in range(dim)])
    div = torch.zeros_like(f)
    for _ in range(iters):
        v = div - f / alpha
        grad = torch_gradient(v)
        del v
        p = (p + tau * grad) / (1 + tau * torch_module(grad))
        del grad
        div = torch_divergence(p)
    return f - alpha * div


def pos(x_pos: torch.Tensor):
    """
    Compute pos(x)

    Parameters
    ----------
    x : array

    Return
    ------
    pos_x : array
    """
    test = torch.nn.functional.relu(x_pos)
    return test


def CP_denoise(f: torch.Tensor, alpha: float = torch.tensor(1e3), iters: int = 20):
    """
    Denoise the image f with Chambolle Pock algorithm

    Parameters
    ----------
    f : array
        noised image, poisson noise
    alpha : float
            TV parameter
    N : int
        number max of iteration

    Return
    ------
    u_bar : array
            denoised image
    """

    # Initialization
    tau = 1.0 / (f.shape[1] + 4)
    sigma1 = 1.0 / f.shape[0]
    sigma2 = 0.5
    theta = 1

    u = torch.zeros_like(f)
    u_bar = torch.zeros_like(f)
    p = torch.zeros_like(f)
    q = torch.zeros(f.dim(), *f.shape)

    # Iterations
    for k in range(iters):
        p = 0.5 * (
            1
            + p
            + sigma1 * u_bar
            - torch.sqrt((p + sigma1 * u_bar - 1) ** 2 + 4 * sigma1 * f)
        )
        grad = torch_gradient(u_bar)
        z = q + sigma2 * grad
        q = alpha * z / torch.maximum(alpha, torch_module(z))
        u_pre = u.clone()
        # u = u - tau * p + tau * torch_divergence(q)
        u = u - tau * p + tau * torch_gradient(q)
        u_bar = u + theta * (u - u_pre)
        u_bar = pos(u_bar)

    return u_bar


def TV_dual_denoising(
    f: torch.Tensor,
    s: torch.Tensor,
    alpha: float,
    NTViter: int = 50,
    epsilon: float = 1e-8,
):
    """
    Compute dual denoising, from Maxim2018. Used for Poisson noise

    Parameters
    ----------
    f : np.array
        image to denoise
    s : np.array
        sensibility
    alpha : float
            TV parameter
    NTViter : int
              Number of iteration
    epsilon : float
              Minimum value of the returned image

    Returns
    -------
    f : np.array
        denoised image
    """

    size = [f.dim(), *f.shape]

    # crop sensibility to get rid of zero value
    s[s < 0.1] = 0.1

    if size[0] == 2:
        den = s - 4 * alpha
        den[den < 0] = torch.min(s)
        Lh = 8 * alpha**2 * s * f / den**2
    elif size[0] == 3:
        den = s - 6 * alpha
        den[den < 0] = torch.min(s)
        Lh = 12 * alpha**2 * s * f / den**2
    else:
        print("Something is wrong, good luck !")

    tau = 0.9 * alpha * div_zer(torch.ones_like(Lh), Lh)

    # Compute phi
    phi = torch.zeros(size, dtype=torch.float32)
    phi_ = torch.zeros(size, dtype=torch.float32)
    tTV = 1

    for k in range(NTViter):
        z = pos(div_zer(s * f, s + alpha * torch_divergence(phi)))
        phi__ = phi - tau * torch_gradient(z)
        phi__ = phi__ / torch.maximum(torch_module(phi__), torch.ones_like(phi__))

        # FISTA
        tTV_ = (1 + sqrt(1 + 4 * tTV**2)) * 0.5
        phi = phi__ + (tTV - 1) / tTV_ * (phi__ - phi_)
        tTV = tTV_
        phi_ = phi__

    f = div_zer(f, 1 + alpha * torch_divergence(phi) / s)
    # Ensure we don't return negative values
    f[f < epsilon] = epsilon

    return f


def div_zer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Both matrix should have the same size"
    new = torch.zeros_like(a)
    mask = b > 0
    new[mask] = a[mask] / b[mask]
    return new
