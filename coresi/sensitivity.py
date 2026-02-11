# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT


from logging import getLogger

import torch
import torchvision.transforms.functional as TF
import numpy as np
import h5py

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

    # Put the volume in the coordinate system of the camera
    camera = cameras[0]
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
        # exp_compton = 1 - torch.exp(-mu_compton * hz * torch.sqrt(sq) / D)
        logger.info(f"exp_total - min: {exp_total.min().item():.6f}, max: {exp_total.max().item():.6f}, mean: {exp_total.mean().item():.6f}")

    rect = ((D * torch.pow(sq, -1.5)) * exp_total * exp_compton).sum(dim=(0, 1, 2))  # sum over m, n, and l
    rect = rect.reshape(sensitivity_vol.values.shape)
    tmp_zxy = rect.permute(2,0,1)  # from (x,y,z) to (z,x,y)
    for i in range(len(cameras)):
        angle_deg = i * (360 / len(cameras))
        tmp = TF.rotate(tmp_zxy, angle = angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
    
        sensitivity_vol.values += tmp.permute(1,2,0)  # back to (x,y,z)
    
    sensitivity_vol.values[~sensitivity_vol.mask] = 0.0
    voldim = volume_config["n_voxels"][0] * volume_config["n_voxels"][1] * volume_config["n_voxels"][2]
    sensitivity_vol.values = (sensitivity_vol.values / torch.linalg.norm(sensitivity_vol.values))* voldim
    sensitivity_vol.values[~sensitivity_vol.mask] = 1.0
    return sensitivity_vol.values


def list_mode_sensitivity(
    cameras: list[Camera],
    E_0: float,
    sens_device: torch.device,
    sub_N: list[int],
    sub_E: float,
    output_hdf5: str,
):
    """
    Generate all possible events for sensitivity calculation using list-mode approach.
    
    Event format: (x1, y1, z1, E1, x2, y2, z2, E2, rsector)
    - (x1, y1, z1, E1): first interaction in detector
    - (x2, y2, z2, E2): second interaction in detector  
    - rsector: rotation sector (0 for camera[0])
    
    Args:
        cameras: List of cameras (we use cameras[0] with rsector=0)
        E_0: Initial photon energy (keV)
        sens_device: Device for computation
        sub_N: Subdivision of detector [Nx, Ny, Nz]
        sub_E: Energy sampling step (keV) for energy splitting
        output_hdf5: Output HDF5 file path
    """
    camera = cameras[0]
    rsector = 0
    
    # Create detector grid - use float32 on GPU
    hx = camera.dim.x / sub_N[0]
    hy = camera.dim.y / sub_N[1]
    hz = camera.dim.z / sub_N[2]
    a = -camera.dim.x / 2
    c = -camera.dim.y / 2
    
    # Detector grid coordinates in camera frame - float32 on GPU
    m = torch.arange(sub_N[0], device=sens_device, dtype=torch.float32)
    n = torch.arange(sub_N[1], device=sens_device, dtype=torch.float32)
    l = torch.arange(sub_N[2], device=sens_device, dtype=torch.float32)
    
    mm, nn, ll = torch.meshgrid(m, n, l, indexing='ij')
    vox_x_pos = (a + (mm + 0.5) * hx).flatten()
    vox_y_pos = (c + (nn + 0.5) * hy).flatten()
    vox_z_pos = ((ll + 0.5) * hz).flatten()
    glob_vox_x_pos = (camera.centre[0] + vox_z_pos) * 10  # Convert cm to mm
    glob_vox_y_pos = vox_y_pos * 10  # Convert cm to mm
    glob_vox_z_pos = vox_x_pos * 10  # Convert cm to mm

    N_det = len(glob_vox_x_pos)
    logger.info(f"Detector grid: {N_det} points ({sub_N[0]}x{sub_N[1]}x{sub_N[2]})")
    
    # Generate ALL detector pairs using meshgrid (excluding self-pairs)
    logger.info(f"Generating all detector pairs on GPU...")
    
    det1_x, det2_x = torch.meshgrid(glob_vox_x_pos, glob_vox_x_pos, indexing='ij')
    det1_y, det2_y = torch.meshgrid(glob_vox_y_pos, glob_vox_y_pos, indexing='ij')
    det1_z, det2_z = torch.meshgrid(glob_vox_z_pos, glob_vox_z_pos, indexing='ij')
    
    det1_x = det1_x.flatten()
    det1_y = det1_y.flatten()
    det1_z = det1_z.flatten()
    det2_x = det2_x.flatten()
    det2_y = det2_y.flatten()
    det2_z = det2_z.flatten()
    
    # Remove self-pairs
    valid_pairs = ~((det1_x == det2_x) & (det1_y == det2_y) & (det1_z == det2_z))
    det1_x = det1_x[valid_pairs]
    det1_y = det1_y[valid_pairs]
    det1_z = det1_z[valid_pairs]
    det2_x = det2_x[valid_pairs]
    det2_y = det2_y[valid_pairs]
    det2_z = det2_z[valid_pairs]

    N_pairs = len(det1_x)
    logger.info(f"Total detector pairs: {N_pairs:,} (excluding self-pairs, order matters)")
    
    # Energy splits - float32 on GPU (commented out: flat energy sampling)
    # E_compton_shoulder = (2.0 / (2.0 + E_0 / 511)) * E_0
    # E1_values = torch.arange(sub_E, E_compton_shoulder, sub_E, device=sens_device, dtype=torch.float32)
    # E2_values = E_0 - E1_values
    # N_energy_splits = len(E1_values)
    # 
    # logger.info(f"Energy splits: {N_energy_splits} (step={sub_E} keV)")
    # logger.info(f"E1 range: [{E1_values.min().item():.2f}, {E1_values.max().item():.2f}] keV")
    # logger.info(f"E2 range: [{E2_values.max().item():.2f}, {E2_values.min().item():.2f}] keV")

    # Angle sampling - flat in theta (degrees), compute energies via Compton kinematics
    theta_deg = torch.arange(10, 180.0, sub_E, device=sens_device, dtype=torch.float32)
    theta_rad = torch.deg2rad(theta_deg)
    cos_theta = torch.cos(theta_rad)
    E2_values = E_0 / (1.0 + (E_0 / 511.0) * (1.0 - cos_theta))
    E1_values = E_0 - E2_values
    N_energy_splits = len(E1_values)

    logger.info(f"Angle splits: {N_energy_splits} (step={sub_E} deg)")
    logger.info(f"Theta range: [{theta_deg.min().item():.2f}, {theta_deg.max().item():.2f}] deg")
    logger.info(f"E1 range: [{E1_values.min().item():.2f}, {E1_values.max().item():.2f}] keV")
    logger.info(f"E2 range: [{E2_values.max().item():.2f}, {E2_values.min().item():.2f}] keV")
    
    # Create event arrays using broadcasting
    logger.info("Building events tensor on GPU...")
    
    # Repeat detector pairs for each energy split
    det1_x_all = det1_x.repeat(N_energy_splits)
    det1_y_all = det1_y.repeat(N_energy_splits)
    det1_z_all = det1_z.repeat(N_energy_splits)
    det2_x_all = det2_x.repeat(N_energy_splits)
    det2_y_all = det2_y.repeat(N_energy_splits)
    det2_z_all = det2_z.repeat(N_energy_splits)
    
    # Repeat each energy value N_pairs times
    E1_all = E1_values.repeat_interleave(N_pairs)
    E2_all = E2_values.repeat_interleave(N_pairs)
    
    # Create rsector array (all zeros) - float32 on GPU
    rsector_all = torch.zeros(N_pairs * N_energy_splits, device=sens_device, dtype=torch.float32)
    
    # Stack all columns together
    events = torch.column_stack([
        det1_x_all,
        det1_y_all,
        det1_z_all,
        E1_all,
        det2_x_all,
        det2_y_all,
        det2_z_all,
        E2_all,
        rsector_all
    ])
    
    N_total_events = events.shape[0]
    logger.info(f"Total events: {N_total_events:,}")
    logger.info(f"Events tensor shape: {events.shape}")
    logger.info(f"Memory usage: ~{events.element_size() * events.nelement() / 1e9:.2f} GB")
    
    # Convert to numpy for HDF5 export (move to CPU only at the end)
    logger.info("Converting to numpy and writing HDF5...")
    events_np = events.cpu().numpy()
    
    with h5py.File(output_hdf5, 'w') as f:
        f.create_dataset('listmode', data=events_np, compression='gzip', dtype='float32')
        f.attrs['columns'] = ['x1_mm', 'y1_mm', 'z1_mm', 'e1_keV', 'x2_mm', 'y2_mm', 'z2_mm', 'e2_keV', 'rsectorID']
        f.attrs['E_0'] = float(E_0)
        f.attrs['sub_N'] = sub_N
        f.attrs['sub_E'] = float(sub_E)
        f.attrs['N_pairs'] = int(N_pairs)
        f.attrs['N_energy_splits'] = int(N_energy_splits)
    
    logger.info(f"Synthetic list-mode HDF5 saved to: {output_hdf5} ({N_total_events:,} events)")
