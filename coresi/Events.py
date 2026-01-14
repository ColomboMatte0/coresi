# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import sys
from logging import getLogger
from math import acos, sqrt


import torch
import h5py
import yaml
import time

from coresi.single_layer_camera import SingleLayerCamera as Camera

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)

class Events:
    """
    Memory-efficient and GPU-friendly list-mode dataset for Compton imaging.
    Loads the entire list-mode file once and stores all event fields in tensors.
    """
    def __init__(
            self,
            data_config: dict,
            constants: dict,
            events_device:torch.device,
            cameras: list[Camera]):
        """
        filename: HDF5 with columns [x1,y1,z1,e1, x2,y2,z2,e2, Rs]
        source_E0: list of possible source energies (or None if unknown)
        """
        start = time.time()
        logger.info(f"Processing {data_config['file_name']}")
        self.device = events_device
        with h5py.File(data_config["file_name"], "r") as f:
            raw = torch.tensor(f["listmode"][:data_config["n_events"]], dtype=torch.float32)

        raw = raw.to(self.device)
    
        self.V1 = raw[:, 0:3] / 10    # [N,3]
        self.Ee = raw[:,3]
        self.V2 = raw[:, 4:7] / 10    # [N,3]
        self.Eg = raw[:,7]
        Rs = raw[:,8].long() 

        axis = self.V1 - self.V2                       # [N,3]
        axis = axis / axis.norm(dim=1, keepdim=True)   # normalize
        self.axis = axis                                # [N,3]

        if data_config["E0"] is None:
            self.E0 = self.Ee + self.Eg
        else:
            self.E0 = torch.full_like(self.Ee, data_config["E0"])

        cosbeta = (1.0 - self.Ee * 511.0 / (self.E0 * (self.E0 - self.Ee))).clamp(-1.0, 1.0)
        self.beta = torch.acos(cosbeta)
        self.camera_Oz = self.get_camera_axis(Rs, cameras) 
        self.depth_z = self.compute_depth_z(Rs,cameras)

        const_db = self.get_doppler_constants()
        sigma_doppler = self.compute_sigma_doppler(self.E0, self.Ee, cosbeta, const_db)

        E_resolution = self.get_energy_resolutions(data_config["E0"], constants)
        sigma_energy = self.compute_sigma_energy(self.E0, self.Ee, E_resolution)  

        spatial_resolutions = self.get_spatial_resolutions(constants)

        sigma_total = torch.sqrt(sigma_doppler**2 + sigma_energy**2)
        self.inv_sigma_total_sq = torch.reciprocal(sigma_total**2)
        self.inv_sigma_total = torch.reciprocal(sigma_total)

        if data_config["enable_event_filtering"]:

            logger.info("Filtering out events with ARM > 0.1 rad")
            self.filter_events(sigma_total=sigma_total,max_ARM=0.1)

        self.n_events = len(self.V1) 
        logger.info(f"Loaded {self.n_events:.2e} events from {data_config['file_name']}")

        self.modify_views()

        total_bytes = 0
        for attr in dir(self):
            val = getattr(self, attr)
            if torch.is_tensor(val):
                total_bytes += val.element_size() * val.numel()
        logger.info(f"Memory usage: {total_bytes / 1024 / 1024:.2f} MB")
        logger.info(f"Took {time.time() - start:.2f} seconds to read and preprocess data")

    #TODO load directly from config file?
    def get_camera_axis(self,
                        Rsector: torch.Tensor,
                        cameras: list[Camera]) -> torch.Tensor:
        """
        Returns the camera axis for each event.
        """
        camera_Oz_table = torch.tensor(
        [[cam.Oz.x, cam.Oz.y, cam.Oz.z] for cam in cameras],
        dtype=torch.float32,
        device=self.device
        )  # shape: [n_cameras, 3]

        return camera_Oz_table[Rsector] 
    
    def compute_depth_z(self,
                      Rsector: torch.Tensor,
                      cameras: list[Camera]) -> torch.Tensor:

        camera_center_table = torch.tensor(
        [[cam.centre.x, cam.centre.y, cam.centre.z] for cam in cameras],
        dtype=torch.float32,
        device=self.device
        )  # shape: [n_cameras, 3]
        camera_centers = camera_center_table[Rsector] # shape: [N, 3]
        res = torch.sum( (camera_centers/camera_centers.norm(dim=1, keepdim=True)) *
                          (self.V1 - camera_centers), dim=1)  # shape: [N,]

        # n_cameras = len(cameras)
        # logger.info("="*60)
        # logger.info("DEPTH_Z STATISTICS PER CAMERA")
        # logger.info("="*60)

        # for cam_idx in range(n_cameras):
        #     mask = Rsector == cam_idx
        #     n_events = mask.sum().item()
            
        #     if n_events > 0:
        #         depth_cam = res[mask]
        #         logger.info(f"Camera {cam_idx:2d} ({n_events:6d} events): "
        #                 f"min={depth_cam.min().item():7.2f} cm, "
        #                 f"max={depth_cam.max().item():7.2f} cm, "
        #                 f"mean={depth_cam.mean().item():7.2f} cm")
        
        # # Overall statistics
        # logger.info("-"*60)
        # logger.info(f"Overall     ({len(res):6d} events): "
        #         f"min={res.min().item():7.2f} cm, "
        #         f"max={res.max().item():7.2f} cm, "
        #         f"mean={res.mean().item():7.2f} cm")
        # logger.info("="*60)

        return res


    
    def get_doppler_constants(self) -> float:
        """docstring for read_constants_material"""
        try:
            with open("constants.yaml", "r") as fh:
                return yaml.safe_load(fh)["doppler_broadening"]["C_LSO"]
        except (IOError, KeyError) as e:
            logger.fatal(f"Failed to load constants for doppler broadening: {e}")
            sys.exit(1)
    
    def get_energy_resolutions(self, E0: float, constants: dict) -> float:

        """Returns the energy resolution sigma for each event."""
        key = f"energy_{E0}"
        try:
            return constants["energy_resolutions"][key]["sigma_E"]
        except KeyError:
            logger.fatal(f"Energy resolution for {key} not found in constants.yaml")
    
    def get_spatial_resolutions(self, constants: dict) -> dict:
        """Returns the spatial resolution sigmas."""
        try:
            return constants["spatial_resolutions"]["sigma_planar"], constants["spatial_resolutions"]["sigma_DOI"]
        except KeyError:
            logger.fatal(f"Spatial resolutions not found in constants.yaml")

    def compute_sigma_doppler(self, 
                      E0: float,
                      Edep : torch.Tensor,
                      cos_beta : torch.Tensor,
                      c_db: float)-> torch.Tensor:
        """Compute the Doppler broadening contribution to energy resolution."""
        # TODO why not use Ee directly?
        A = E0 - Edep        
        B = E0 / 511.0       
        sin_theta = torch.sqrt(1.0 - cos_beta**2)
        delta_E_dop = (A / E0) * torch.sqrt(E0**2 + A**2 - 2 * E0 * A * cos_beta) * c_db

        return ((1.0 + B * (1.0 - cos_beta))**2 / (E0 * B * sin_theta)) * delta_E_dop

    def compute_sigma_energy(self,
                             E0: float,
                             Edep: torch.Tensor,
                             sigma_E:float) -> torch.Tensor:
        """Compute the total energy resolution sigma for each event."""
        frac = E0/511
        delta_E_res = E0/((E0-Edep)*torch.sqrt(2* E0 * Edep * frac - (Edep**2) * (1+ 2*frac)))
        return delta_E_res * sigma_E
    
    def filter_events(self, sigma_total,max_ARM: float):
        """Filter out events with ARM greater than max_ARM."""
        mask = sigma_total <= max_ARM
        self.V1 = self.V1[mask].contiguous()
        self.V2 = self.V2[mask].contiguous()
        self.axis = self.axis[mask].contiguous()
        self.Ee = self.Ee[mask].contiguous()
        self.Eg = self.Eg[mask].contiguous()
        self.E0 = self.E0[mask].contiguous()
        self.beta = self.beta[mask].contiguous()
        self.camera_Oz = self.camera_Oz[mask].contiguous()
        self.depth_z = self.depth_z[mask].contiguous()
        self.inv_sigma_total_sq = self.inv_sigma_total_sq[mask].contiguous()
        self.inv_sigma_total = self.inv_sigma_total[mask].contiguous()

    def modify_views(self):
        self.V1 = self.V1.view(-1, 3, 1, 1, 1)       # [N, 3, 1, 1, 1]
        self.V2 = self.V2.view(-1, 3, 1, 1, 1)       # [N, 3, 1, 1, 1]
        self.axis = self.axis.view(-1, 3, 1, 1, 1)   # [N, 3, 1, 1, 1]
        self.Ee = self.Ee.view(-1, 1, 1, 1)          # [N, 1, 1, 1]
        self.Eg = self.Eg.view(-1, 1, 1, 1)          # [N, 1, 1, 1]
        self.E0 = self.E0.view(-1, 1, 1, 1)          # [N, 1, 1, 1]
        self.beta = self.beta.view(-1, 1, 1, 1)          # [N, 1, 1, 1]
        self.camera_Oz = self.camera_Oz.view(-1, 3, 1, 1, 1)   # [N, 3, 1, 1, 1]
        self.depth_z = self.depth_z.view(-1,1,1,1)  # [N,1,1,1]
        self.inv_sigma_total_sq = self.inv_sigma_total_sq.view(-1,1,1,1)
        self.inv_sigma_total = self.inv_sigma_total.view(-1,1,1,1)


    



