# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import json,h5py
import sys
from logging import getLogger
from pathlib import Path
import time

import numpy as np
import torch

import coresi.sensitivity as sensitivity_models
from coresi.single_layer_camera import SingleLayerCamera as Camera
from coresi.Events import Events
from coresi.image import Image
from coresi.system_matrix_model import SM_Model
from coresi.tv import TV_dual_denoising

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)


class Algorithm(object):
    def __init__(
        self,
        config_algo: dict,
        config_volume: dict,
        cameras: list[Camera],
        run_name: str,
        E_0: float,
        device: torch.device
    ):
        super(Algorithm, self).__init__()

        self.device = device
        self.config_algo = config_algo

        self.run_name = run_name
        
        self.m_e = torch.tensor(
            511, dtype=torch.float, device=self.device
        )  # electron mass in EV

        self.config_volume = config_volume

        sm_line_model = SM_Model(self.config_algo, 
                                self.config_volume,
                                cameras,
                                E_0,
                                self.device)
        
        self.SM_line = sm_line_model.SM_line
    
    def run_OSEM(
        self,
        events: Events,
    ):
        logger.info(f"Using events batch size: {self.config_algo['batch_size']} events")
        logger.info(f"OSEM with {self.config_algo['OSEM_n_subsets']} subsets")
        self.clean_save_dir()
        logger.info("Starting Reconstruction")

        result = Image(self.config_volume, self.device, init="ones")        
        next_result = Image(self.config_volume, self.device, init="zeros")

        n_subsets = self.config_algo["OSEM_n_subsets"]
        all_indices = torch.arange(events.n_events, device=events.device)

        start = time.time()
        
        for iter in range(self.config_algo["first_iter"], self.config_algo["last_iter"] + 1):
            logger.info(f"Iteration {iter}")

            for subset_id in range(n_subsets):
                subset_indices = all_indices[subset_id::n_subsets]

                for batch_start in range(0, len(subset_indices), self.config_algo["batch_size"]):
                    batch_end = min(batch_start + self.config_algo["batch_size"], len(subset_indices))
                    batch_indices = subset_indices[batch_start:batch_end]
                    
                    batch_lines_tensor = self.SM_line(events, batch_indices)

                    forward_proj = (batch_lines_tensor * result.values).sum(dim=(1, 2, 3))
                    
                    valid_mask = forward_proj > 0.
                    if (iter == 0) & (not valid_mask.all()):
                        n_invalid = (~valid_mask).sum().item()
                        logger.warning(f"Found {n_invalid} events with zero forward projection in batch, skipping them")

                    backprojection = (batch_lines_tensor[valid_mask] / forward_proj[valid_mask].view(-1, 1, 1, 1)).sum(dim=0)
                                        
                    next_result.values += backprojection
                
                result.values = torch.where(
                    result.mask,
                    (result.values / self.sensitivity.values) * next_result.values,
                    torch.tensor(0.0, device=self.device)
                )

                next_result.values = torch.zeros(
                    next_result.dim_in_voxels.x,
                    next_result.dim_in_voxels.y,
                    next_result.dim_in_voxels.z,
                    device=self.device,
                )

            if iter % self.config_algo["save_every"] == 0 or iter == self.config_algo["last_iter"]:
                with h5py.File(self.save_reco_dir / f"{self.run_name}.iter.{iter}.h5", "w") as f:
                    f.create_dataset("image", data=result.values.cpu().numpy())

        elapsed = time.time() - start
        n_iterations = self.config_algo["last_iter"] - self.config_algo["first_iter"] + 1
        time_per_iter = elapsed / n_iterations if n_iterations > 0 else 0
        logger.info(f"Took {elapsed:.2f} seconds for the reconstruction ({n_iterations} iterations, {time_per_iter:.2f} s/iter)")
        logger.info(f"Reconstruction saved in {self.save_reco_dir}")
        
        return result

    def init_sensitivity(self) -> None:
        self.sensitivity = Image(self.config_volume, init="ones", device=self.device)
        if (self.config_algo["use_sensitivity"]):
            logger.info(f"Taking sensitivity from file {self.config_algo['sensitivity_file']}" )
            # If the file is saved with numpy or CORESI in C++
            if self.config_algo["sensitivity_file"].split(".")[-1] in ["npy", "raw"]:
                self.sensitivity.values = torch.from_numpy(
                    np.fromfile(self.config_algo["sensitivity_file"])
                ).reshape(self.sensitivity.values.shape).to(self.device)

            # Load sensitivity file and move to the correct device
            else:
                self.sensitivity.values = torch.load(
                    self.config_algo["sensitivity_file"],
                    map_location=self.device,
                    weights_only=True
                )
        else:
            logger.info("Sensivitiy is disabled, setting it to ones")

    @staticmethod
    def compute_sensitivity(
        E_0: float,
        volume_config: dict,
        cameras: list[Camera],
        config_sens: dict,
        torch_device: torch.device,
    ) -> torch.Tensor:
        """docstring for compute sensitivity"""
        sensitivity = Image(volume_config,torch_device, init="ones")
        x, y, z = SM_Model.create_mesh_axes(
            [
                sensitivity.corner.x + (sensitivity.voxel_size.x / 2),
                sensitivity.corner.x
                + sensitivity.dim_in_cm.x
                - (sensitivity.voxel_size.x / 2),
            ],
            sensitivity.dim_in_voxels.x,
            [
                sensitivity.corner.y + (sensitivity.voxel_size.y / 2),
                sensitivity.corner.y
                + sensitivity.dim_in_cm.y
                - (sensitivity.voxel_size.y / 2),
            ],
            sensitivity.dim_in_voxels.y,
            [
                sensitivity.corner.z + (sensitivity.voxel_size.z / 2),
                sensitivity.corner.z
                + sensitivity.dim_in_cm.z
                - (sensitivity.voxel_size.z / 2),
            ],
            sensitivity.dim_in_voxels.z,
        )
                
        if config_sens["sensitivity_model"] == "solid_angle":
                logger.info(f"Computing sensitivity with model:  solid angle with attenuation: {config_sens['include_attenuation']}")
                sensitivity.values = sensitivity_models.solid_angle(
                    cameras, 
                    volume_config, 
                    E_0, 
                    torch_device,
                    config_sens["sub_N"],
                    config_sens["include_attenuation"],
                    x, y, z
                )
        else:
            logger.fatal(
                f"Sensitivity model {config_sens['sensitivity_model']} not recognized"
            )
            sys.exit(1)

        if config_sens["file_name"] is not None:
             sens_filename = Path(config_sens["file_name"])
        else:
            sens_filename = Path(
                "sens_"
                + str(config_sens["sensitivity_model"])
                + "_vol_" + str(volume_config["volume_dimensions"][0])+"_"+str(volume_config["volume_dimensions"][1])+"_"+str(volume_config["volume_dimensions"][2])
                + "_subN_" + str(config_sens["sub_N"][0])+"_"+str(config_sens["sub_N"][1])+"_"+str(config_sens["sub_N"][2])
            + "_Attn_" + str(config_sens["include_attenuation"])
            + ".pth"
        )
            
        save_path = Path(config_sens['save_dir']) / sens_filename
        logger.info(f"Sensitivity done, saving to {str(save_path)}"
        )
        torch.save(sensitivity.values.cpu(), save_path)
        return sensitivity.values

    def clean_save_dir(self) -> None:
        self.save_reco_dir = Path(self.config_algo["save_dir"])
        self.save_reco_dir.mkdir(parents=True, exist_ok=True)
        for file in self.save_reco_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete file {file}: {e}")