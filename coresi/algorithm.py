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
import torchvision.transforms.functional as TF

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
        data_config: dict,
        E_0: float,
        device: torch.device
    ):
        super(Algorithm, self).__init__()

        self.device = device
        self.config_algo = config_algo
        self.data_config = data_config
        
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
        self.create_save_dir()
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
                
                # result.values = torch.where(
                #     result.mask,
                #     (result.values / self.sensitivity.values) * next_result.values,
                #     torch.tensor(0.0, device=self.device)
                # )
                result.values = (result.values / self.sensitivity.values) * next_result.values

                next_result.values.zero_()

            if iter % self.config_algo["save_every"] == 0 or iter == self.config_algo["last_iter"]:
                with h5py.File(self.save_reco_dir / f"iter.{iter}.h5", "w") as f:
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

        self.sensitivity.values = self.sensitivity.values /self.config_algo["OSEM_n_subsets"]

    def compute_sensitivity(self,
        constants: dict,
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
            

        elif config_sens["sensitivity_model"] == "fake_list_mode":
            logger.info(f"Computing sensitivity with model:  fake list-mode with attenuation:")

            sensitivity_models.list_mode_sensitivity(
                cameras, 
                E_0, 
                torch_device,
                config_sens["sub_N"],
                config_sens["sub_E"],
                config_sens["output_hdf5"]
            )
            
            # Load generated events and backproject into sensitivity image
            logger.info(f"Loading events from {config_sens['output_hdf5']}")
            
            # Create Events object from the synthetic list-mode data
            sens_events = Events(config_sens["output_hdf5"],
                                constants,
                                torch_device,
                                cameras,
                                None,
                                E_0,
                                0.0)
            
            logger.info(f"Backprojecting {sens_events.n_events:,} events into sensitivity image...")
            sensitivity.values.zero_()
            
            all_indices = torch.arange(sens_events.n_events, device=torch_device)
            
            batch_size = self.config_algo["batch_size"]
            log_every = max(1, sens_events.n_events // (10 * batch_size))  # Log ~10 times
            
            for batch_idx, batch_start in enumerate(range(0, sens_events.n_events, batch_size)):
                batch_end = min(batch_start + batch_size, sens_events.n_events)
                batch_indices = all_indices[batch_start:batch_end]
                
                batch_lines = self.SM_line(sens_events, batch_indices)
                sensitivity.values += batch_lines.sum(dim=0)
                
                # Progress logging
                if (batch_idx % log_every == 0) or (batch_end == sens_events.n_events):
                    progress_pct = 100 * batch_end / sens_events.n_events
                    logger.info(f"Backprojection: {batch_end:,}/{sens_events.n_events:,} events ({progress_pct:.1f}%)")
            
            # Rotate sensitivity over all cameras
            logger.info(f"Rotating sensitivity image over {len(cameras)} cameras...")
            tmp_zxy = sensitivity.values.permute(2, 0, 1)  # from (x,y,z) to (z,x,y)
            rotated_sum = torch.zeros_like(sensitivity.values)
            for i in range(len(cameras)):
                angle_deg = i * (360 / len(cameras))
                tmp = TF.rotate(tmp_zxy, angle=angle_deg, interpolation=TF.InterpolationMode.BILINEAR)
                rotated_sum += tmp.permute(1, 2, 0)  # back to (x,y,z)
            sensitivity.values = rotated_sum
            
            # Normalize sensitivity
            sensitivity.values[~sensitivity.mask] = 0.0
            voldim = volume_config["n_voxels"][0] * volume_config["n_voxels"][1] * volume_config["n_voxels"][2]
            sensitivity.values = (sensitivity.values / torch.linalg.norm(sensitivity.values)) * voldim
            sensitivity.values[~sensitivity.mask] = 1.0
            
            logger.info("Sensitivity backprojection complete")

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
                + "_shape_" + str(volume_config["n_voxels"][0])+"_"+str(volume_config["n_voxels"][1])+"_"+str(volume_config["n_voxels"][2])
                + "_subN_" + str(config_sens["sub_N"][0])+"_"+str(config_sens["sub_N"][1])+"_"+str(config_sens["sub_N"][2])
            + "_Attn_" + str(config_sens["include_attenuation"])
            + ".pth"
        )
            
        save_path = Path(config_sens['save_dir']) / sens_filename
        logger.info(f"Sensitivity done, saving to {str(save_path)}"
        )
        torch.save(sensitivity.values.cpu(), save_path)
        return sensitivity.values

    def create_save_dir(self) -> None:
        # Extract LM filename without path, extension, and "_LM" suffix
        lm_file = Path(self.data_config["file_name"]).stem
        if lm_file.endswith("_LM"):
            lm_file = lm_file[:-3]
        
        if self.config_algo.get("save_file", None) is None:
            # Create parameter string for folder name
            img_shape = f"{self.config_volume['n_voxels'][0]}x{self.config_volume['n_voxels'][1]}x{self.config_volume['n_voxels'][2]}"
            params_str = ( f"{img_shape}__"
                f"Ev_filt_{self.data_config['max_ARM_sigma']:.2f}__"
                f"Nev_{self.data_config['n_events']:.0e}__"
                f"osem_Nss{self.config_algo['OSEM_n_subsets']}__"
                f"model_{self.config_algo['cone_thickness']}_{self.config_algo['model']}__"
                f"sens_{int(self.config_algo['use_sensitivity'])}__"
                f"attn_{int(self.config_algo['use_attenuation'])}__"
                f"Eres_{int(self.config_algo.get('energy_resolution',1))}"        
            )
            #TODO fix energy resolution parsing
            
            # Build hierarchical path: recons/{lm_filename}/{params}/
            base_dir = Path(self.config_algo["save_dir"])
            self.save_reco_dir = base_dir / lm_file / params_str
        else:
            self.save_reco_dir = Path(self.config_algo["save_dir"]) / lm_file/self.config_algo["save_file"]
        
        # Check if folder exists and has files
        if self.save_reco_dir.exists() and any(self.save_reco_dir.iterdir()):
            logger.warning(f"Folder {self.save_reco_dir} already exists and contains files")
            response = input("Do you want to overwrite the existing files? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Reconstruction cancelled by user")
                sys.exit(0)
            
            # Clean existing files
            for file in self.save_reco_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete file {file}: {e}")
        
        # Create directory if it doesn't exist
        self.save_reco_dir.mkdir(parents=True, exist_ok=True)