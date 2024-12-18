# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import json
import os
import sys
from logging import getLogger
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import coresi.sensitivity as sensitivity_models
from coresi.camera import Camera
from coresi.event import Event
from coresi.image import Image
from coresi.system_matrix_model import SM_Model
from coresi.tv import TV_dual_denoising

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)


class LM_MLEM(object):
    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        run_name: str,
        energies: list[int],
        tol: float,
    ):
        super(LM_MLEM, self).__init__()

        self.cone_thickness = config_mlem["cone_thickness"]
        self.tv = config_mlem["tv"]
        self.alpha_tv = config_mlem["alpha_tv"]
        logger.info(f"MLEM config: {json.dumps(config_mlem)}")

        self.n_skipped_events = 0
        self.run_name = run_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m_e = torch.tensor(
            511, dtype=torch.float, device=self.device
        )  # electron mass in EV
        self.n_energies = len(energies)
        if self.n_energies == 0:
            logger.fatal("The configuration file has an empty E0 key")
            sys.exit(1)

        logger.info(
            f"Using device {'cpu (' + str(os.cpu_count()) +' cpu available)' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}"
        )

        self.config_volume = config_volume
        self.line = Image(self.n_energies, self.config_volume)
        sm_line_model = SM_Model(config_mlem, config_volume, cameras, energies, tol)

        if self.cone_thickness == "parallel":
            if self.n_energies > 1 or config_mlem["force_spectral"]:
                # Alias to avoid selecting the right algorithm in the run loop
                self.SM_line = sm_line_model.SM_parallel_thickness_spectral
            else:
                self.SM_line = sm_line_model.SM_parallel_thickness
        elif self.cone_thickness in ["angular", "angular_precise"]:
            if self.n_energies == 1 and not config_mlem["force_spectral"]:
                self.SM_line = sm_line_model.SM_angular_thickness
            elif self.cone_thickness == "angular":
                self.SM_line = sm_line_model.SM_angular_thickness_spectral
            elif self.cone_thickness == "angular_precise":
                self.SM_line = sm_line_model.SM_angular_thickness_spectral_precise

        logger.info(f"Using algorithm {self.SM_line.__name__}")

    def run(
        self,
        events: list[Event],
        last_iter: int,
        first_iter: int = 0,
        save_every: int = 10,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        """docstring for run"""

        if first_iter > last_iter:
            logger.fatal(
                f"The first iteration should be less than the last iteration, first is {first_iter} and last is {last_iter}"
            )
            sys.exit(1)
        # Was lambda in C++ but lambda is a reserved keyword in Python
        result = Image(self.n_energies, self.config_volume, init="ones")

        # Load a checkpoint if necessary
        if first_iter > 0:
            try:
                logger.info(
                    f"The first iteration is set to {str(first_iter)}, trying to load {checkpoint_dir / self.run_name}.iter.{str(first_iter - 1)}.npy"
                )
                checkpoint = torch.load(
                    checkpoint_dir / f"{self.run_name}.iter.{str(first_iter - 1)}.npy"
                )
            except IOError as e:
                logger.fatal(f"The checkpoint could not be loaded: {e}")
                sys.exit(1)

            if checkpoint.shape != result.values.shape:
                logger.fatal(
                    f"The checkpointed volume does not have the same shape as the current volume. Current volume is {str(result.values.shape)} and checkpointed volume  is {str(checkpoint.shape)}"
                )
                sys.exit(1)

            result.values = checkpoint
            # Delete the checkpoint as we no longer use it and this takes quite a
            # bit of memory
            del checkpoint

        # It must be initialized as zero as temporary values are sumed
        next_result = Image(self.n_energies, self.config_volume, init="zeros")

        for iter in range(first_iter, last_iter + 1):
            logger.info(f"Iteration {str(iter)}")
            to_delete = []
            for idx, event in enumerate(events):
                try:
                    # Compute the system matrix line.
                    # iter - first_iter is a hacky trick to make SM_line believe that
                    # the first iter is 0 even if this is not the case. this works
                    # because the iter param is only used to check whether we need to
                    # verify if the cone intersect the voxel i.e. if we are at the
                    # first iteration
                    # This for the case of a first-iter different than 0.
                    line = self.SM_line(event, (iter - first_iter) == 0)
                except ValueError as e:
                    logger.debug(f"Skipping event {event.id} REASON: {e}")
                    # Remove it from the list because we know we don't need to
                    # look at it anymore
                    to_delete.append(idx)
                    continue

                # # Iteration 0 is a simple backprojection
                if iter == 0:
                    next_result.values += line.values
                else:
                    forward_proj = torch.mul(line.values, result.values).sum()
                    next_result.values += line.values / forward_proj

            if len(to_delete) > 0:
                events = np.delete(events, to_delete)
                logger.warning(
                    f"Skipped {str(len(to_delete))} events when computing the system matrix at iteration {str(iter)}"
                )
                self.n_skipped_events = len(to_delete)

            # Do not take sensitivity into account at iteration 0
            if iter == 0:
                result.values = next_result.values
            else:
                result.values = (
                    result.values / self.sensitivity.values * next_result.values
                )
            if self.tv is True:
                for idx in range(self.n_energies):
                    result.values[idx] = TV_dual_denoising(
                        result.values[idx], self.sensitivity.values[idx], self.alpha_tv
                    )

            # It must be re-initialized as zero as temporary values are asumed
            next_result.values = torch.zeros(
                self.n_energies,
                next_result.dim_in_voxels.x,
                next_result.dim_in_voxels.y,
                next_result.dim_in_voxels.z,
                device=self.device,
            )

            if iter % save_every == 0 or iter == last_iter:
                torch.save(
                    result.values,
                    checkpoint_dir / f"{self.run_name}.iter.{str(iter)}.npy",
                )
        return result

    def run_projector(
        self,
        events: list[Event],
        last_iter: int,
        first_iter: int = 0,
        save_every: int = 10,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        """docstring for run"""

        if first_iter > last_iter:
            logger.fatal(
                f"The first iteration should be less than the last iteration, first is {first_iter} and last is {last_iter}"
            )
            sys.exit(1)
        # Was lambda in C++ but lambda is a reserved keyword in Python
        result = Image(self.n_energies, self.config_volume, init="ones")

        # Load a checkpoint if necessary
        if first_iter > 0:
            try:
                logger.info(
                    f"The first iteration is set to {str(first_iter)}, trying to load {checkpoint_dir / self.run_name}.iter.{str(first_iter - 1)}.npy"
                )
                checkpoint = torch.load(
                    checkpoint_dir / f"{self.run_name}.iter.{str(first_iter - 1)}.npy"
                )
            except IOError as e:
                logger.fatal(f"The checkpoint could not be loaded: {e}")
                sys.exit(1)

            if checkpoint.shape != result.values.shape:
                logger.fatal(
                    f"The checkpointed volume does not have the same shape as the current volume. Current volume is {str(result.values.shape)} and checkpointed volume  is {str(checkpoint.shape)}"
                )
                sys.exit(1)

            result.values = checkpoint
            # Delete the checkpoint as we no longer use it and this takes quite a
            # bit of memory
            del checkpoint

        self.events = events
        for iter in range(first_iter, last_iter + 1):
            logger.info(f"Iteration {str(iter)}")
            if iter == 0:
                result.values = self.backward(
                    torch.ones(len(events)), check_valid_events=True
                )
            else:
                proj = self.forward(result.values, check_valid_events=False)
                result.values = result.values * (
                    self.backward(1 / proj, check_valid_events=False)
                    / self.sensitivity.values
                )
            if self.tv is True:
                for idx in range(self.n_energies):
                    result.values[idx] = TV_dual_denoising(
                        result.values[idx], self.sensitivity.values[idx], self.alpha_tv
                    )

            if iter % save_every == 0 or iter == last_iter:
                torch.save(
                    result.values,
                    checkpoint_dir / f"{self.run_name}.iter.{str(iter)}.npy",
                )
        return result

    def forward(
        self, object, subset_idx=None, check_valid_events: bool = True
    ) -> torch.Tensor:
        # There is probably a faster implementation, but I am trying to keep it simple for illustration purposes
        proj = []
        for event in self.events:
            try:
                line = self.SM_line(event, check_valid_events)
            except ValueError as e:
                # Remove it from the list because we know we don't need to
                # look at it anymore
                continue
            proj.append(torch.mul(line.values, object).sum())

        return torch.stack(proj)

    def backward(
        self, y, subset_idx=None, check_valid_events: bool = True
    ) -> torch.Tensor:
        back_proj = torch.zeros_like(self.line.values)
        to_delete = []
        for idx, event in enumerate(self.events):
            try:
                line = self.SM_line(event, check_valid_events=check_valid_events)
                # line.values = line.values
            except ValueError as e:
                logger.debug(f"Skipping event {event.id} REASON: {e}")
                # Remove it from the list because we know we don't need to
                # look at it anymore
                to_delete.append(idx)
                continue
            back_proj += line.values * y[idx]
        if len(to_delete) > 0:
            self.events = np.delete(self.events, to_delete)
            logger.warning(
                f"Skipped {str(len(to_delete))} events when computing the system matrix"
            )
            self.n_skipped_events = len(to_delete)
            # In MLEM, backproject the errors which are 1 / forward proj. the
            # complete equation is sum_i t_ij * proj_i

        return back_proj

    def init_sensitivity(self, config_mlem: dict, checkpoint_dir: Path) -> None:
        self.sensitivity = Image(self.n_energies, self.config_volume, init="ones")
        if (
            config_mlem["sensitivity"]
            and "sensitivity_file" in config_mlem
            and config_mlem["sensitivity_file"] is not None
        ):
            logger.info(
                f"Taking sensitivity from file {config_mlem['sensitivity_file']}"
            )
            # If the file is saved with numpy or CORESI in C++
            if config_mlem["sensitivity_file"].split(".")[-1] in ["npy", "raw"]:
                self.sensitivity.values = torch.from_numpy(
                    np.fromfile(config_mlem["sensitivity_file"])
                ).reshape(self.sensitivity.values.shape)

            # If the file was saved with torch.save
            elif torch.cuda.is_available():
                self.sensitivity.values = torch.load(
                    config_mlem["sensitivity_file"]
                ).to(self.device)
            # If the sensitivity was computed from GPU but no GPU is available
            else:
                self.sensitivity.values = torch.load(
                    config_mlem["sensitivity_file"], map_location=torch.device("cpu")
                )
            logger.info(self.sensitivity.values.size())
        else:
            logger.info("Sensivitiy is disabled, setting it to ones")

    @staticmethod
    def compute_sensitivity(
        energies: list,
        volume_config: dict,
        cameras: list[Camera],
        SM_line: Callable[[Event, bool], Image],
        config_mlem: dict,
        checkpoint_dir: Path,
    ) -> torch.Tensor:
        """docstring for compute sensitivity"""
        # TODO: if interpolate need to increase volume dim and crop after
        # interpolation
        sensitivity = Image(len(energies), volume_config, init="ones")
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
        if config_mlem["sensitivity_model"] == "solid_angle":
            logger.info("Computing sensitivity values using solid angle")
            sensitivity.values = sensitivity_models.block(
                cameras, volume_config, x, y, z
            )
            if len(energies) > 1:
                # With this model the sensitivity is the same for all energies
                sensitivity.values = sensitivity.values.repeat(len(energies), 1, 1, 1)
        elif config_mlem["sensitivity_model"] == "solid_angle_with_attn":
            logger.info("Computing sensitivity values using layers attenuation")
            sensitivity.values = sensitivity_models.attenuation_exp(
                cameras, 
                volume_config, 
                energies, 
                x, y, z
            )
        elif config_mlem["sensitivity_model"] == "like_system_matrix":
            logger.info(
                f"Computing sensitivity values with a Monte Carlo simulation, {config_mlem['model']} and {SM_line.__name__}"
            )
            sensitivity.values = sensitivity_models.sm_like(
                cameras,
                volume_config,
                energies,
                SM_line,
                mc_samples=config_mlem["mc_samples"],
            )

        sens_filename = (
            "sens"
            + (
                "_MC_" + str(config_mlem["mc_samples"])
                if config_mlem["sensitivity_model"] == "like_system_matrix"
                else "_geo"
            )
            +config_mlem["cone_thickness"]
            + ".pth"
        )
        logger.info(
            f"Sensitivity done, saving to {str(checkpoint_dir / sens_filename)}"
        )
        torch.save(sensitivity.values.cpu(), checkpoint_dir / sens_filename)
        return sensitivity.values
