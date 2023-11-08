import sys
from logging import getLogger
from math import pi
from pathlib import Path

import numpy as np
import torch
import yaml

from camera import Camera, DetectorType
from event import Event
from image import Image

logger = getLogger("__main__." + __name__)
torch.set_grad_enabled(False)


class LM_MLEM(object):
    """docstring for LM_MLEM"""

    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        events: list[Event],
        sensitivity_file: str,
        run_name: str,
        energies: list[int],
        tol: float,
    ):
        super(LM_MLEM, self).__init__()

        self.cone_thickness = config_mlem["cone_thickness"]

        self.compute_theta_j = True
        if config_mlem["model"] == "cos0rho0":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False

            def model(kbl_j):
                return kbl_j

        elif config_mlem["model"] == "cos0rho2":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False

            def model(rho_j):
                return 1 / rho_j**2

        elif config_mlem["model"] == "cos1rho2":

            def model(cos_theta_j, rho_j):
                return abs(cos_theta_j) / rho_j**2

        else:
            logger.fatal(
                f"Model {config_mlem['model']} is not supported, either use cos0rho2 or cos1rho2"
            )
            sys.exit(1)

        self.model = model
        self.cameras = cameras
        self.events = events
        self.run_name = run_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m_e = torch.tensor(
            511, dtype=torch.float, device=self.device
        )  # electron mass in EV
        self.energies = torch.tensor(sorted(energies), device=self.device)
        self.n_energies = len(energies)
        self.tol = tol
        if self.n_energies == 0:
            logger.fatal("The configuration file has an empty E0 key")
            sys.exit(1)

        constants = self.read_constants("constants.yaml")
        logger.info(
            f"Using device {'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}"
        )

        self.config_volume = config_volume
        self.line = Image(self.n_energies, self.config_volume)
        self.sensitivity = Image(self.n_energies, self.config_volume, init="ones")
        if sensitivity_file is not None:
            self.sensitivity.values = torch.from_numpy(
                np.fromfile(sensitivity_file)
            ).reshape(self.sensitivity.values.shape)

        if config_mlem["cone_thickness"] == "parallel":
            self.sigma_beta = (
                self.line.voxel_size.norm2() * config_mlem["width_factor"] / 2
            )
            # Skip the Gaussian above n_sigma * Gaussian std
            self.limit_sigma = self.sigma_beta * config_mlem["n_sigma"]
            if self.n_energies > 1:
                # Alias to avoid selecting the right algorithm in the run loop
                self.SM_line = self.SM_parallel_thickness_spectral
            else:
                self.SM_line = self.SM_parallel_thickness
        elif config_mlem["cone_thickness"] == "angular":
            # TODO: If E0 is unknown, we need to interpolate the values for each
            # event based on the known constants
            # TODO: What we don't have the constants for a given energy?
            # Spectral or not. Same. interpolate
            # We need constants for each energy. Angular resolution modeling
            # is a double Gaussian, its parameters depend on the energy
            self.a1, self.a2, self.sigma_beta_1, self.sigma_beta_2 = [], [], [], []
            for energy in self.energies:
                key = f"energy_{energy}"
                if key in constants["doppler_broadening"]:
                    self.a1.append(constants["doppler_broadening"][key]["a1"])
                    self.a2.append(constants["doppler_broadening"][key]["a2"])
                    self.sigma_beta_1.append(
                        constants["doppler_broadening"][key]["sigma_beta_1"]
                    )
                    self.sigma_beta_2.append(
                        constants["doppler_broadening"][key]["sigma_beta_2"]
                    )
                else:
                    # interpolate
                    # TODO: sort values according to the right key
                    known_energies = [
                        int(key.split("_")[1])
                        for key in constants["doppler_broadening"].keys()
                    ]
                    self.a1.append(
                        np.interp(
                            energy,
                            known_energies,
                            [
                                value["a1"]
                                for value in constants["doppler_broadening"].values()
                            ],
                        )
                    )
                    self.a2.append(
                        np.interp(
                            energy,
                            known_energies,
                            [
                                value["a2"]
                                for value in constants["doppler_broadening"].values()
                            ],
                        )
                    )
                    self.sigma_beta_1.append(
                        np.interp(
                            energy,
                            known_energies,
                            [
                                value["sigma_beta_1"]
                                for value in constants["doppler_broadening"].values()
                            ],
                        )
                    )
                    self.sigma_beta_2.append(
                        np.interp(
                            energy,
                            known_energies,
                            [
                                value["sigma_beta_2"]
                                for value in constants["doppler_broadening"].values()
                            ],
                        )
                    )
                self.limit_sigma = [
                    max([self.sigma_beta_1[idx], self.sigma_beta_2[idx]])
                    * config_mlem["n_sigma"]
                    for idx in range(len(self.sigma_beta_1))
                ]
            if self.n_energies > 1:
                # Alias to avoid selecting the right algorithm in the run loop
                self.SM_line = self.SM_angular_thickness_spectral
            else:
                # If only one energy, tranform the array into a single value
                self.a1 = self.a1[0]
                self.a2 = self.a2[0]
                self.sigma_beta_1 = self.sigma_beta_1[0]
                self.sigma_beta_2 = self.sigma_beta_2[0]
                self.limit_sigma = (
                    max([self.sigma_beta_1, self.sigma_beta_2]) * config_mlem["n_sigma"]
                )
                self.SM_line = self.SM_angular_thickness

        logger.info(f"Using algorithm {self.SM_line.__name__}")

        # Sample points along each volume dimension. use voxel size to center
        # the points on the voxels
        self.x = torch.linspace(
            self.line.corner.x + (self.line.voxel_size.x / 2),
            self.line.corner.x + self.line.dim_in_cm.x - (self.line.voxel_size.x / 2),
            self.line.dim_in_voxels.x,
        )
        self.y = torch.linspace(
            self.line.corner.y + (self.line.voxel_size.y / 2),
            self.line.corner.y + self.line.dim_in_cm.y - (self.line.voxel_size.y / 2),
            self.line.dim_in_voxels.y,
        )
        self.z = torch.linspace(
            self.line.corner.z + (self.line.voxel_size.z / 2),
            self.line.corner.z + self.line.dim_in_cm.z - (self.line.voxel_size.z / 2),
            self.line.dim_in_voxels.z,
        )
        # Used to go through the volume
        self.xx, self.yy, self.zz = np.meshgrid(
            self.x, self.y, self.z, sparse=True, indexing="ij"
        )
        self.xx = torch.from_numpy(self.xx).to(self.device)
        self.yy = torch.from_numpy(self.yy).to(self.device)
        self.zz = torch.from_numpy(self.zz).to(self.device)

    def run(
        self,
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
            for idx, event in enumerate(self.events):
                try:
                    # Compute the system matrix line.
                    # iter - first_iter is a hacky trick to make SM_line believe that
                    # the first iter is 0 even if this is not the case. this works
                    # because the iter param is only used to check whether we need to
                    # verify if the cone intersect the voxel i.e. if we are at the
                    # first iteration
                    line = self.SM_line(iter - first_iter, event, self.energies != [-1])
                except ValueError as e:
                    logger.debug(f"Skipping event {event.id} REASON: {e}")
                    # Remove it from the list because we know we don't need to
                    # look at it anymore
                    to_delete.append(idx)
                    continue

                # # Iteration 0 is a simple backprojection
                # if iter == 0:
                #     next_result.values += line.values
                # else:
                #     # independent dot for the energies
                #     forward_proj = torch.einsum(
                #         "ijkl,ijkl->i", line.values, result.values
                #     ).reshape(-1, 1, 1, 1)
                #     next_result.values += torch.divide(
                #         line.values, forward_proj, where=forward_proj != 0
                #     )
                for energy in range(self.n_energies):
                    # Iteration 0 is a simple backprojection
                    if iter == 0:
                        next_result.values[energy] += line.values[energy]
                    elif event.xsection[energy] > 0.0:
                        forward_proj = torch.mul(
                            line.values[energy], result.values[energy]
                        ).nansum()
                        next_result.values[energy] += line.values[energy] / forward_proj

            if len(to_delete) > 0:
                self.events = np.delete(self.events, to_delete)
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

            # It must be re-initialized as zero as temporary values are sumed
            next_result.values = torch.zeros(
                self.n_energies,
                next_result.dim_in_voxels.x,
                next_result.dim_in_voxels.y,
                next_result.dim_in_voxels.z,
                device=self.device
            )

            if iter % save_every == 0 or iter == last_iter:
                torch.save(
                    result.values,
                    checkpoint_dir / f"{self.run_name}.iter.{str(iter)}.npy",
                )
        return result

    def SM_angular_thickness(self, iter: int, event: Event, known_E0: bool) -> Image:
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        ).unsqueeze(0)

        # delta j is  angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j
        # ddelta is the angle from the voxels to the cone surface
        self.line.values = torch.abs(torch.arccos(cos_delta_j) - event.beta)

        # Discard voxels not within the "thick" cone
        mask = self.line.values > self.limit_sigma
        self.line.values[mask] = 0.0

        # If the cone does not intersect with the volume at all, discard the self.line
        if iter == 0 and not torch.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id}"
            )
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0.0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = self.a1 * torch.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta_1**2
        ) + self.a2 * torch.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta_2**2
        )

        # lambda / lambda prime
        KN = 1.0 / (1.0 + event.E0 / self.m_e * (1.0 - cos_delta_j[mask]))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j[mask] ** 2)

        # Compute the angle between the camera z axis and the voxel
        # "solid angle". theta_j
        if self.compute_theta_j:
            # TODO: Apply mask here?
            cos_theta_j = (
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
            self.line.values *= self.model(cos_theta_j, rho_j)
        else:
            self.line.values *= self.model(rho_j)

        # Ti is here i.e. the system matrix for an event i
        self.line.values[mask] = KN * self.line.values[mask]

        return self.line

    def SM_angular_thickness_spectral(
        self, iter: int, event: Event, known_E0: bool
    ) -> Image:
        if event.energy_bin >= self.n_energies:
            logger.fatal(
                f"The energy bin has not been determinted correctly for event {str(event.id)}"
            )
            sys.exit(1)
        camera = self.cameras[event.idx_V1]

        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        # We use the meshgrid twice to avoid storing the result because it's 3 times the
        # volume size
        cos_theta_j = (
            (event.V1.x - self.xx) * event.normal_to_layer_V1.x
            + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
            + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
        ) / rho_j

        # delta j is the angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # Geometry
        for idx in torch.where(event.xsection > 0)[0]:
            cos_beta = 1.0 - (
                self.m_e
                * event.Ee
                / (self.energies[idx] * (self.energies[idx] - event.Ee))
            )

            if iter == 0 and ((cos_beta < -1) or (cos_beta > 1)):
                event.xsection[idx] = 0.0
                continue

            # ddelta is the angle from the voxels to the cone surface
            self.line.values[idx] = torch.abs(
                torch.arccos(cos_delta_j) - torch.arccos(cos_beta)
            )
            mask_cone = self.line.values[idx] <= self.limit_sigma[idx]

            self.line.values[idx][~mask_cone] = 0.0
            # If the cone does not intersect the volume for a given energy,
            # continue
            if iter == 0 and torch.all(~mask_cone):
                event.xsection[idx] = 0.0
                continue

            # Gauss
            self.line.values[idx][mask_cone] = event.xsection[idx] * (
                self.a1[idx]
                * torch.exp(
                    -self.line.values[idx][mask_cone] ** 2
                    / (2 * self.sigma_beta_1[idx] ** 2)
                )
                + (
                    self.a2[idx]
                    * torch.exp(
                        -self.line.values[idx][mask_cone] ** 2
                        / (2 * self.sigma_beta_2[idx] ** 2)
                    )
                )
            )
            # TODO: This is bad. We repurpose rho j for physic part but rhoj was
            # gloabl so not reinitialized for geometry. We need to either re
            # init it or have a E_gamma & co for all energies
            rho_j = torch.sqrt(
                (event.V1.x - self.xx) ** 2
                + (event.V1.y - self.yy) ** 2
                + (event.V1.z - self.zz) ** 2
            )
            # We use the meshgrid twice to avoid storing the result because it's 3 times the
            # volume size
            cos_theta_j = (
                (event.V1.x - self.xx) * event.normal_to_layer_V1.x
                + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
                + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
            ) / rho_j
            if self.compute_theta_j:
                self.line.values[idx] *= self.model(cos_theta_j, rho_j)
            else:
                self.line.values[idx] *= self.model(rho_j)

            # Repurpose cos_theta_j for E gamma' volume i.e. energy of the
            # scattered photon
            cos_theta_j = self.energies[idx] / (
                1 + (self.energies[idx] / self.m_e) * (1 - cos_theta_j)
            )

            sca_compton_diff_xsection = camera.get_compton_diff_xsection(
                self.energies[idx],
                # ENRIQUE: In Enrique thesis the cosbeta here is
                # the one of the known energy constant
                # for all voxels. Instead we use
                # cos_delta_j which is a matrix
                cos_delta_j[mask_cone],
            )
            # Create a generic x_section variable for different physical
            # effects. This reused to avoid creating new volumes and optimize
            # memory
            x_section = camera.get_photo_diff_xsection(cos_theta_j, DetectorType.ABS)
            # Absorbition total is Photoelectric, partial absorbition is either
            # compton or pair production
            # Photoelectric
            int2Xsect = 0.0
            if abs(self.energies[idx] - event.E0) < self.tol:
                int2Xsect += camera.abs_density * x_section
                x_section = camera.get_pair_diff_xsection(cos_theta_j, DetectorType.ABS)
                # Include double absorption probability after pair creation
                int2Xsect += camera.abs_density * x_section * 2 * self.tol
                x_section = camera.get_photo_diff_xsection(self.m_e, DetectorType.ABS)
                int2Xsect += torch.pow(
                    (
                        1
                        - torch.exp(
                            -x_section
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                            / 2.0
                        )
                    ),
                    2,
                )
            # Compton allowed
            else:
                cos_beta_2 = 1.0 - (
                    self.m_e
                    * event.Eg
                    / (
                        (self.energies[idx] - event.Ee)
                        * (self.energies[idx] - event.Ee - event.Eg)
                    )
                )
                if abs(cos_beta_2) <= 1.0:
                    abs_compton_diff_xsection = camera.get_compton_diff_xsection(
                        self.energies[idx],
                        # ENRIQUE: In Enrique thesis the cosbeta here is
                        # the one of the known energy constant
                        # for all voxels. Instead we use
                        # cos_delta_j which is a matrix
                        cos_beta_2,
                    )

                    int2Xsect = (
                        camera.abs_n_eff
                        * abs_compton_diff_xsection
                        * self.m_e
                        * pi
                        * 4
                        # dE = tol * 2
                        # ENRIQUE: dE is not documented in the original C++ code
                        * self.tol
                        / (torch.pow(self.energies[idx] - event.E0, 2))
                    )

                # Test for pair production
                if (
                    abs(self.energies[idx] - (event.E0 + (2 * self.m_e))) < 2 * self.tol
                ):  # with double escape
                    x_section = camera.get_pair_diff_xsection(
                        cos_theta_j, DetectorType.ABS
                    )

                    int2Xsect += camera.abs_density * x_section
                    # Recompute x_section instead of computing it to save memory
                    x_section = camera.get_photo_diff_xsection(
                        self.m_e, DetectorType.ABS
                    )
                    int2Xsect += self.xp.exp(
                        -x_section
                        * camera.abs_density
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V2
                        ].thickness
                    )

                elif (
                    abs(self.energies[idx] - (event.E0 + self.m_e)) < 2 * self.tol
                ):  # with single escape
                    x_section = camera.get_pair_diff_xsection(
                        cos_theta_j, DetectorType.ABS
                    )
                    int2Xsect += 2 * camera.abs_density * x_section
                    x_section = camera.get_photo_diff_xsection(
                        self.m_e, DetectorType.ABS
                    )
                    int2Xsect += (
                        self.xp.exp(
                            -x_section
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                            / 2.0
                        )
                        # ENRIQUE: why difference of two exponentials. Why not
                        # multiplication of probabilities?. and exp * (1-exp)?
                        - self.xp.exp(
                            -x_section
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                        )
                    )
                if int2Xsect != 0:
                    self.line.values[idx] = self.line.values[idx] * int2Xsect
                else:
                    event.xsection[idx] = 0.0
                    continue

            # ENRIQUE: Different from Enrique's thesis: no V2V1^2 is considered here, the
            # constants, and the probability of escape of the secondary photon (i.e. no interaction)
            # The voxel's size is not taken into account either.
            # The rho_j variable is reused here to make sure no additional
            # volume is created. It now contains the probabilities of first and
            # second interaction
            # First interation is here
            rho_j = (
                camera.sca_n_eff
                * sca_compton_diff_xsection
                * self.m_e
                * 4
                * pi
                / torch.pow(cos_theta_j[mask_cone], 2)
            )

            # Multiply by exponential terms
            # See List mode em reconstruction of Compton Sctter Camera Images in 3D.
            # Wilderman et al. for help and Enrique's thesis.
            # Probabilities for the first photon to reach the middle of the
            # scatterer layer
            rho_j *= torch.exp(
                # zd11
                -camera.get_total_diff_xsection(
                    self.energies[idx], event.detector_type_V1
                )
                # Attenuation coefficient is proportinal to the density
                # (which depends on the medium's physical state)
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                # The thickness. layer_idx_V1 is in a defined layer i.e. it
                # will not be in a None absortber layer as this is checked
                # beforehand
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                # Assume the interaction is the middle of the material in a
                # orthogonal line?
                # TODO: take the incident angle into account?
                / 2
            )

            # zd12
            # Probability of of going out of the scatterer layer and to the
            # absorber
            rho_j *= torch.exp(
                -camera.get_total_diff_xsection(
                    cos_theta_j[mask_cone], event.detector_type_V1
                )
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                / 2
            ) * torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        cos_theta_j[mask_cone], event.detector_type_V2
                    )
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V2
                    ].thickness
                    / 2
                    # why not density here?
                    * (
                        camera.sca_density
                        if event.detector_type_V2 == DetectorType.SCA
                        else camera.abs_density
                    )
                )
            )
            # Attenuation in planes not triggered
            # Before scatterer
            # TODO: If first interaction can be in the absorber. In this case, we
            # would need to compute the number of scatteres the gamma went
            # through without interactions
            # Probability of first and 2nd photon to go through not triggered
            # layers, respectivelly
            if event.layer_idx_V1 > 0:
                rho_j *= torch.exp(
                    -(
                        camera.get_total_diff_xsection(
                            self.energies[idx], event.detector_type_V1
                        )
                        * (
                            camera.sca_density
                            if event.detector_type_V1 == DetectorType.SCA
                            else camera.abs_density
                        )
                        * event.layer_idx_V1
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V1
                        ].thickness
                    )
                )
            # After scatterer
            # TODO: This works only for 1 single absorber below the scatterers
            # But what if the interaction is in an absorber on the sides. If
            # that happens, the scatterers below may needs to be ignored
            rho_j *= torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        cos_theta_j[mask_cone], event.detector_type_V1
                    )
                    * (
                        camera.sca_density
                        if event.detector_type_V1 == DetectorType.SCA
                        else camera.abs_density
                    )
                    * abs(event.layer_idx_V2 - event.layer_idx_V1 - 1)
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V1
                    ].thickness
                )
            )

            self.line.values[idx][mask_cone] = rho_j * self.line.values[idx][mask_cone]

        if iter == 0 and torch.all(event.xsection == 0.0):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id} for all tried energies"
            )

        return self.line

    def SM_parallel_thickness(self, iter: int, event: Event, known_E0: bool) -> Image:
        """docstring for SM_parallel_thickness"""
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        ).unsqueeze(dim=0)
        # delta j is the distance from the voxel to the cone axis
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        # Cosinus of delta_j
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
        # and multiply by rhoj to get the distance from
        # the voxels to the cone surface
        self.line.values = rho_j * torch.abs(
            event.cosbeta * torch.sqrt(1 - cos_delta_j**2)
            - event.sinbeta * cos_delta_j
        )
        # Discard voxels not within the "thick" cone
        mask = self.line.values > self.limit_sigma
        self.line.values[mask] = 0.0

        # If the cone does not intersect with the voxel at all, discard the self.line
        if iter == 0 and not torch.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the voxel for event {event.id}"
            )
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0.0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = torch.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta**2
        )

        if self.compute_theta_j:
            cos_theta_j = (
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
            self.line.values *= self.model(cos_theta_j, rho_j)
        else:
            self.line.values *= self.model(rho_j)

        # lambda / lambda prime
        KN = 1.0 / (1.0 + event.E0 / self.m_e * (1.0 - cos_delta_j[mask]))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j[mask] ** 2)

        # Ti is here i.e. the system matrix for an event i
        self.line.values[mask] = KN * self.line.values[mask]
        return self.line

    def SM_parallel_thickness_spectral(
        self, iter: int, event: Event, known_E0: bool
    ) -> Image:
        """docstring for SM_parallel_thickness_spectral"""
        if event.energy_bin >= self.n_energies:
            logger.fatal(
                f"The energy bin has not been determinted correctly for event {str(event.id)}"
            )
            sys.exit(1)
        camera = self.cameras[event.idx_V1]
        self.line.values = torch.zeros(self.line.values.shape)

        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        if self.compute_theta_j:
            # We use the meshgrid twice to avoid storing the result because it's 3 times the
            # volume size
            cos_theta_j = (
                (event.V1.x - self.xx) * event.normal_to_layer_V1.x
                + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
                + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
            ) / rho_j

        # delta j is the angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # Geometry
        for idx in torch.where(event.xsection > 0)[0]:
            cos_beta = 1.0 - (
                self.m_e
                * event.Ee
                / (self.energies[idx] * (self.energies[idx] - event.Ee))
            )

            if iter == 0 and ((cos_beta < -1) or (cos_beta > 1)):
                event.xsection[idx] = 0.0
                continue
            sin_beta = torch.sqrt(1 - torch.pow(cos_beta, 2))
            # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
            # and multiply by rhoj to get the distance from
            # the voxels to the cone surface
            self.line.values[idx] = rho_j * torch.abs(
                cos_beta * torch.sqrt(1 - cos_delta_j**2) - sin_beta * cos_delta_j
            )
            mask_cone = self.line.values[idx] <= self.limit_sigma

            self.line.values[idx][~mask_cone] = 0.0
            # If the cone does not intersect the volume for a given energy,
            # continue
            if iter == 0 and torch.all(~mask_cone):
                event.xsection[idx] = 0.0
                continue

            sca_compton_diff_xsection = torch.zeros_like(cos_delta_j)
            sca_compton_diff_xsection[mask_cone] = camera.get_compton_diff_xsection(
                self.energies[idx],
                # ENRIQUE: In Enrique thesis the cosbeta here is
                # the one of the known energy constant
                # for all voxels. Instead we use
                # cos_delta_j which is a matrix
                cos_delta_j[mask_cone],
            )

            # ENRIQUE: Different from Enrique's thesis: no V2V1^2 is considered here, the
            # constants, and the probability of escape of the secondary photon (i.e. no interaction)
            # The voxel's size is not taken into account either.
            kbl_j = torch.zeros_like(cos_delta_j)
            kbl_j[mask_cone] = (
                camera.sca_n_eff
                * sca_compton_diff_xsection[mask_cone]
                * self.m_e
                * 2
                * torch.pi
                # What Enrique did - use the E1 from the event
                / torch.pow(self.energies[idx] - event.Ee, 2)
            )

            int2Xsect = 0.0
            photo_x_section = camera.get_photo_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            photo_x_section_m_e = camera.get_photo_diff_xsection(
                self.m_e, DetectorType.ABS
            )

            pair_x_section = camera.get_pair_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            # Absorbition total is Photoelectric, partial absorbition is either
            # compton or pair production
            # Photoelectric
            if abs(self.energies[idx] - event.E0) < self.tol:
                int2Xsect = (
                    camera.abs_density * photo_x_section
                    # Include double absorption probability after pair creation
                    + camera.abs_density
                    * pair_x_section
                    * 2
                    * self.tol
                    * torch.pow(
                        (
                            1
                            - torch.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                        ),
                        2,
                    )
                )
            # Compton allowed
            else:
                cos_beta_2 = 1.0 - (
                    self.m_e
                    * event.Eg
                    / (
                        (self.energies[idx] - event.Ee)
                        * (self.energies[idx] - event.Ee - event.Eg)
                    )
                )
                if abs(cos_beta_2) <= 1.0:
                    abs_compton_diff_xsection = camera.get_compton_diff_xsection(
                        self.energies[idx],
                        # ENRIQUE: In Enrique thesis the cosbeta here is
                        # the one of the known energy constant
                        # for all voxels. Instead we use
                        # cos_delta_j which is a matrix
                        cos_beta_2,
                    )

                    int2Xsect = (
                        camera.abs_n_eff
                        * abs_compton_diff_xsection
                        * self.m_e
                        * pi
                        * 4
                        # dE = tol * 2
                        # ENRIQUE: dE is not documented in the original C++ code
                        * self.tol
                        / (torch.pow(self.energies[idx] - event.E0, 2))
                    )

                # Test for pair production
                if (
                    abs(self.energies[idx] - (event.E0 + (2 * self.m_e))) < 2 * self.tol
                ):  # with double escape
                    int2Xsect += (
                        camera.abs_density
                        * pair_x_section
                        * np.exp(
                            -photo_x_section_m_e
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                        )
                    )

                elif (
                    abs(self.energies[idx] - (event.E0 + self.m_e)) < 2 * self.tol
                ):  # with single escape
                    int2Xsect += (
                        2
                        * camera.abs_density
                        * pair_x_section
                        * (
                            np.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                            # ENRIQUE: why difference of two exponentials. Why not
                            # multiplication of probabilities?. and exp * (1-exp)?
                            - np.exp(
                                --photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                            )
                        )
                    )
            if int2Xsect == 0.0:
                event.xsection[idx] = 0.0
                continue

            # Multiply by exponential terms
            # See List mode em reconstruction of Compton Sctter Camera Images in 3D.
            # Wilderman et al. for help and Enrique's thesis.
            # Probabilities for the first photon to reach the middle of the
            # scatterer layer
            kbl_j[mask_cone] *= torch.exp(
                # zd11
                -camera.get_total_diff_xsection(
                    self.energies[idx], event.detector_type_V1
                )
                # Attenuation coefficient is proportinal to the density
                # (which depends on the medium's physical state)
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                # The thickness. layer_idx_V1 is in a defined layer i.e. it
                # will not be in a None absortber layer as this is checked
                # beforehand
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                # Assume the interaction is the middle of the material in a
                # orthogonal line?
                # TODO: take the incident angle into account?
                / 2
            )

            # zd12
            # Probability of of going out of the scatterer layer and to the
            # absorber
            kbl_j[mask_cone] *= torch.exp(
                -camera.get_total_diff_xsection(
                    self.energies[idx] - event.Ee, event.detector_type_V1
                )
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                / 2
            ) * torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        self.energies[idx] - event.Ee, event.detector_type_V2
                    )
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V2
                    ].thickness
                    / 2
                    * (
                        camera.sca_density
                        if event.detector_type_V2 == DetectorType.SCA
                        else camera.abs_density
                    )
                )
            )
            # Attenuation in planes not triggered
            # Before scatterer
            # TODO: If first interaction can be in the absorber. In this case, we
            # would need to compute the number of scatteres the gamma went
            # through without interactions
            # Probability of first and 2nd photon to go through not triggered
            # layers, respectivelly
            if event.layer_idx_V1 > 0:
                kbl_j[mask_cone] *= torch.exp(
                    -(
                        camera.get_total_diff_xsection(
                            self.energies[idx], event.detector_type_V1
                        )
                        * (
                            camera.sca_density
                            if event.detector_type_V1 == DetectorType.SCA
                            else camera.abs_density
                        )
                        * event.layer_idx_V1
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V1
                        ].thickness
                    )
                )
            # After scatterer
            # TODO: This works only for 1 single absorber below the scatterers
            # But what if the interaction is in an absorber on the sides. If
            # that happens, the scatterers below may needs to be ignored
            kbl_j[mask_cone] *= torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        self.energies[idx] - event.Ee, event.detector_type_V1
                    )
                    * (
                        camera.sca_density
                        if event.detector_type_V1 == DetectorType.SCA
                        else camera.abs_density
                    )
                    * abs(event.layer_idx_V2 - event.layer_idx_V1 - 1)
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V1
                    ].thickness
                )
            )

            if self.compute_theta_j:
                kbl_j = self.model(cos_theta_j, rho_j)
            else:
                kbl_j = self.model(rho_j)

            # Gauss
            self.line.values[idx][mask_cone] = (
                int2Xsect
                * kbl_j[mask_cone]
                * torch.exp(
                    -self.line.values[idx][mask_cone] ** 2 / (2 * self.sigma_beta**2)
                )
            )

        if iter == 0 and torch.all(event.xsection == 0.0):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id} for all tried energies"
            )

        return self.line

    def read_constants(self, constants_filename: str):
        try:
            with open(constants_filename, "r") as fh:
                return yaml.safe_load(fh)
        except IOError as e:
            logger.fatal(f"Failed to open the constants file: {e}")
            sys.exit(1)
