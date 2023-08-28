import sys
from logging import getLogger

import yaml

from camera import Camera
from event import Event
from gpuoptional import array_module
from image import Image

logger = getLogger("__main__." + __name__)


class LM_MLEM(object):
    """docstring for LM_MLEM"""

    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        events: list[Event],
        sensitivity_file: str,
    ):
        super(LM_MLEM, self).__init__()

        self.xp = array_module()

        self.cone_thickness = config_mlem["cone_thickness"]

        self.cameras = cameras
        self.events = events
        constants = self.read_constants("constants.yaml")

        self.config_volume = config_volume
        # TODO: Figure out whether using a sparse (CPU or GPU) matrix is worth it
        self.line = Image(self.config_volume)
        self.sensitivity = Image(self.config_volume, init="ones")
        if sensitivity_file is not None:
            self.sensivity.values = self.xp.load(sensitivity_file)

        if config_mlem["cone_thickness"] == "angular":
            # Parameters related to Doppler broadening as a sum of two Gaussians
            self.a1 = constants["doppler_broadening"]["a1"]
            self.a2 = constants["doppler_broadening"]["a2"]
            self.sigma_beta_1 = (
                constants["doppler_broadening"]["sigma_beta_1"]
                * config_mlem["width_factor"]
            )
            self.sigma_beta_2 = (
                constants["doppler_broadening"]["sigma_beta_2"]
                * config_mlem["width_factor"]
            )
            self.limit_sigma = (
                max([self.sigma_beta_1, self.sigma_beta_2]) * config_mlem["n_sigma"]
            )
            # Alias to avoid redoing the test between angular and parallel in
            # run
            logger.debug("Using angular thickness")
            self.SM_line = self.SM_angular_thickness
        # Parallel thickness
        else:
            self.sigma_beta = (
                self.line.voxel_size.norm2() * config_mlem["width_factor"] / 2
            )
            self.limit_sigma = self.sigma_beta * config_mlem["n_sigma"]
            logger.debug("Using parallel thickness")
            self.SM_line = self.SM_parallel_thickness

        # Skip the Gaussian above n_sigma * Gaussian std

        # Sample points along each volume dimension. use voxel size to center
        # the points on the voxels
        self.x = self.xp.linspace(
            self.line.corner.x + (self.line.voxel_size.x / 2),
            self.line.corner.x + self.line.dim_in_cm.x - (self.line.voxel_size.x / 2),
            self.line.dim_in_voxels.x,
        )
        self.y = self.xp.linspace(
            self.line.corner.y + (self.line.voxel_size.y / 2),
            self.line.corner.y + self.line.dim_in_cm.y - (self.line.voxel_size.y / 2),
            self.line.dim_in_voxels.y,
        )
        self.z = self.xp.linspace(
            self.line.corner.z + (self.line.voxel_size.z / 2),
            self.line.corner.z + self.line.dim_in_cm.z - (self.line.voxel_size.z / 2),
            self.line.dim_in_voxels.z,
        )
        # Used to go through the volume
        self.xx, self.yy, self.zz = self.xp.meshgrid(
            self.x, self.y, self.z, sparse=True, indexing="ij"
        )

    def run(self, last_iter: int, first_iter: int = 0):
        """docstring for run"""
        # Was lambda in C++ but lambda is a reserved keyword in Python
        # TODO: We should be able to load previous results saved in a file
        result = Image(self.config_volume, init="ones")
        # It must be initialized as zero as temporary values are sumed
        next_result = Image(self.config_volume, init="zeros")

        for iter in range(first_iter, last_iter):
            logger.info(f"Iteration {str(iter)}")
            # Temporary test
            print(f"size of events: {str(len(self.events))}")
            skipped_events = 0
            # It must be initialized as zero as temporary values are sumed
            next_result.values = self.xp.zeros(
                (
                    next_result.dim_in_voxels.x,
                    next_result.dim_in_voxels.y,
                    next_result.dim_in_voxels.z,
                )
            )
            to_delete = []
            for event in self.events:
                try:
                    line = self.SM_line(iter, event)
                except ValueError as e:
                    skipped_events += 1
                    logger.warning(f"Skipping event {event.id} REASON: {e}")
                    # Remove it from the list because we known we don't need to
                    # look at it anymore
                    to_delete.append(int(event.id) - 1)
                    continue
                forward_proj = self.xp.vdot(line.values, result.values)
                next_result.values += line.values / forward_proj

            if len(to_delete) > 0:
                self.events = self.xp.delete(self.events, to_delete)
            result.values = result.values / self.sensitivity.values * next_result.values

            if skipped_events > 0:
                logger.warning(
                    f"Skipped {str(skipped_events)} events when computing the system matrix at iteration {str(iter)}"
                )

        return result

    def SM_angular_thickness(self, iter: int, event: Event) -> Image:
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = self.xp.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )

        # delta j, angle from the cone axis to the voxel
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
        self.line.values = self.xp.abs(self.xp.arccos(cos_delta_j) - event.beta)

        # Discard voxels not within the "thick" cone
        self.line.values[self.line.values > self.limit_sigma] = 0

        # If the cone does not intersect with the voxel at all, discard the self.line
        if iter == 0 and not self.xp.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the voxel for event {event.id}"
            )
        mask = self.line.values == 0
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = self.a1 * self.xp.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta_1**2
        ) + self.a2 * self.xp.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta_2**2
        )
        self.line.values = self.line.values * self.xp.abs(
            (
                # Compute the angle between the camera z axis and the voxel
                # "solid angle". theta_j
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
        )

        # lambda / lambda prime
        KN = 1.0 / (1.0 + event.E0 / 511.0 * (1.0 - cos_delta_j))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j**2)

        # Ti is here i.e. the system matrix for an event i
        self.line.values = KN * self.line.values / rho_j**2
        return self.line

    def SM_parallel_thickness(self, iter: int, event: Event) -> Image:
        """docstring for SM_parallel_thickness"""
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = self.xp.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        # delta j, angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
        # and multiply by rhoj to get the distance from
        # the voxels to the cone surface
        self.line.values = rho_j * self.xp.abs(
            event.cosbeta * self.xp.sqrt(1 - cos_delta_j**2)
            - event.sinbeta * cos_delta_j
        )
        # Discard voxels not within the "thick" cone
        self.line.values[self.line.values > self.limit_sigma] = 0

        # If the cone does not intersect with the voxel at all, discard the self.line
        if iter == 0 and not self.xp.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the voxel for event {event.id}"
            )
        mask = self.line.values == 0
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = self.xp.exp(
            -self.line.values[mask] ** 2 * 0.5 / self.sigma_beta**2
        )
        self.line.values = self.line.values * self.xp.abs(
            (
                # Compute the angle between the camera z axis and the voxel
                # "solid angle". theta_j
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
        )

        # lambda / lambda prime
        KN = 1.0 / (1.0 + event.E0 / 511.0 * (1.0 - cos_delta_j))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j**2)

        # Ti is here i.e. the system matrix for an event i

        self.line.values = KN * self.line.values / rho_j**2
        return self.line

    def read_constants(self, constants_filename: str):
        try:
            with open(constants_filename, "r") as fh:
                return yaml.safe_load(fh)
        except IOError as e:
            logger.fatal(f"Failed to open the constants file: {e}")
            sys.exit(1)
