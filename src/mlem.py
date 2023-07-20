import sys
from logging import getLogger
from math import cos, sin, sqrt

from gpuoptional import array_module

import yaml

from camera import Camera
from event import Event
from image import Image
from point import Point

logger = getLogger("__main__." + __name__)


class LM_MLEM(object):
    """docstring for LM_MLEM"""

    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        events: list[Event],
    ):
        super(LM_MLEM, self).__init__()
        self.cone_thickness = config_mlem["cone_thickness"]

        self.cameras = cameras
        self.events = events
        constants = self.read_constants("constants.yaml")

        # Parameters related to Doppler broadening as a sum of two Gaussians
        # TODO: put constants in a separate constants.yaml file
        self.a1 = constants["doppler_broadening"]["a1"]
        self.a2 = constants["doppler_broadening"]["a2"]
        self.sigma_beta_1 = constants["doppler_broadening"]["sigma_beta_1"]
        self.sigma_beta_2 = constants["doppler_broadening"]["sigma_beta_2"]
        self.xp = array_module()

        # Skip the Gaussian above n_sigma * Gaussian std
        self.limit_sigma = (
            max([self.sigma_beta_1, self.sigma_beta_2]) * config_mlem["n_sigma"]
        )
        self.config_volume = config_volume

    def run(self, last_iter: int, first_iter: int = 0):
        """docstring for run"""
        result = Image(self.config_volume)
        for iter in range(first_iter, last_iter):
            skipped_events = 0
            for event in self.events:
                try:
                    line = self.SM_angular_thickness(event)
                except ValueError as e:
                    # TODO: remove event from events for future iteratiosn
                    skipped_events += 1
                    logger.warning(f"Skipping event {line.strip()} REASON: {e}")
                    continue

            logger.warning(
                f"Skipped {str(skipped_events)} events when computing the system matrix"
            )

        return result

    def SM_angular_thickness(self, event: Event) -> Image:
        line = Image(self.config_volume)

        # Sample points along each volume dimension. use voxel size to center
        # the points on the voxels
        x = self.xp.linspace(
            line.corner.x + (line.voxel_size.x / 2),
            line.corner.x + line.dim_in_cm.x - (line.voxel_size.x / 2),
            line.dim_in_voxels.x,
        )
        y = self.xp.linspace(
            line.corner.y + (line.voxel_size.y / 2),
            line.corner.y + line.dim_in_cm.y - (line.voxel_size.y / 2),
            line.dim_in_voxels.y,
        )
        z = self.xp.linspace(
            line.corner.z + (line.voxel_size.z / 2),
            line.corner.z + line.dim_in_cm.z - (line.voxel_size.z / 2),
            line.dim_in_voxels.z,
        )
        # Used to go through the volume
        xx, yy, zz = self.xp.meshgrid(x, y, z, sparse=True, indexing="ij")

        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = self.xp.sqrt(
            (event.V1.x - xx) ** 2 + (event.V1.y - yy) ** 2 + (event.V1.z - zz) ** 2
        )

        # delta j, angle from the cone axis to the voxel
        # xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (xx - event.V1.x)
            + event.axis.y * (yy - event.V1.y)
            + event.axis.z * (zz - event.V1.z)
        ) / rho_j
        # ddelta is the angle from the voxel to the cone border
        line.values = self.xp.abs(self.xp.arccos(cos_delta_j) - event.beta)

        # Discard voxels not within the "thick" cone
        line.values[line.values > self.limit_sigma] = 0

        # If the cone does not intersect with the voxel at all, discard the line
        if not self.xp.any(line.values):
            raise ValueError(
                f"The cone does not intersect the voxel for event {event.id}"
            )
        mask = line.values == 0
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        line.values[mask] = self.a1 * self.xp.exp(
            -line.values[mask] ** 2 * 0.5 / self.sigma_beta_1**2
        ) + self.a2 * self.xp.exp(
            -line.values[mask] ** 2 * 0.5 / self.sigma_beta_2**2
        )
        line.values = line.values * self.xp.abs(
            (
                # Compute the angle between the camera z axis and the voxel
                # "solid angle"
                (camera_V1_Oz.x * (xx - event.V1.x))
                + (camera_V1_Oz.y * (yy - event.V1.y))
                + (camera_V1_Oz.z * (zz - event.V1.z)) / rho_j
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

        # Ti is here i.e. the system matrix for an event
        return KN * line.values / rho_j**2

    def read_constants(self, constants_filename: str):
        try:
            with open(constants_filename, "r") as fh:
                return yaml.safe_load(fh)
        except IOError as e:
            logger.fatal(f"Failed to open the configuration file: {e}")
            sys.exit(1)

    def SM_constant_thickness(self, event: Event) -> Image:
        """docstring for SM_parallel_thickness"""
        return
        # sigmabeta = 0.5 * line.VoxelSize.norm2() * width_factor;
        # double sin_beta =
        # sqrt(1.0 - ev.get_cosbeta() * ev.get_cosbeta());
        # ddelta = fabs(sq * ev.get_cosbeta() - Oj_cone.z * sin_beta);

        # if (ddelta > limit_sigma || ddelta > 1.0)
        # continue; // outside the conical shell
        # rho_j = Oj_cone.norm2();
