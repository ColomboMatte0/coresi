import sys
from logging import getLogger
from math import acos, sqrt

import torch

from coresi.camera import Camera, DetectorType
from coresi.point import Point

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)


class Event(object):
    def __init__(
        self,
        line_idx: int,
        line: str,
        E0: list[float],
        volume_center: Point,
        volume_dim: Point,
        format: str = "GATE",
        tol: float = 1e1,
    ):
        """
        :param int line_idx: Used to store the id of the line for debugging
        :param str line: The event line
        :param float E0: The energy of the source. -1 if unknown
        :param str format: The format of the line. Only gate is implemented at the moment

        :raises ValueError: If the computed beta of the event is faulty
        """
        super(Event, self).__init__()
        self.source_E0 = E0
        self.n_energies = len(E0)
        # Keep track of valid energies if spectral
        # This is used as a pseudo boolean flag to skip incompatible energies
        # TODO: rename
        self.xsection = torch.tensor(
            [1.0 for _ in range(self.n_energies)], dtype=torch.float
        )
        self.id = str(line_idx)
        self.tol = tol
        self.volume_center = volume_center
        self.volume_dim = volume_dim
        if format == "GATE":
            self.read_gate_dat_file(line)

    def read_gate_dat_file(self, line: str) -> None:
        data = line.split("\t")
        # We only deal with two interaction events, don't load other fields
        # Convert str fields to floats
        x1, y1, z1, e1, x2, y2, z2, e2 = [
            float(data[idx]) for idx in [2, 3, 4, 5, 7, 8, 9, 10]
        ]

        # Convert to centimeters
        self.V1 = Point(x1 / 10, y1 / 10, z1 / 10)
        self.V2 = Point(x2 / 10, y2 / 10, z2 / 10)
        # Check if the local axis is needed
        # self.axis_local = self.V1 - self.V2
        self.axis = (self.V1 - self.V2).normalized()

        # Ee is the energy of the electron in a scatterer
        self.Ee = e1
        # Eg is the energy of the scattered gama caught in an absorber
        self.Eg = e2

        # Compute E0 if it's not supplied in the configuration file
        if self.source_E0 == [-1] or len(self.source_E0) > 1:
            self.E0 = self.Ee + self.Eg
        else:
            self.E0 = self.source_E0[0]
        if self.volume_intersects_camera(self.V1):
            logger.fatal(
                f"The volume intersects the camera for first hit, check whether the camera configuration matches GATE's, and the volume dimension and position. Event: {line}"
            )
            sys.exit(1)
        if self.volume_intersects_camera(self.V2):
            logger.fatal(
                f"The volume intersects the camera for second hit, check whether the camera configuration matches GATE's, and the volume dimension and position. Event: {line}"
            )
            sys.exit(1)

        if self.n_energies < 2:
            # Apply the Compton formula
            # https://en.wikipedia.org/wiki/Compton_scattering
            # 511 is the electron mass
            self.cosbeta = 1.0 - self.Ee * 511.0 / (self.E0 * (self.E0 - self.Ee))
            if self.cosbeta < -1 or self.cosbeta > 1:
                raise ValueError("Invalid cosbeta")
            self.sinbeta = sqrt(1 - self.cosbeta**2)
            self.beta = acos(self.cosbeta)

        self.energy_bin = self.get_position_energy(self.E0, self.tol)
        self.xsection[0 : self.energy_bin] = 0.0

    def set_camera_index(self, cameras: list[Camera]) -> None:
        """Set the index of the camera where the event has occured"""
        # TODO: determine whether local coords need to be stored for later use
        found = False
        # Go through the cameras. If the event is in a camera, set the id of the
        # camera, the type of detector and the detector layer, then stop the
        # loop
        for idx, camera in enumerate(cameras):
            where = self.is_hit_in_camera(self.V1, camera)
            if where[0] is True:
                # return whether the hit is in an absorber or a scatter and its
                # index
                (
                    self.idx_V1,
                    self.detector_type_V1,
                    self.layer_idx_V1,
                    self.normal_to_layer_V1,
                ) = (
                    idx,
                    where[1],
                    where[2],
                    where[3],
                )
                found = True
                break
        if not found:
            # Raise an exception so that the event is discarded
            raise (ValueError("V1 does not belong in a known camera"))

        found = False

        # Go through the cameras. If the event is in a camera, set the id of the
        # camera, the type of detector and the detector layer, then stop the
        # loop
        for idx, camera in enumerate(cameras):
            where = self.is_hit_in_camera(self.V2, camera)
            if where[0] is True:
                # return whether the hit is in an absorber or a scatter and its
                # index
                (
                    self.idx_V2,
                    self.detector_type_V2,
                    self.layer_idx_V2,
                    self.normal_to_layer_V2,
                ) = (
                    idx,
                    where[1],
                    where[2],
                    where[3],
                )
                found = True
                break

        # Raise an exception so that the event is discarded
        if not found:
            raise (ValueError("V2 does not belong in a known camera"))

    def volume_intersects_camera(self, V: Point) -> bool:
        return (
            abs(V.x - self.volume_center.x) <= self.volume_dim.x / 2
            and abs(V.y - self.volume_center.y) <= self.volume_dim.y / 2
            and abs(V.z - self.volume_center.z) <= self.volume_dim.z / 2
        )

    def is_hit_in_camera(
        self, V: Point, camera: Camera
    ) -> tuple[bool, DetectorType, int, Point]:
        """Go through the components of the camera and test if the point is
        contained in one of them. If found return True, the detector type and
        the idx of layer within the detector
        """
        V_local = V.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)

        # Index is the layer number. It needs to be unique over all the layer
        # list as V1 and V2 may be in an abritrary layer of any detector.
        # Especially important since backscattering odds increases with low
        # energies
        index = list(range(0, len(camera.sca_layers + camera.abs_layers)))
        n_layer_scatterer = len(camera.sca_layers)
        for idx, layer in zip(index, camera.sca_layers + camera.abs_layers):
            if layer is None:
                continue
            if (
                abs(V_local.x - layer.center.x) <= layer.dim.x / 2
                and abs(V_local.y - layer.center.y) <= layer.dim.y / 2
                and abs(V_local.z - layer.center.z) <= layer.dim.z / 2
            ):
                # The normal changes according to the obsorber position.
                # the index of the absorber informs us about the relativre
                # absorber position
                if layer.detector_type == DetectorType.SCA or (
                    layer.detector_type == DetectorType.ABS
                    and (idx - n_layer_scatterer) == 0
                ):
                    return True, layer.detector_type, idx, camera.Oz
                elif (
                    layer.detector_type == DetectorType.ABS
                    and (idx - n_layer_scatterer) == 1
                ):
                    return True, layer.detector_type, idx, -camera.Ox
                elif (
                    layer.detector_type == DetectorType.ABS
                    and (idx - n_layer_scatterer) == 2
                ):
                    return True, layer.detector_type, idx, camera.Ox
                elif (
                    layer.detector_type == DetectorType.ABS
                    and (idx - n_layer_scatterer) == 3
                ):
                    return True, layer.detector_type, idx, -camera.Oy
                elif (
                    layer.detector_type == DetectorType.ABS
                    and (idx - n_layer_scatterer) == 4
                ):
                    return True, layer.detector_type, idx, camera.Oy

        return False, "", -1, Point(0, 0, 0)

    def get_position_energy(self, E0, tol=1e-5) -> int:
        """The goal is identify the energy bin of the measured energy"""
        for idx, energy in enumerate(self.source_E0, start=1):
            if E0 <= energy + tol:
                return idx - 1

        # If E1 + E2 is larger than the maximum energy considered in the
        # configuration
        return self.n_energies - 1

    def __str__(self):
        return str(vars(self))
