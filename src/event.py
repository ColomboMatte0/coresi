import numpy as np

from camera import Camera
from point import Point


class Event(object):
    def __init__(self, line_idx: int, line: str, E0: float, format: str = "MACACO"):
        """
        :param int line_idx: Used to store the id of the line for debugging
        :param str line: The event line
        :param float E0: The energy of the source. -1 if unknown
        :param str format: The format of the line. Only MACACO is implemented at the moment

        :raises ValueError: If the computed beta of the event is faulty
        """
        super(Event, self).__init__()
        self.E0 = E0
        self.id = line_idx
        if format == "MACACO":
            self.read_macaco(line)

    def read_macaco(self, line: str) -> None:
        data = line.split("\t")
        # We only deal with two interaction events, don't load other fields
        # Convert str fields to floats
        x1, y1, z1, e1, x2, y2, z2, e2 = [
            float(data[idx]) for idx in [2, 3, 4, 5, 7, 8, 9, 10]
        ]

        # Convert to centimeters
        self.V1 = Point(x1 / 10, y1 / 10, z1 / 10)
        self.V2 = Point(x2 / 10, y2 / 10, z2 / 10)
        self.axis_local = self.V1 - self.V2
        self.axis = self.V1 - self.V2

        # Ee is the energy of the electron in a scatterer
        self.Ee = e1
        # Eg is the energy of the scattered gama caught in an absorber
        self.Eg = e2

        # Apply the Compton formula
        # https://en.wikipedia.org/wiki/Compton_scattering
        self.cosbeta = 1.0 - self.Ee * 511.0 / (self.Eg * (self.Eg + self.Ee))
        if self.cosbeta < -1 or self.cosbeta > 1:
            raise ValueError("Invalid cosbeta")
        self.beta = np.arccos(self.cosbeta)

        # Compute E0 if it's not supplied in the configuration file
        if self.E0 == -1:
            self.E0 = self.Ee + self.Eg

        # The K refers to the Klein-Nishima formula used for differential cross-section
        # https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        self.p = self.Eg / self.E0
        self.K = (
            self.p
            * self.p
            * 0.5
            * (self.p + (1.0 / self.p) - 1.0 + self.cosbeta * self.cosbeta)
        )

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
                self.idx_V1, self.detector_type_V1, self.layer_idx_V1 = (
                    idx,
                    where[1],
                    where[2],
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
                self.idx_V2, self.detector_type_V2, self.layer_idx_V2 = (
                    idx,
                    where[1],
                    where[2],
                )
                found = True
                break

        # Raise an exception so that the event is discarded
        if not found:
            raise (ValueError("V2 does not belong in a known camera"))

    def is_hit_in_camera(self, V: Point, camera: Camera) -> tuple[bool, str, int]:
        """Go through the components of the camera and test if the point is
        contained in one of them. If found return True, the detector type and
        the idx of layer within the detector
        """
        V_local = V.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)

        # Index is the layer number
        index = list(range(0, len(camera.sca_layers))) + list(
            range(0, len(camera.abs_layers))
        )
        for idx, layer in zip(index, camera.sca_layers + camera.abs_layers):
            if (
                abs(V_local.x - layer.center.x) <= layer.dim.x / 2
                and abs(V_local.y - layer.center.y) <= layer.dim.y / 2
                and abs(V_local.z - layer.center.z) <= layer.dim.z / 2
            ):
                return True, layer.detector_type, idx

        return False, "", -1

    def __str__(self):
        return str(vars(self))
