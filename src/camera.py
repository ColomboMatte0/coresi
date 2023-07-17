from enum import StrEnum
from logging import getLogger
from typing import Union

from point import Point

logger = getLogger("__main__." + __name__)


class DetectorType(StrEnum):

    SCA = "scatterer"
    ABS = "absorber"


class Camera(object):
    """docstring for Camera"""

    def __init__(self, attrs: dict, position: dict):
        super(Camera, self).__init__()
        self.sca_layers = self.setup_scatterers(attrs)
        logger.debug("sca layers list" + str([str(layer) for layer in self.sca_layers]))
        self.abs_layers = self.setup_absorbers(attrs)
        logger.debug("abs layers list" + str([str(layer) for layer in self.abs_layers]))

        # Define the center of the scatterer (as if the layers would contained in a box)
        # The center is between the first and last layer

        # TODO: The camera might be translated, the center might not be x: 0 and
        # y:0
        self.sca_centre = Point(
            0.0, 0.0, (self.sca_layers[-1].center.z + self.sca_layers[0].center.z) / 2
        )
        logger.debug("sca.center.z" + str(self.sca_centre))

        # Dimensions of the scatterer box
        self.sca_dim = Point(
            x=self.sca_layers[0].dim.x,
            y=self.sca_layers[0].dim.y,
            z=self.sca_layers[0].center.z
            - self.sca_layers[-1].center.z
            + self.sca_layers[0].dim.z,
        )
        logger.debug("sca dim" + str(self.sca_dim))

        self.origin = Point(*position["frame_origin"])
        self.Ox = Point(*position["Ox"]).normalized()
        self.Oy = Point(*position["Oy"]).normalized()
        self.Oz = Point(*position["Oz"]).normalized()

    def setup_scatterers(self, attrs: dict) -> list["Layer"]:
        """The scatterer has multiple layers. For each layer, create a Layer
        object and sort them by their z axis"""

        self.n_sca_layers = attrs["n_sca_layers"]
        # The scatterer is composed of multiple layers
        return list(
            sorted(
                [
                    Layer(attrs[f"sca_layer_{idx}"], DetectorType.SCA)
                    for idx in range(self.n_sca_layers)
                ],
                key=lambda layer: layer.center.z,
                reverse=True,
            )
        )

    def setup_absorbers(self, attrs: dict) -> list[Union["Layer", None]]:
        """Their might be multiple absorbers to improve the detection
        accuracy. Create a Layer object per absorber"""

        # The index of the absorber layer roughly defines its position relative to the
        # others. Therefore, if an absorber layer is missing, add None in the
        # list to avoid breaking the convention
        self.n_abs_layers = attrs["n_absorbers"]
        abs_layers: list["Layer" | None] = []
        for idx in range(self.n_abs_layers):
            if attrs[f"abs_layer_{idx}"] is None:
                abs_layers.append(None)
            else:
                abs_layers.append(Layer(attrs[f"abs_layer_{idx}"], DetectorType.ABS))

        return abs_layers


def setup_cameras(config_cameras: dict) -> list[Camera]:
    """docstring for setup_cameras"""
    cameras = [
        Camera(
            config_cameras["common_attributes"],
            config_cameras[f"position_{camera_idx}"],
        )
        for camera_idx in range(int(config_cameras["n_cameras"]))
    ]
    logger.info(f"Got {str(len(cameras))} cameras")
    return cameras


class Layer(object):
    """A Layer of an absorber or a scatterer. A Layer is defined by its size,
    its center and its type. Keep track of the type for the events containement
    tests."""

    def __init__(self, attrs: dict, detector_type: DetectorType):
        super(Layer, self).__init__()
        # Center is the z axis because scaterers are aligned in x and y
        self.center = Point(*attrs["center"])
        self.dim = Point(*attrs["size"])
        self.detector_type = detector_type

    def __str__(self) -> str:
        return f"center: {self.center}, dim: {self.dim}"
