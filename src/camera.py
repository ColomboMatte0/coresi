import sys
from enum import StrEnum
from logging import getLogger
from typing import Union

import numpy as np
import torch
import yaml

from point import Point

torch.set_grad_enabled(False)
logger = getLogger("__main__." + __name__)


class DetectorType(StrEnum):
    SCA = "scatterer"
    ABS = "absorber"


class Material(StrEnum):
    Silicium = "Si"
    BismuthGermaniumOxide = "BGO"
    LanthanumBromide = "LaBr3"


class Camera(object):
    """docstring for Camera"""

    def __init__(self, attrs: dict, position: dict):
        super(Camera, self).__init__()

        self.sca_layers = self.setup_scatterers(attrs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # There are cases where a scatterer or absorber might be composed of
        # layers of multiple materials. This is not handled currently.
        self.sca_material = Material(attrs["sca_material"])
        self.avogadro, self.m_e, self.r_e = self.get_physics_constants()
        constants_material_sca = self.read_constants_material(self.sca_material)
        self.sca_nist = torch.tensor(constants_material_sca.pop("NIST"), device=device)
        self.sca_nist_slice = self.sca_nist[:, 0].contiguous()
        # effective density of electrons, number of electrons per unit mass
        self.sca_n_eff = self.get_n_eff(**constants_material_sca)
        self.sca_density = constants_material_sca["density"]
        logger.debug("sca layers list" + str([str(layer) for layer in self.sca_layers]))
        self.abs_layers = self.setup_absorbers(attrs)
        self.abs_material = Material(attrs["abs_material"])
        constants_material_abs = self.read_constants_material(self.abs_material)
        self.abs_nist = torch.tensor(constants_material_abs.pop("NIST"), device=device)
        self.abs_nist_slice = self.abs_nist[:, 0].contiguous()
        self.abs_n_eff = self.get_n_eff(**constants_material_abs)
        self.abs_density = constants_material_abs["density"]
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
                    Layer(attrs[f"sca_layer_{idx}"], idx, DetectorType.SCA)
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
                abs_layers.append(
                    Layer(attrs[f"abs_layer_{idx}"], idx, DetectorType.ABS)
                )

        return abs_layers

    def get_n_eff(self, eff: float, density: float, moll_mass: float) -> float:
        """effective number density of electrons in mol*cm^-3"""
        return eff * density * self.avogadro / moll_mass

    def read_constants_material(self, material: Material):
        """docstring for read_constants_material"""
        try:
            with open("constants.yaml", "r") as fh:
                return yaml.safe_load(fh)["materials"][material]
        except (IOError, KeyError) as e:
            logger.fatal(f"Failed to load constants for material {material}: {e}")
            sys.exit(1)

    def get_compton_diff_xsection(self, energy: int, cosbeta):
        P = 1.0 / (1.0 + (energy / self.m_e) * (1 - (cosbeta)))
        # ENRIQUE: Why avogadro number is involved, and issue with the units
        # compared to regular KN formula and Enrique's thesis?
        return (
            (np.power(self.r_e, 2) / 2.0)
            * torch.pow(P, 2)
            * (P + 1.0 / P - 1.0 + torch.pow(cosbeta, 2))
        )

    def get_photo_diff_xsection(
        self, energy: int, detector_type: DetectorType
    ) -> float:
        table_index, nist_table = self.get_table_and_index(energy, detector_type)
        if isinstance(energy, float):
            return (
                nist_table[table_index][2]
                + (nist_table[table_index + 1][2] - nist_table[table_index][2])
                * ((energy / 1000) - nist_table[table_index][0])
            ) / (nist_table[table_index + 1][0] - nist_table[table_index][0])
        else:
            return nist_table[table_index, 2] + (
                nist_table[table_index + 1, 2]
                - nist_table[table_index, 2]
                * ((energy / 1000) - nist_table[table_index, 0])
            ) / (nist_table[table_index + 1, 0] - nist_table[table_index, 0])

    def get_pair_diff_xsection(self, energy: int, detector_type: DetectorType) -> float:
        table_index, nist_table = self.get_table_and_index(energy, detector_type)
        if torch.all(energy < 1022):
            return torch.zeros_like(energy)
        if isinstance(energy, float):
            return nist_table[table_index][3] + (
                nist_table[table_index + 1][3] - nist_table[table_index][3]
            ) * ((energy / 1000) - nist_table[table_index][0]) / (
                nist_table[table_index + 1][0] - nist_table[table_index][0]
            )
        else:
            return nist_table[table_index, 3] + (
                nist_table[table_index + 1, 3] - nist_table[table_index, 3]
            ) * ((energy / 1000) - nist_table[table_index, 0]) / (
                nist_table[table_index + 1, 0] - nist_table[table_index, 0]
            )

    def get_total_diff_xsection(
        self, energy: int, detector_type: DetectorType
    ) -> float:
        table_index, nist_table = self.get_table_and_index(energy, detector_type)
        if isinstance(energy, float):
            return nist_table[table_index][4] + (
                nist_table[table_index + 1][4] - nist_table[table_index][4]
            ) * ((energy / 1000) - nist_table[table_index][0]) / (
                nist_table[table_index + 1][0] - nist_table[table_index][0]
            )
        else:
            return nist_table[table_index, 4] + (
                nist_table[table_index + 1, 4] - nist_table[table_index, 4]
            ) * ((energy / 1000) - nist_table[table_index, 0]) / (
                nist_table[table_index + 1, 0] - nist_table[table_index, 0]
            )

    def get_table_and_index(
        self, energy: float, detector_type: DetectorType
    ) -> tuple[int, torch.tensor]:
        # Convert to MeV
        # Divide this way to avoid modify by reference
        energy = energy / 1000
        if detector_type == DetectorType.SCA:
            if (isinstance(energy, float) and energy < self.sca_nist[0][0]) or (
                not isinstance(energy, float) and energy.min() < self.sca_nist[0][0]
            ):
                logger.fatal(
                    f"Table index energy of {str(energy)} below minimum in NIST table"
                )
                sys.exit(1)
            elif (isinstance(energy, float) and energy > self.sca_nist[65][0]) or (
                not isinstance(energy, float) and energy.max() > self.sca_nist[65][0]
            ):
                logger.fatal(
                    f"Table index energy of {str(energy)} above maximum in NIST table = {str(self.sca_nist[65][0])}"
                )
                sys.exit(1)

            return torch.searchsorted(self.sca_nist_slice, energy) - 1, self.sca_nist
        else:
            if (isinstance(energy, float) and energy < self.abs_nist[0][0]) or (
                not isinstance(energy, float) and energy.min() < self.abs_nist[0][0]
            ):
                logger.fatal(
                    f"Table index energy of {str(energy)} below minimum in NIST table"
                )
                sys.exit(1)
            elif (isinstance(energy, float) and energy > self.abs_nist[65][0]) or (
                not isinstance(energy, float) and energy.max() > self.abs_nist[65][0]
            ):
                logger.fatal(
                    f"Table index energy of {str(energy)} above maximum in NIST table = {str(self.abs_nist[65][0])}"
                )
                sys.exit(1)

            return torch.searchsorted(self.abs_nist_slice, energy) - 1, self.abs_nist

    @staticmethod
    def get_physics_constants() -> tuple[float, float, float]:
        try:
            with open("constants.yaml", "r") as fh:
                data = yaml.safe_load(fh)
                return data["avogadro"], data["m_e"], data["r_e"]
        except (IOError, KeyError) as e:
            logger.fatal(f"Failed to load avogadro: {e}")
            sys.exit(1)


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

    def __init__(self, attrs: dict, idx: int, detector_type: DetectorType):
        super(Layer, self).__init__()
        # Center is the z axis because scaterers are aligned in x and y
        self.center = Point(*attrs["center"])
        self.dim = Point(*attrs["size"])
        self.detector_type = detector_type
        if self.detector_type == DetectorType.SCA or (
            self.detector_type == DetectorType.ABS and idx == 0
        ):
            self.thickness = self.dim.z
        elif idx in [1, 2]:
            self.thickness = self.dim.x
        elif idx in [3, 4]:
            self.thickness = self.dim.y

    def __str__(self) -> str:
        return f"center: {self.center}, dim: {self.dim}"
