# SPDX-FileCopyrightText: 2024 Matteo Colombo
# SPDX-License-Identifier: MIT

"""
Single-layer camera model for detectors without separate scatterer/absorber.
Suitable for LSO, CZT, LaBr3, NaI ring detectors in Compton imaging.
"""

import sys
from logging import getLogger
from math import pi
from enum import StrEnum

import numpy as np
import torch
import yaml

from coresi.point import Point

_ = torch.set_grad_enabled(False)
logger = getLogger("CORESI")

class Material(StrEnum):
    LutetiumOxyorthosilicate = "LSO"


class SingleLayerCamera(object):
    """
    Simplified camera model for single-layer detectors (e.g., LSO ring).
    
    Unlike the standard Camera class which assumes scatterer+absorber sandwich,
    this model treats the detector as a single material where both Compton
    scattering and photoelectric absorption can occur.
    
    Key differences from Camera:
    - No separate scatterer/absorber layers
    - Single material for all interactions
    - Simplified geometry (can be extended to multi-head ring)
    - Same physics (Klein-Nishina, NIST cross-sections)
    """

    def __init__(self, comm_attrs: dict, position: dict):
        super(SingleLayerCamera, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Single material for both scattering and absorption
        self.material = Material(comm_attrs["material"])
        self.avogadro, self.m_e, self.r_e = self.get_physics_constants()
        
        # Load material properties
        constants_material = self.read_constants_material(self.material)
        
        self.nist = torch.tensor(constants_material.pop("NIST"), device=self.device)
        self.nist_slice = self.nist[:, 0].contiguous()
        self.n_eff = self.get_n_eff(**constants_material)
        self.density = constants_material["density"]
        
        self.centre = Point(*position["center"])

        logger.debug(f"Detector centre: {self.centre}")
        

        self.dim = Point(*comm_attrs["size"])
        logger.debug(f"Detector dimensions: {self.dim}")
        
        # Reference frame
        self.origin = Point(*comm_attrs["frame_origin"])
        self.Ox = Point(*position["Ox"]).normalized()
        self.Oy = Point(*position["Oy"]).normalized()
        self.Oz = Point(*position["Oz"]).normalized()

    def read_constants_material(self, material: Material) -> dict:
        """docstring for read_constants_material"""
        try:
            with open("constants.yaml", "r") as fh:
                return yaml.safe_load(fh)["materials"][material]
        except (IOError, KeyError) as e:
            logger.fatal(f"Failed to load constants for material {material}: {e}")
            sys.exit(1)
    
    @staticmethod
    def get_physics_constants() -> tuple[float, float, float]:
        try:
            with open("constants.yaml", "r") as fh:
                data = yaml.safe_load(fh)
                return data["avogadro"], data["m_e"], data["r_e"]
        except (IOError, KeyError) as e:
            logger.fatal(f"Failed to load avogadro: {e}")
            sys.exit(1)
    
    def get_n_eff(self, eff: float, density: float, moll_mass: float) -> float:
        """effective number density of electrons in mol*cm^-3"""
        return eff * density * self.avogadro / moll_mass

    def get_incoherent_diff_xsection(
        self, energy: int | torch.Tensor
        ) -> float:
        table_index, nist_table = self.get_table_and_index(energy)
        if isinstance(energy, float):
            return nist_table[table_index][1] + (
                nist_table[table_index + 1][1] - nist_table[table_index][1]
            ) * ((energy / 1000) - nist_table[table_index][0]) / (
                nist_table[table_index + 1][0] - nist_table[table_index][0]
            )
        else:
            return nist_table[table_index, 1] + (
                nist_table[table_index + 1, 1] - nist_table[table_index, 1]
            ) * ((energy / 1000) - nist_table[table_index, 0]) / (
                nist_table[table_index + 1, 0] - nist_table[table_index, 0]
            )
    
    def get_total_diff_xsection(
        self, energy: int | torch.Tensor
    ) -> float:
        table_index, nist_table = self.get_table_and_index(energy)
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
        self, energy: float
    ) -> tuple[int, torch.Tensor]:
        # Convert to MeV
        # Divide this way to avoid modifying by reference
        energy = energy / 1000

        if (isinstance(energy, float) and energy < self.nist[0][0]) or (
            not isinstance(energy, float) and energy.min() < self.nist[0][0]
        ):
            logger.fatal(
                f"Table index energy of {str(energy)} below minimum in NIST table"
            )
            sys.exit(1)
        elif (isinstance(energy, float) and energy > self.nist[49][0]) or (
            not isinstance(energy, float) and energy.max() > self.nist[49][0]
        ):
            logger.fatal(
                f"Table index energy of {str(energy)} above maximum in NIST table = {str(self.nist[49][0])}"
            )
            sys.exit(1)
        return torch.searchsorted(self.nist_slice, energy) - 1, self.nist


def setup_single_layer_cameras(config_cameras: dict) -> list[SingleLayerCamera]:
    """
    Setup single-layer cameras from configuration.
    
    Args:
        config_cameras: Configuration dictionary with camera parameters
    
    Returns:
        List of SingleLayerCamera instances
    """
    cameras = [
        SingleLayerCamera(
            config_cameras["common_attributes"],
            config_cameras[f"position_{camera_idx}"],
        )
        for camera_idx in range(int(config_cameras["n_cameras"]))
    ]
    logger.info(f"Created {len(cameras)} single-layer camera(s) with material {cameras[0].material}")
    return cameras

