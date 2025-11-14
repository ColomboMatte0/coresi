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

    def __init__(self, attrs: dict, position: dict):
        super(SingleLayerCamera, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Single material for both scattering and absorption
        self.material = attrs["material"]
        
        # Physics constants
        self.avogadro, self.m_e, self.r_e = self.get_physics_constants()
        
        # Load material properties
        constants_material = self.read_constants_material(self.material)
        self.nist = torch.tensor(constants_material.pop("NIST"), device=self.device)
        self.nist_slice = self.nist[:, 0].contiguous()
        self.n_eff = self.get_n_eff(**constants_material)
        self.density = constants_material["density"]
        
        logger.info(f"Single-layer camera with material: {self.material}")
        logger.info(f"Density: {self.density} g/cm³, n_eff: {self.n_eff:.2e} electrons/cm³")
        

        self.centre = Point(*position["center"])

        logger.debug(f"Detector centre: {self.centre}")
        

        self.dim = Point(*attrs["size"])
        logger.debug(f"Detector dimensions: {self.dim}")
        
        # Reference frame
        self.origin = Point(*attrs["frame_origin"])
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
    logger.info(f"Created {len(cameras)} single-layer camera(s)")
    return cameras
