# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import os
import re
import subprocess
import sys
import tempfile
import unittest
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
import torch

from camera import setup_cameras
from data import read_data_file
from image import Image
from mlem import LM_MLEM

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))

coresi_base_filename = "ConfigIECPhantom_CylinderSources_SingleE"

with open(test_dir / "test_config_mlem.yaml", "r") as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])


class MLEM(unittest.TestCase):
    def test_mlem_run(self):
        events = read_data_file(
            config["data_file"],
            n_events=10,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            volume_config=config["volume"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )
        mlem = LM_MLEM(
            config["lm_mlem"],
            config["volume"],
            cameras,
            # Supply the sensitivity file if provided
            "test_config",
            config["E0"],
            config["energy_threshold"],
        )
        mlem.init_sensitiviy(config["lm_mlem"], test_dir)
        result = mlem.run(
            events,
            config["lm_mlem"]["last_iter"],
            config["lm_mlem"]["first_iter"],
            config["lm_mlem"]["save_every"],
            test_dir,
        )
        np.testing.assert_array_equal(
            np.load(
                test_dir / "test_mlem_spectral.npy",
            ),
            result.values,
        )
        self.assertEqual(mlem.n_skipped_events, 0)
        events = read_data_file(
            config["data_file"],
            n_events=500,
            E0=[140],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )
        mlem = LM_MLEM(
            config["lm_mlem"],
            config["volume"],
            cameras,
            events,
            # Supply the sensitivity file if provided
            config["sensitivity_file"] if "sensitivity_file" in config else None,
            "test_config",
            [140],
            config["energy_threshold"],
        )

        mlem.init_sensitiviy(config["lm_mlem"], test_dir)
        result = mlem.run(
            config["lm_mlem"]["last_iter"], config["lm_mlem"]["first_iter"]
        )
        np.testing.assert_array_equal(
            np.load(
                test_dir / "test_mlem_mono.npy",
            ),
            result.values,
        )
        self.assertEqual(mlem.n_skipped_events, 5)

    def test_compare_cpp(self):
        events = read_data_file(
            config["data_file"],
            n_events=config["n_events"],
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            volume_config=config["volume"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=config["starts_at"],
        )
        mlem = LM_MLEM(
            config["lm_mlem"],
            config["volume"],
            cameras,
            "test_mlem",
            # Supply the sensitivity file if provided
            config["E0"],
            config["energy_threshold"],
        )
        mlem.init_sensitiviy(config["lm_mlem"], test_dir)

        result = mlem.run(
            events,
            config["lm_mlem"]["last_iter"],
            config["lm_mlem"]["first_iter"],
            config["lm_mlem"]["save_every"],
            test_dir,
        )

        # Open a tmp directory in which the CORESI results will be stored
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            with open(test_dir / (coresi_base_filename + ".m"), "r") as fh:
                # Update the config file with tmp directory name
                coresi_config = fh.read().replace(
                    "RESULT_PLACEHOLDER", f"'{tmpdirname}/'"
                )
            config_path = str(tmpdirname / "tmp_config")
            # CORESI config files must end in .m otherwise it will complain
            with open(config_path + ".m", mode="w") as fp:
                fp.write(coresi_config)
                fp.flush()

            # Run CORESI with the updated tmp cofig file
            subprocess.run(
                [
                    test_dir / "CORESI",
                    config_path + ".m",
                ]
            )

            print(list(glob(str(tmpdirname / "*.bin"))))
            # Sort the CORESI iterations by the iteration number and return the
            # last one
            regex = re.compile(config_path + ".sample0.iter(\d+).bin")
            last_coresi_iter = list(
                sorted(
                    glob(str(tmpdirname / "*.bin")),
                    key=lambda item: int(regex.search(item).group(1)),
                )
            )[-1]

            result_mlem = Image(len(config["E0"]), config["volume"], init="ones")
            result_mlem.values = torch.from_numpy(
                np.fromfile(last_coresi_iter)
                .reshape(result.values.shape)
                .transpose(-4, -2, -3, -1)
            )

            for energy in range(len(config["E0"])):
                result.display_z(
                    energy=energy, title=" energy " + str(config["E0"][energy])
                )
                result_mlem.display_z(
                    energy=energy,
                    title=" energy " + str(config["E0"][energy]) + " c++",
                )
                plt.show()

            # Load CORESI results and compare with Python
            np.testing.assert_allclose(
                result_mlem.values, result.values, atol=1e-10, rtol=1e-3
            )


if __name__ == "__main__":
    unittest.main()
