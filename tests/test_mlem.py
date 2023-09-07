import os
import re
import subprocess
import sys
import tempfile
import unittest
from glob import glob
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np

from camera import setup_cameras
from data import read_data_file
from mlem import LM_MLEM

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))

coresi_base_filename = "ConfigIECPhantom_CylinderSources_SingleE"

with open(test_dir / "test_config_mlem.yaml", "r") as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])


class MLEM(unittest.TestCase):
    def test_mlem_run(self):
        events = read_data_file(
            test_dir / "test_mlem.dat",
            n_events=10,
            E0=-1,
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
        )
        result = mlem.run(
            config["lm_mlem"]["last_iter"], config["lm_mlem"]["first_iter"]
        )
        self.assertEqual(mlem.n_skipped_events, 46)
        np.testing.assert_array_equal(
            np.load(
                test_dir / "test_mlem.npy",
            ),
            result.values,
        )

    def test_compare_cpp(self):
        events = read_data_file(
            test_dir / "test_mlem.dat",
            n_events=20000,
            E0=-1,
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
        )
        result = mlem.run(
            config["lm_mlem"]["last_iter"], config["lm_mlem"]["first_iter"]
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

            # Sort the CORESI iterations by the iteration number and return the
            # last one
            regex = re.compile(config_path + ".sample0.iter(\d+).bin")
            last_coresi_iter = list(sorted(glob(str(tmpdirname / "*.bin")),
                                      key=lambda item:
                                      int(regex.search(item).group(1))))[-1]


            # Load CORESI results and compare with Python
            np.testing.assert_allclose(
                np.fromfile(last_coresi_iter).reshape(
                    result.values.shape
                )[:, :, 0],
                result.values[:, :, 0].T,
            )


if __name__ == "__main__":
    unittest.main()
