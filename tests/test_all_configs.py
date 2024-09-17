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

import git
import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../coresi")

import numpy as np
import torch

from camera import setup_cameras
from data import read_data_file
from image import Image
from mlem import LM_MLEM

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))
test_dir = test_dir / "config_all"


with open(test_dir / "configs.yaml", "r") as fh:
    configs = yaml.safe_load(fh)

repo = git.Repo(search_parent_directories=True)
commit = repo.git.rev_parse("HEAD", short=True)


class MLEM(unittest.TestCase):
    def test_mlem_all(self):
        for config_name, config in configs.items():
            cameras = setup_cameras(config["cameras"])
            events = read_data_file(
                config["data_file"],
                n_events=config["n_events"],
                E0=config["E0"],
                cameras=cameras,
                energy_range=config["energy_range"],
                remove_out_of_range_energies=config["remove_out_of_range_energies"],
                start_position=config["starts_at"],
                volume_config=config["volume"],
            )
            mlem = LM_MLEM(
                config["lm_mlem"],
                config["volume"],
                cameras,
                "test_config",
                config["E0"],
                config["energy_threshold"],
            )
            # Open a tmp directory in which the CORESI results will be stored
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = Path(tmpdirname)
                mlem.init_sensitiviy(config["lm_mlem"], tmpdirname)
                result = mlem.run(
                    events,
                    config["lm_mlem"]["last_iter"],
                    config["lm_mlem"]["first_iter"],
                    config["lm_mlem"]["save_every"],
                    tmpdirname,
                )

            # Open a tmp directory in which the CORESI results will be stored
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = Path(tmpdirname)
                with open(test_dir / (config_name + ".m"), "r") as fh:
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
                        test_dir / ".." / "CORESI",
                        config_path + ".m",
                    ]
                )
                # Sort the CORESI iterations by the iteration number and return the
                # last one
                regex = re.compile(config_path + ".sample0.iter(\d+).bin")
                last_coresi_iter = list(
                    sorted(
                        glob(str(tmpdirname / "*iter*.bin")),
                        key=lambda item: int(regex.search(item).group(1)),
                    )
                )[-1]

                result_cpp = Image(len(config["E0"]), config["volume"], init="ones")
                result_cpp.values = torch.from_numpy(
                    np.fromfile(last_coresi_iter)
                    .reshape(result.values.shape)
                    .transpose(-4, -2, -3, -1)
                )
                result.save_all(config_name, config, commit=commit)
                result_cpp.save_all(
                    config_name + "_cpp", config, cpp=True, commit=commit
                )


if __name__ == "__main__":
    unittest.main()
