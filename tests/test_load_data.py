# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from camera import setup_cameras
from data import read_data_file
from event import Event

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))

with open(test_dir / "test_config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])


class LoadData(unittest.TestCase):
    maxDiff = None

    def test_read_data_file(self):
        events = read_data_file(
            config["data_file"],
            n_events=-1,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )

        self.assertEqual(len(events), 19745, "Wrong number of events")

        events = read_data_file(
            config["data_file"],
            n_events=2,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=24,
        )

        self.assertEqual(len(events), 2, "Wrong number of events")

        events = read_data_file(
            config["data_file"],
            n_events=1,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )

        self.assertEqual(len(events), 1, "Wrong number of events")

        with self.assertRaises(ValueError):
            with open(
                config["data_file"],
                "r",
            ) as data_fh:
                for line_n, line in enumerate(data_fh):
                    if line_n == 13:
                        Event(line_n, line, [-1])

    def test_dat_data(self):
        events = read_data_file(
            config["data_file"],
            n_events=26,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )
        self.assertEqual(
            [event.E0 for event in events],
            [
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
                140,
            ],
            "Wrong E0",
        )

        self.assertEqual(
            [event.beta for event in events],
            [
                0.4182568122851019,
                0.22384966573292128,
                0.7147128323209218,
                0.5999540765928002,
                0.8808115944597691,
                0.4786858195119354,
                0.5784773466892846,
                0.6313589367865102,
                0.737791223739845,
                0.620778218363845,
                0.3151080753671737,
                1.1298679542070034,
                1.3575048697602667,
                0.13228955506936058,
                0.1595109919444389,
                0.34044360029890564,
                0.9349821353202604,
                0.8969974280016303,
                0.21199406475599475,
                0.5545564102499143,
                0.43385796317618286,
                0.23882414520406095,
                0.41662245503270073,
                0.9874727074051177,
            ],
            "Wrong beta",
        )


if __name__ == "__main__":
    unittest.main()
