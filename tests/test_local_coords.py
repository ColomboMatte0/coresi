# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
#
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from camera import DetectorType, setup_cameras
from data import read_data_file
from event import Event

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))

with open(test_dir / "test_config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))


class LoadData(unittest.TestCase):
    def test_camera_index_and_norm(self):
        events = read_data_file(
            config["data_file"],
            n_events=1,
            E0=[-1],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=26,
        )
        self.assertEqual(events[0].idx_V1, 0, "Wrong camera for V1")
        self.assertEqual(events[0].idx_V2, 0, "Wrong camera for V2")
        self.assertEqual(
            events[0].detector_type_V1, DetectorType.SCA, "Wrong detector type for V1"
        )
        self.assertEqual(
            events[0].detector_type_V2, DetectorType.ABS, "Wrong detector type for V2"
        )
        camera = cameras[events[0].idx_V1]
        self.assertEqual(
            events[0]
            .V1.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .x,
            1.73403,
            "Wrong x for V1",
        )
        self.assertEqual(
            events[0]
            .V1.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .y,
            2.47067,
            "Wrong y for V1",
        )
        self.assertEqual(
            events[0]
            .V1.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .z,
            -14.081100000000001,
            "Wrong z for V1",
        )
        camera = cameras[events[0].idx_V2]
        self.assertEqual(
            events[0]
            .V2.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .x,
            -6.40207,
            "Wrong x for V2",
        )
        self.assertEqual(
            events[0]
            .V2.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .y,
            1.0940699999999999,
            "Wrong y for V2",
        )
        self.assertEqual(
            events[0]
            .V2.get_local_coord(camera.origin, camera.Ox, camera.Oy, camera.Oz)
            .z,
            -29.5636,
            "Wrong z for V2",
        )

        with self.assertRaises(ValueError) as cm:
            with open(
                test_dir / "faulty_test.dat",
                "r",
            ) as data_fh:
                for line_n, line in enumerate(data_fh):
                    if line_n == 25:
                        event = Event(line_n, line, [-1])
                        event.set_camera_index(cameras)
        self.assertEqual(
            str(cm.exception),
            "V1 does not belong in a known camera",
            "Should say that V1 is not in a known camera",
        )


if __name__ == "__main__":
    unittest.main()
