import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from camera import setup_cameras
from data import read_data_file
from event import Event

with open(
    Path(os.path.dirname(os.path.realpath(__file__)) + "/test_config.yaml"), "r"
) as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])


class LoadData(unittest.TestCase):
    def test_read_data_file(self):
        events = read_data_file(
            Path(os.path.dirname(os.path.realpath(__file__)) + "/test.dat"),
            n_events=-1,
            E0=-1,
            cameras=cameras,
            start_position=14,
        )

        self.assertEqual(len(events), 11, "Wrong number of events")

        events = read_data_file(
            Path(os.path.dirname(os.path.realpath(__file__)) + "/test.dat"),
            n_events=2,
            E0=-1,
            cameras=cameras,
            start_position=14,
        )

        self.assertEqual(len(events), 2, "Wrong number of events")

        events = read_data_file(
            Path(os.path.dirname(os.path.realpath(__file__)) + "/test.dat"),
            n_events=1,
            E0=-1,
            cameras=cameras,
            start_position=0,
        )

        self.assertEqual(len(events), 1, "Wrong number of events")

        with self.assertRaises(ValueError):
            with open(
                os.path.dirname(os.path.realpath(__file__)) + "/test.dat", "r"
            ) as data_fh:
                for line_n, line in enumerate(data_fh):
                    if line_n == 13:
                        Event(line_n, line, -1)

    def test_macaco(self):

        events = read_data_file(
            Path(os.path.dirname(os.path.realpath(__file__)) + "/test.dat"),
            n_events=-1,
            E0=-1,
            cameras=cameras,
            start_position=0,
        )
        self.assertEqual(
            [event.E0 for event in events],
            [
                140.00008,
                140.000485,
                139.99968,
                139.9996,
                139.9999,
                140.0004,
                140.00038,
                140.00014000000002,
                139.99993,
                140.00025,
                140.00041,
                140.0003,
                140.0004,
                140.000337,
                140.000241,
                140.00030999999998,
                140.0,
                140.0003,
                140.00043399999998,
                139.99957,
                139.9997,
                140.00027,
                139.99982,
                139.9995,
            ],
            "Wrong E0",
        )

        self.assertEqual(
            [event.beta for event in events],
            [
                0.4182565668709084,
                0.2238488843404098,
                0.714714596461741,
                0.5999558863741418,
                0.8808123012342127,
                0.47868440364425235,
                0.578475695233285,
                0.631358266335426,
                0.7377916241046397,
                0.6207770435285515,
                0.31510713856486067,
                1.1298650244328245,
                1.3574997629002692,
                0.13228923578335036,
                0.1595107162959119,
                0.3404428331102263,
                0.9349821353202604,
                0.896995259466335,
                0.21199340308118214,
                0.5545581944650999,
                0.4338589197250816,
                0.23882368061438372,
                0.41662300494484755,
                0.9874767882413626,
            ],
            "Wrong beta",
        )


if __name__ == "__main__":
    unittest.main()
