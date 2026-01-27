# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import socket
import sys
import time
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
import torch

from coresi.single_layer_camera import setup_single_layer_cameras
from coresi.Events import Events
from coresi.algorithm import Algorithm

parser = argparse.ArgumentParser(
    description="CORESI - Code for Compton camera image reconstruction (default action)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable debug output",
)
parser.add_argument(
    "-c",
    "--config",
    default="config.yaml",
    help="Path to the configuration file",
    type=Path,
)
parser.add_argument(
    "--sensitivity",
    action="store_true",
    help="Compute the sensitivity and quits",
)
parser.add_argument(
    "--simulation",
    action="store_true",
    help="Do a simulation and quit",
)
parser.add_argument(
    "--display",
    action="store_true",
    help="Display the reconstructed image after the reconstruction",
)
parser.add_argument(
    "--device",
    choices=["cuda", "mps", "cpu"],
    default="cpu",
    help="Device to use for computation",
)

args = parser.parse_args()

logger = logging.getLogger("CORESI")

try:
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
except IOError as e:
    logger.error(f"Failed to open the configuration file: {e}")

try:
    with open("constants.yaml", "r") as fh:
        constants = yaml.safe_load(fh)
except IOError as e:
    logger.error(f"Failed to open the constants file: {e}")


job_name = environ["PBS_JOBID"] if "PBS_JOBID" in environ else "local"
log_dir = Path(config["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(
    filename=log_dir / "_".join(["coresi", job_name, str(int(time.time())) + ".log"]),
    mode="w",
)
handlers = (file_handler, logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO if not args.verbose else logging.DEBUG,
    format="[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    handlers=handlers,
)

if args.device== "cuda":
    device_info = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
elif args.device == "mps":
    device_info = f"Apple Silicon GPU (MPS)"
else:
    device_info = f"CPU ({os.cpu_count()} cores available)"
logger.info(f"Using device: {device_info}")


logger.info(f"Starting job {job_name} on {socket.gethostname()}")
logger.info(f"Read configuration file {args.config}")

cameras = setup_single_layer_cameras(config["cameras"])
sens_save_dir = Path(config["sensitivity"]["save_dir"])

def run():

    reco = Algorithm(
        config["lm_algo"],
        config["volume"],
        cameras,
        config["data"],
        config["data"]["E0"],
        args.device,
    )

    if args.sensitivity:
        _ = reco.compute_sensitivity(
            config["data"]["E0"],
            config["volume"],
            cameras,
            config["sensitivity"],
            args.device,
        )
        sys.exit(0)

    reco.init_sensitivity()
    
    events = Events(config["data"],
                    constants,
                    args.device, 
                    cameras)

    result = reco.run_OSEM(events)

    if args.display:
        for e in range(len(config["data"]["E0"])):
            result.display_z(energy=e, title=f"{str(config['data']['E0'][e])} keV")
    plt.show()

if __name__ == "__main__":
    run()

# if __name__ == "__main__":
#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         run()
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
