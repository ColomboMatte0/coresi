import argparse
import logging
import socket
import sys
import time
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from coresi.camera import setup_cameras
from coresi.data import read_data_file
from coresi.mlem import LM_MLEM
from coresi.simulation import simulate

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

args = parser.parse_args()


try:
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
except IOError as e:
    print(f"Failed to open the configuration file: {e}")
    sys.exit(1)


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

logger = logging.getLogger("CORESI")
logger.info(f"Starting job {job_name} on {socket.gethostname()}")
logger.info(f"Read configuration file {args.config}")


# Setup the cameras' list according to their characteristics
cameras = setup_cameras(config["cameras"])

checkpoint_dir = Path(config["lm_mlem"]["checkpoint_dir"])
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def run():
    start = time.time()
    if args.simulation:
        lines = simulate(
            config["simulation"]["phantom"],
            config,
            cameras,
            config["simulation"]["n_events"],
            config["E0"][0],
            visualize_generated_source=config["simulation"][
                "visualize_generated_source"
            ],
            angle_threshold=config["simulation"]["angle_threshold"],
        )
        with open(config["simulation"]["output_file"], "w") as fh:
            fh.write(lines)
        sys.exit(0)
    if args.sensitivity:
        mlem = LM_MLEM(
            config["lm_mlem"],
            config["volume"],
            cameras,
            args.config.name.split(".")[0],
            config["E0"],
            config["energy_threshold"],
        )
        _ = LM_MLEM.compute_sensitivity(
            config["E0"],
            config["volume"],
            cameras,
            mlem.SM_line,
            config["lm_mlem"],
            checkpoint_dir,
        )
        sys.exit(0)

    mlem = LM_MLEM(
        config["lm_mlem"],
        config["volume"],
        cameras,
        args.config.name.split(".")[0],
        config["E0"],
        config["energy_threshold"],
    )
    mlem.init_sensitiviy(config["lm_mlem"], checkpoint_dir)

    logger.info(f"Processing {config['data_file']}")

    # Process events from the data file and associate them with the cameras
    events = read_data_file(
        Path(config["data_file"]),
        n_events=config["n_events"],
        E0=config["E0"],
        cameras=cameras,
        # Needed to remove events with energy outside of a given range
        remove_out_of_range_energies=config["remove_out_of_range_energies"],
        energy_range=config["energy_range"],
        start_position=config["starts_at"],
        tol=config["energy_threshold"],
        # Used to determine if hit is in volume
        volume_config=config["volume"],
    )

    logger.info(f"Took {time.time() - start} ms to read the data")

    # Reinitialize the timer for MLEM
    start = time.time()

    logger.info("Doing MLEM")

    result = mlem.run(
        events,
        config["lm_mlem"]["last_iter"],
        config["lm_mlem"]["first_iter"],
        config["lm_mlem"]["save_every"],
        checkpoint_dir,
    )

    logger.info(f"Took {time.time() - start} ms for MLEM")

    if args.display:
        for e in range(len(config["E0"])):
            result.display_z(energy=e, title=f"{str(config['E0'][e])} keV")
    plt.show()


if __name__ == "__main__":
    run()
