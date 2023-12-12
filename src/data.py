from logging import getLogger
from pathlib import Path

from camera import Camera
from event import Event
from point import Point

logger = getLogger("CORESI")


def read_data_file(
    file_name: Path,
    n_events: int,
    E0: list[float],
    cameras: list[Camera],
    energy_range: list,
    volume_config: dict,
    remove_out_of_range_energies: bool = True,
    start_position: int = 0,
    tol: float = 1e1,
) -> list[Event]:
    events = []
    skipped_events = 0
    volume_center = Point(*volume_config["volume_centre"])
    volume_dim = Point(*volume_config["volume_dimensions"])
    with open(file_name, "r") as data_fh:
        for line_n, line in enumerate(data_fh):
            if line_n >= start_position:
                try:
                    event = Event(
                        line_n, line, E0, volume_center, volume_dim, tol=tol
                    )
                    # This links the event to the camera(s) in which it occurred
                    event.set_camera_index(cameras)
                    events.append(event)
                except ValueError as e:
                    skipped_events += 1
                    logger.debug(f"Skipping event {line.strip()} REASON: {e}")
                    continue
            if n_events != -1 and line_n >= n_events + start_position - 1:
                break

    logger.info(f"Got {str(len(events))} events")

    if skipped_events > 0:
        logger.warning(f"Skipped {str(skipped_events)} events in data parsing")

    n_events = len(events)
    if remove_out_of_range_energies is True:
        events = filter_bad_events(events, energy_range)

    if (n_filtered := n_events - len(events)) > 0:
        logger.warning(f"Skipped {str(n_filtered)} events not within the energy range")

    return events


def filter_bad_events(events: list[Event], energy_range: list) -> list[Event]:
    """docstring for filter_bad_events"""
    return list(
        filter(
            lambda event: energy_range[0] <= event.E0 and energy_range[1] >= event.E0,
            events,
        )
    )
