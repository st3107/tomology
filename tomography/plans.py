import math

import typing
import uuid

import numpy as np
import itertools
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from bluesky.utils import short_uid

from ophyd import Signal, Kind


class TomoPlanError(Exception):
    pass


def _extarct_motor_pos(mtr):
    ret = yield from bps.read(mtr)
    if ret is None:
        return None
    return next(
        itertools.chain(
            (ret[k]["value"] for k in mtr.hints.get("fields", [])),
            (v["value"] for v in ret.values()),
        )
    )


def configure_area_det(det, exposure, acq_time):
    """Configure exposure time of a detector in continuous acquisition mode"""
    if exposure < acq_time:
        raise TomoPlanError("exposure time < frame acquisition time: {} < {}".format(exposure, acq_time))
    yield from bps.mv(det.cam.acquire_time, acq_time)
    res = yield from bps.read(det.cam.acquire_time)
    real_acq_time = res[det.cam.acquire_time.name]["value"] if res else 1
    if hasattr(det, "images_per_set"):
        # compute number of frames
        num_frame = math.ceil(exposure / real_acq_time)
        yield from bps.mv(det.images_per_set, num_frame)
    else:
        # The dexela detector does not support `images_per_set` so we just
        # use whatever the user asks for as the thing
        num_frame = 1
    computed_exposure = num_frame * real_acq_time

    # print exposure time
    print(
        "INFO: requested exposure time = {} - > computed exposure time"
        "= {}".format(exposure, computed_exposure)
    )
    return num_frame, real_acq_time, computed_exposure


def dark_plan(detector):
    # Restage to ensure that dark frames goes into a separate file.
    yield from bps.unstage(detector)
    yield from bps.stage(detector)

    yield from bps.trigger_and_read([detector], name="dark")

    # Restage.
    yield from bps.unstage(detector)
    yield from bps.stage(detector)


def fly_scan_2d(
    area_det,
    other_dets: list,
    shutter,
    fly_motor,
    fly_start: float,
    fly_stop: float,
    fly_pixels: int,
    step_motor,
    step_start: float,
    step_stop: float,
    step_pixels: int,
    dwell_time: float,
    time_per_frame: float,
    shutter_open: typing.Any,
    shutter_close: typing.Any,
    move_velocity: float,
    *,
    shutter_wait_open: float = 1.,
    shutter_wait_close: float = 2.,
    take_dark: bool = True,
    md: dict = None,
    backoff: float = 0.,
    snake: bool = False,
):
    """
    Collect a 2D XRD map by "flying" in one direction.
    Parameters
    ----------
    area_det :
        The area detector.
    other_dets :
        The other detectors.
    shutter :
        The motor to control the fast shutter.
    fly_motor :
       The motor that will be moved continuously during collection
       (aka "flown")
    fly_start, fly_stop :
       The start and stop position of the "fly" direction
    fly_pixels : int
       The target number of pixels in the "fly" direction
    step_motor :
       The "slow" axis
    step_start, step_stop :
       The first and last position for the slow direction
    step_pixels :
       How many pixels in the slow direction
    dwell_time :
       How long in s to dwell in each pixel.  combined with *fly_pixels*
       this will be used to compute the motor velocity
    md : Optional[Dict[str, Any]]
       User-supplied meta-data
    backoff :
       How far to move beyond the fly dimensions to get up to speed
    snake :
       If we should "snake" or "comb" the fly axis
    """
    ad = area_det
    del area_det
    # all dets
    dets = [ad] + other_dets
    # rename area_det
    # rename here to use better internal names (!!)
    req_dwell_time = dwell_time
    del dwell_time
    # check values
    if fly_pixels <= 0:
        raise TomoPlanError("fly_pixels cannot be non-positive: {}.".format(fly_pixels))

    plan_args_cache = {
        k: v
        for k, v in locals().items()
        if k not in ("dets", "fly_motor", "step_motor", "dark_plan", "shutter")
    }

    num_frame, acq_time, computed_dwell_time = yield from configure_area_det(
        ad, req_dwell_time, time_per_frame
    )
    if computed_dwell_time <= 0:
        raise TomoPlanError("computed_dwell_time cannot be non-positive: {}.".format(computed_dwell_time))

    # set up metadata
    sp = {
        "time_per_frame": acq_time,
        "num_frames": num_frame,
        "requested_exposure": req_dwell_time,
        "computed_exposure": computed_dwell_time,
        "type": "ct",
        "uid": str(uuid.uuid4()),
        "plan_name": "map_scan",
    }
    _md = {
        "detectors": [det.name for det in dets],
        "plan_args": plan_args_cache,
        "map_size": (fly_pixels, step_pixels),
        "hints": {},
        "sp": sp,
        "extents": [(fly_start, fly_stop), (step_stop, step_start)],
        **{f"sp_{k}": v for k, v in sp.items()},
    }
    _md.update(md or {})
    _md["hints"].setdefault(
        "dimensions",
        [((f"start_{fly_motor.name}",), "primary"), ((step_motor.name,), "primary")],
    )
    #_md["hints"].setdefault(
    #    "extents", [(fly_start, fly_stop), (step_stop, step_start)],
    #)

    # soft signal to use for tracking pixel edges
    px_start = Signal(name=f"start_{fly_motor.name}", kind=Kind.normal)
    px_stop = Signal(name=f"stop_{fly_motor.name}", kind=Kind.normal)

    # or get the gating working below.
    speed = abs(fly_stop - fly_start) / (fly_pixels * computed_dwell_time)

    @bpp.reset_positions_decorator([fly_motor.velocity])
    @bpp.set_run_key_decorator(f"xrd_map_{uuid.uuid4()}")
    @bpp.stage_decorator(dets)
    @bpp.run_decorator(md=_md)
    def inner():
        _fly_start, _fly_stop = fly_start, fly_stop
        _backoff = backoff

        # yield from bps.mv(fly_motor.velocity, speed)
        for step in np.linspace(step_start, step_stop, step_pixels):
            yield from bps.checkpoint()
            yield from bps.mv(fly_motor.velocity, move_velocity)
            pre_fly_group = short_uid("pre_fly")
            yield from bps.abs_set(step_motor, step, group=pre_fly_group)
            yield from bps.abs_set(
                fly_motor, _fly_start - _backoff, group=pre_fly_group
            )
            yield from bps.wait(group=pre_fly_group)

            # wait for the pre-fly motion to stop
            yield from bps.mv(fly_motor.velocity, speed)
            yield from bps.mv(shutter, shutter_open)
            yield from bps.sleep(shutter_wait_open)
            fly_group = short_uid("fly")
            yield from bps.abs_set(fly_motor, _fly_stop + _backoff, group=fly_group)
            # TODO gate starting to take data on motor position
            for j in range(fly_pixels):

                fly_pixel_group = short_uid("fly_pixel")
                for d in dets:
                    yield from bps.trigger(d, group=fly_pixel_group)

                # grab motor position right after we trigger
                start_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_start, start_pos)
                # wait for frame to finish
                yield from bps.wait(group=fly_pixel_group)

                # grab the motor position
                stop_pos = yield from _extarct_motor_pos(fly_motor)
                yield from bps.mv(px_stop, stop_pos)
                # generate the event
                yield from bps.create("primary")
                for obj in dets + [px_start, px_stop, step_motor, fly_motor.velocity]:
                    yield from bps.read(obj)
                yield from bps.save()
            yield from bps.checkpoint()
            yield from bps.mv(shutter, shutter_close)
            yield from bps.sleep(shutter_wait_close)
            if take_dark:
                yield from dark_plan(ad)
            yield from bps.wait(group=fly_group)
            yield from bps.checkpoint()
            if snake:
                # if snaking, flip these for the next pass through
                _fly_start, _fly_stop = _fly_stop, _fly_start
                _backoff = -_backoff

    yield from inner()
