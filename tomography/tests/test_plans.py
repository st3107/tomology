from bluesky import RunEngine
from bluesky.simulators import summarize_plan

import tomography.sim as sim
import tomography.plans as plans


def test_fly_scan_2d():
    shutter = sim.SynSignal(name="shutter")
    area_det = sim.FakeAreaDetector(name="dexela")
    area_det.cam.configure({"acquire_time": 1.})
    motor_y = sim.DelayedSynAxis(name="motor_y")
    motor_y.configure({"velocity": 1.})
    motor_x = sim.DelayedSynAxis(name="motor_x")
    motor_x.configure({"velocity": 1.})

    plan = plans.fly_scan_2d(
        area_det, [], shutter, motor_x, 0., 2., 3, motor_y, 0., 1., 2,
        dwell_time=1., time_per_frame=1., move_velocity=1., shutter_close=1, shutter_open=0., shutter_wait=0.
    )

    summarize_plan(plan)
