import numpy as np
import time
from mujoco_robots.robots import MujocoWAM7, MujocoWAM4


if __name__ == "__main__":
    # robot = MujocoWAM7(render=True)
    robot = MujocoWAM4(render=True)

    goal_state = np.ones(robot.n_dof)
    start_state = robot.home_pos
    max_time = 3.

    def control_callback():
        t = robot.time
        robot.pos_des = np.minimum(1., (t / max_time)) * (goal_state - start_state) + start_state
        robot.vel_des = np.zeros(robot.n_dof)
        robot.tau_des = np.zeros(robot.n_dof)

    robot.set_control_cb(control_callback)
    robot.start_spinning()
    time.sleep(4.)
    robot.stop_spinning()