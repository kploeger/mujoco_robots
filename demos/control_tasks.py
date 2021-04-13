import numpy as np
import time
from mujoco_robots.robots import MjWam7, MjWam4


if __name__ == "__main__":
    # robot = MjWAM7(render=True)
    robot = MjWam4(render=True)

    goal_pos = np.ones(robot.n_dof)
    goal_vel = np.zeros(robot.n_dof)
    max_time = 3.

    robot.start_spinning()

    # define a high level task...
    robot.goto_joint_cubic(goal_pos, goal_vel, max_time)
    # ...and wait until it is done.
    robot.wait_for_task()

    robot.goto_home()
    robot.wait_for_task()

    robot.stop_spinning()
    robot.close()
