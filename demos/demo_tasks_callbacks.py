import numpy as np
import time
from mujoco_robots.robots import MujocoWAM7, MujocoWAM4


if __name__ == "__main__":
    # robot = MujocoWAM7(render=True)
    robot = MujocoWAM4(render=True)

    goal_pos = np.ones(robot.n_dof)
    goal_vel = np.zeros(robot.n_dof)
    max_time = 3.

    robot.start()

    robot.goto_joint_cubic(goal_pos, goal_vel, max_time)
    robot.wait_for_task()

    robot.go_home()
    robot.wait_for_task()

    robot.stop()