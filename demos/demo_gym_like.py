import numpy as np
from mujoco_robots.robots import MujocoWAM7, MujocoWAM4

if __name__ == "__main__":
    wam = MujocoWAM7(render=True)
    # wam = MujocoWAM4(render=True)

    goal_state = np.ones(wam.n_dof)

    def compute_des_state(start, goal, max_time, t):
        des_pos = np.minimum(1., (t / max_time)) * (goal - start) + start
        des_vel = np.zeros_like(start)
        return des_pos, des_vel

    for _ in range(2):
        pos, vel, time = wam.reset()
        for i in range(2000):
            des_pos, des_vel = compute_des_state(wam.init_pos, goal_state, 3., time)
            pos, vel, time = wam.step(des_pos, des_vel)