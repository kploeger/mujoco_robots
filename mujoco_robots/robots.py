import os, inspect
import numpy as np
import time
import datetime
import mujoco_py
import pandas as pd
from threading import Thread
import imageio
import matplotlib.pyplot as plt

from mujoco_robots.math import transform


class MujocoRobot():

    def __init__(self, xml_path, object_names=[], cb=None, render=True, init_pos=None, init_vel=None,
                 gravity_compensation=False):
        assert xml_path == None or os.path.isfile(xml_path)
        assert init_pos is None or (len(init_pos) == self.n_dof and isinstance(init_pos, np.ndarray))
        assert init_vel is None or (len(init_vel) == self.n_dof and isinstance(init_vel, np.ndarray))

        assert hasattr(self, 'n_dof')     # Int
        assert hasattr(self, 'home_pos')  # numpy.ndarray
        assert hasattr(self, 'p_gains')   # numpy.ndarray
        assert hasattr(self, 'd_gains')   # numpy.ndarray
        assert hasattr(self, 'max_ctrl')  # numpy.ndarray
        assert hasattr(self, 'min_ctrl')  # numpy.ndarray
        assert hasattr(self, 'f_ctrl')    # Float

        self.object_names = object_names
        self.n_objects = len(object_names)

        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=1)
        self.render = render

        self._gravity_compensation = gravity_compensation
        if self._gravity_compensation:
            self.gravity_compensation_controller = GravityCompensationController(self.sim, self.n_dof,
                                                                                 self.mass_names, self.masses)


        # time and task
        self.time = self.sim.data.time
        self.timestep = 0.
        self.time_task = 0.
        self.timestep_task = 0.
        self.task_done = False

        self.dt = 1./self.f_ctrl

        # Set the initial position and velocity
        if init_pos is None:
            self.init_pos = self.home_pos
        else:
            self.init_pos = init_pos.ravel().copy()
        if init_vel is None:
            self.init_vel = np.zeros(self.n_dof)
        else:
            self.init_vel = init_vel.ravel().copy()

        self.reset()

        # Create the viewer and initialize the time
        self.viewer = None
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer._hide_overlay = True
            self.viewer._run_speed = 1.0
            self.viewer_setup()
            # self.viewer._record_video = True  # TODO: Video recording does not work yet

        if cb is None:
            self.wait()
        else:
            self.control_callback = cb

        self._recording = False
        self._video_recording = False

    # ========== PUBLIC ==========
    def viewer_setup(self):
        raise NotImplementedError()
        # if self.render:
        #     self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        #     self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        #     self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        #     self.viewer.cam.lookat[1] += 0.5
        #     self.viewer.cam.lookat[2] += 0.5
        #     self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        #     self.viewer.cam.azimuth = 0


    def start_recording(self):
        self._recording = True


    def stop_recording(self):
        self._recording = False


    def start_video_recording(self, cameras, video_subsampling=10):
        self.cameras = cameras
        self.video_subsampling = video_subsampling
        self._recorded_video = [[] for _ in range(len(cameras))]
        self._video_recording = True


    def stop_video_recording(self):
        self._video_recording = False


    def step(self, des_pos=None, des_vel=None, tau=None):
        assert des_pos is None or len(des_pos) == self.n_dof and isinstance(des_pos, np.ndarray)
        assert des_vel is None or len(des_vel) == self.n_dof and isinstance(des_vel, np.ndarray)
        assert tau is None or len(tau) == self.n_dof and isinstance(tau, np.ndarray)

        # control:
        motor_torques = np.zeros(self.n_dof)
        if not des_pos is None:
            motor_torques += self.p_gains * (des_pos.ravel() - self.pos)
        if not des_vel is None:
            motor_torques += self.d_gains * (des_vel.ravel() - self.vel)
        if not tau is None:
            motor_torques += tau.ravel()
        if self._gravity_compensation:
            tau = self.gravity_compensation_controller.compute()
            motor_torques += tau
        motor_torques = np.maximum(np.minimum(motor_torques, self.max_ctrl), self.min_ctrl)

        # apply actions and advance the simulation
        self.sim.data.qfrc_applied[:self.n_dof] = motor_torques
        self.sim.step()

        # get sim state / observations
        self.pos = self.sim.data.qpos[:self.n_dof].copy()
        self.vel = self.sim.data.qvel[:self.n_dof].copy()

        # update time
        self.time = self.sim.data.time
        self.timestep += 1
        self.time_task += self.dt
        self.timestep_task += 1

        # Render the simulation
        if self.render:
            self.viewer.render()

        return self.pos, self.vel, self.time


    def reset(self):
        self.sim.reset()

        self.timestep = 0.
        self.time = 0.
        self.timestep_task = 0.
        self.time_task = 0.
        self.task_done = False

        self.sim.data.qpos[:self.n_dof] = self.init_pos
        self.sim.data.qvel[:self.n_dof] = self.init_vel
        self.pos = self.sim.data.qpos[:self.n_dof].copy()
        self.vel = self.sim.data.qvel[:self.n_dof].copy()
        self.pos_des = self.sim.data.qpos[:self.n_dof].copy()
        self.vel_des = self.sim.data.qvel[:self.n_dof].copy()
        self.tau_des = np.zeros(self.n_dof)  # feed forward

        self.sim.forward()

        return self.pos, self.vel, self.time


    def set_control_cb(self, cb):
        self.control_callback = cb


    def start_spinning(self):
        self._spinning = True
        self.control_thread = Thread(target=self._spin)
        self.control_thread.start()


    def stop_spinning(self):
        self._spinning = False


    def get_image(self, camera_name, width=128, height=128, depth=False):
        img = self.sim.render(camera_name=camera_name, width=width, height=height, depth=depth)
        return img


    def set_joint_state(self, pos, vel):
        self.sim.data.qpos[:self.n_dof] = pos
        self.sim.data.qvel[:self.n_dof] = vel


    def get_site_pos(self, site_name, robot_coords=True):
        world_pos = self.sim.data.get_site_xpos(site_name)
        if robot_coords:
            return transform(self.WORLD2ROBOT, pos=world_pos)
        else:
            return world_pos


    def get_site_vel(self, site_name, robot_coords=True):
        world_vel = self.sim.data.get_site_xvelp(site_name)
        if robot_coords:
            return transform(self.WORLD2ROBOT, vel=world_vel)
        else:
            return world_vel


    def get_site_vel_rot(self, site_name, robot_coords=True):
        raise NotImplementedError('coordinate transform of rotational velocities is not implemented')


    def get_site_jacp(self, site_name, robot_coords=True):
        jacp = np.zeros(3 * self.model.nv)
        self.sim.data.get_site_jacp(site_name, jacp=jacp)
        jacp = np.array(jacp).reshape((3, self.model.nv))[:, :self.n_dof]
        if robot_coords:
            return transform(self.WORLD2ROBOT, jac=jacp)
        else:
            return jacp


    def set_mocap_pos(self, mocap_name, pos, robot_coords=True):
        if robot_coords:
            self.sim.data.set_mocap_pos(mocap_name, transform(self.ROBOT2WORLD, pos=pos))
        else:
            self.sim.data.set_mocap_pos(mocap_name, pos)


    # ========== TASKS ==========

    def wait_for_task(self):
        while not self._ready:
            time.sleep(0.1)


    def start(self, sync_mode = False):
        self.start_spinning()


    def stop(self, task_name='STOP'):
        self.stop_spinning()


    def wait(self, task_name='WAIT'):
        """keeps current position - usually called automatically"""
        self._ready = True
        self._task = task_name
        self.vel_des = np.zeros(self.n_dof)
        self._register_new_task(self._wait_cb)


    def goto_joint_cubic(self, q, dq, duration, task_name='GOTO_CUBIC'):
        self._task = task_name
        self._ready = False

        q_start = self.pos_des
        # dq_start = self.vel_des
        dq_start = np.zeros_like(q_start)
        q_stop = np.array(q)
        dq_stop = np.array(dq)

        n_steps = float(duration) / self.dt
        self._goto_joint_qubic_task_d = q_start
        self._goto_joint_qubic_task_c = dq_start
        self._goto_joint_qubic_task_b = 3 * (q_stop - q_start) / n_steps ** 2 - (dq_stop + 2 * dq_start) / n_steps
        self._goto_joint_qubic_task_a = - 2 * (q_stop - q_start) / n_steps ** 3 + (dq_stop + dq_start) / n_steps ** 2
        self._goto_joint_qubic_task_T = n_steps
        self._register_new_task(self._goto_joint_cubic_cb, float(duration)/self.dt)


    def zero_torque(self, duration=-1, task_name='ZERO_GRAVITY'):
        self._task = task_name
        self._ready = False
        self.pos_des = None
        self.vel_des = None
        self.tau_des = None
        self._register_new_task(self._zero_torque_cb, float(duration) / self.dt)


    def go_home(self, duration=4.):
        """returns the robot to its home position"""
        self._ready = False
        self.goto_joint_cubic(self.home_pos, np.zeros(self.n_dof), duration, task_name='GO_HOME')


    # ========== PRIVATE ==========

    def _spin(self):
        while self._spinning:
            if not self.control_callback is None:
                self.control_callback()
                self.step(self.pos_des, self.vel_des, self.tau_des)


    def _register_new_task(self, cb, task_duration = -1.):
        self.timestep_task = 0.
        self.task_duration = task_duration
        self.control_callback = cb


    def _record_state(self):
        obj_states = []
        for obj in range(self.n_objects):
            obj_states += self.obj_pos[obj].tolist()
            obj_states += self.obj_quat[obj].tolist()
        self._recorded_trajectory.append([self.time, self.time_task, self._task]
                                         + self.pos_des.tolist()
                                         + self.vel_des.tolist()
                                         + self.pos.tolist()
                                         + self.vel.tolist()
                                         + obj_states)


    # ========== TASK CALLBACKS ==========

    def _wait_cb(self):
        pass


    def _goto_joint_cubic_cb(self):
        a = self._goto_joint_qubic_task_a
        b = self._goto_joint_qubic_task_b
        c = self._goto_joint_qubic_task_c
        d = self._goto_joint_qubic_task_d
        t = self.timestep_task

        self.pos_des = a * t ** 3 + b * t ** 2 + c * t + d
        self.vel_des = (3 * a * t ** 2 + 2 * b * t + c) / self.dt

        if self.timestep_task >= self.task_duration:  # TODO: erase 5
            self.task_done = True
            self.wait(task_name='WAIT_AFTER_' + self._task)


    def _zero_torque_cb(self):
        if self.timestep_task >= self.task_duration:
            self.task_done = True
            self.wait(task_name='WAIT_AFTER_' + self._task)



class MujocoWAM7(MujocoRobot):

    default_xml_file = "wam7/wam_7dof.xml"

    # robot properties
    n_dof = 7
    home_pos = np.array([0., -1.986, 0., 3.146, 0., 0., 0.])

    masses = np.array([10.76768767,
                       3.87493756,
                       1.80228141,
                       1.06513649,
                       0.315])  # same as in xml file, for gravity compensation # TODO: fix masses of 7dof wam
    mass_names = ["sites/shoulder_yaw",
                  "sites/shoulder_pitch",
                  "sites/upper_arm",
                  "sites/forearm",
                  "sites/tool"]

    p_gains  = np.array([200.0, 300.0, 100.0, 100.0,  10.0,  10.0,   2.50])
    d_gains  = np.array([  7.0,  15.0,   5.0,   2.5,   0.3,   0.3,   0.05])
    max_ctrl = np.array([150.0, 125.0,  40.0,  60.0,   5.0,   5.0,   2.00])
    min_ctrl = -max_ctrl
    f_ctrl = 500.

    # transformations
    ROBOT2WORLD = np.array([[ 0, -1,  0,  0   ],
                            [ 1,  0,  0,  0   ],
                            [ 0,  0,  1,  0.84],
                            [ 0,  0,  0,  1   ]])
    WORLD2ROBOT = np.array([[ 0,  1,  0,  0   ],
                            [-1,  0,  0,  0   ],
                            [ 0,  0,  1, -0.84],
                            [ 0,  0,  0,  1   ]])
    OPTI2WORLD  = np.array([[-1,  0,  0,  0   ],
                            [ 0,  0,  1,  1.21],
                            [ 0,  1,  0,  0   ],
                            [ 0,  0,  0,  1   ]])
    WORLD2OPTI  = np.array([[-1,  0,  0,  0   ],
                            [ 0,  0,  1,  0   ],
                            [ 0,  1,  0, -1.21],
                            [ 0,  0,  0,  1   ]])
    OPTI2ROBOT  = np.array([[ 0,  0,  1,  1.21],
                            [ 1,  0,  0,  0   ],
                            [ 0,  1,  0, -0.84],
                            [ 0,  0,  0,  1   ]])
    ROBOT2OPTI  = np.array([[ 0,  1,  0,  0   ],
                            [ 0,  0,  1,  0.84],
                            [ 1,  0,  0, -1.21],
                            [ 0,  0,  0,  1   ]])

    def __init__(self, cb=None, xml_path=None, render=True, init_pos=None, init_vel=None):

        if xml_path == None:
            script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            xml_path = script_path + '/../robot_description/' + self.default_xml_file

        MujocoRobot.__init__(self, xml_path, cb=cb, render=render, init_pos=init_pos, init_vel=init_vel)

    def viewer_setup(self):
        if self.render:
            self.viewer.cam.distance = 2.5
            self.viewer.cam.lookat[0] += 0.15
            self.viewer.cam.elevation = -25
            self.viewer.cam.azimuth = -90


class MujocoWAM4(MujocoRobot):

    default_xml_file = "wam4/wam_4dof.xml"

    # robot properties
    n_dof = 4
    home_pos = np.array([0., -1.986, 0., 3.146])

    masses = np.array([10.76768767,
                       3.87493756,
                       1.80228141,
                       1.06513649,
                       0.315])  # same as in xml file, for gravity compensation
    mass_names = ["sites/shoulder_yaw",
                  "sites/shoulder_pitch",
                  "sites/upper_arm",
                  "sites/forearm",
                  "sites/tool"]

    p_gains  = np.array([200.0, 300.0, 100.0, 100.0])
    d_gains  = np.array([  7.0,  15.0,   5.0,   2.5])
    max_ctrl = np.array([150.0, 125.0,  40.0,  60.0])
    min_ctrl = -max_ctrl
    f_ctrl = 500.

    # transformations
    ROBOT2WORLD = np.array([[ 0, -1,  0,  0   ],
                            [ 1,  0,  0,  0   ],
                            [ 0,  0,  1,  0.84],
                            [ 0,  0,  0,  1   ]])
    WORLD2ROBOT = np.array([[ 0,  1,  0,  0   ],
                            [-1,  0,  0,  0   ],
                            [ 0,  0,  1, -0.84],
                            [ 0,  0,  0,  1   ]])
    OPTI2WORLD  = np.array([[-1,  0,  0,  0   ],
                            [ 0,  0,  1,  1.21],
                            [ 0,  1,  0,  0   ],
                            [ 0,  0,  0,  1   ]])
    WORLD2OPTI  = np.array([[-1,  0,  0,  0   ],
                            [ 0,  0,  1,  0   ],
                            [ 0,  1,  0, -1.21],
                            [ 0,  0,  0,  1   ]])
    OPTI2ROBOT  = np.array([[ 0,  0,  1,  1.21],
                            [ 1,  0,  0,  0   ],
                            [ 0,  1,  0, -0.84],
                            [ 0,  0,  0,  1   ]])
    ROBOT2OPTI  = np.array([[ 0,  1,  0,  0   ],
                            [ 0,  0,  1,  0.84],
                            [ 1,  0,  0, -1.21],
                            [ 0,  0,  0,  1   ]])

    def __init__(self, cb=None, xml_path=None, object_names=[], render=True, init_pos=None, init_vel=None,
                 gravity_compensation=False):

        if xml_path == None:
            script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            xml_path = script_path + '/../robot_description/' + self.default_xml_file

        MujocoRobot.__init__(self, xml_path, object_names=object_names, cb=cb, render=render, init_pos=init_pos,
                             init_vel=init_vel, gravity_compensation=gravity_compensation)

        # data recording
        self._recorded_trajectory = []
        self._recorded_video = []
        self._task = None
        self._recording = False

    def viewer_setup(self):
        if self.render:
            self.viewer.cam.distance = 2.5
            self.viewer.cam.lookat[0] += 0.15
            self.viewer.cam.elevation = -25
            self.viewer.cam.azimuth = -90


    def step(self, des_pos=None, des_vel=None, tau=None):
        super().step(des_pos, des_vel, tau)
        if self._recording:
            # print('pos {} vel {}'.format(self.pos, self.vel))
            self._record_state()

        if self._video_recording:
            if self.timestep % self.video_subsampling == 0:
                self._record_img_frame()

        return self.pos, self.vel, self.time


    def save_recorging(self, path):
        path_ = path + '/mujoco/'
        if not os.path.exists(path_):
            os.makedirs(path_)
        columns = ['time', 'time_task', 'task',
                   'q1_des', 'q2_des', 'q3_des', 'q4_des', 'dq1_des', 'dq2_des', 'dq3_des', 'dq4_des',
                   'q1', 'q2', 'q3', 'q4', 'dq1', 'dq2', 'dq3', 'dq4']
        for i in range(self.n_objects):
            columns += ['pos_x_obj{}'.format(i+1),
                        'pos_y_obj{}'.format(i+1),
                        'pos_z_obj{}'.format(i+1),
                        'quat_x_obj{}'.format(i+1),
                        'quat_y_obj{}'.format(i+1),
                        'quat_z_obj{}'.format(i+1),
                        'quat_w_obj{}'.format(i+1)]
        traj_df = pd.DataFrame(self._recorded_trajectory[:], columns=columns)
        traj_df.to_pickle(path_ + 'trajctory_' + datetime.datetime.now().isoformat() + '.p')


    def save_video_recording(self, path):
        fps = 50
        video_name = 'test_video.avi'
        for cam_idx in range(len(self.cameras)):
            writer = imageio.get_writer(video_name, fps=fps)
            for i in range(len(self._recorded_video[cam_idx])):
                img = self._recorded_video[cam_idx][i]
                if i < 10:
                    plt.imshow(img)
                    plt.show()
                writer.append_data(img)
            writer.close()


    def _record_img_frame(self):
        for i in range(len(self.cameras)):
            img = self.get_image(self.cameras[i])
            self._recorded_video[i].append(img)



class GravityCompensationController():

    def __init__(self, robot, n_dof, mass_names, masses, gravity=np.array([0., 0., -9.81])):
        assert len(mass_names) == len(masses)
        self.robot = robot
        self.n_dof = n_dof
        self.mass_names = mass_names
        self.masses = masses
        self.g = gravity

    def compute(self):
        tau = np.zeros(self.n_dof)
        for i in range(len(self.mass_names)):
            # get jacobian
            target_jacp = np.zeros(3 * self.robot.model.nv)
            self.robot.data.get_site_jacp(self.mass_names[i], jacp=target_jacp)

            # get rid of additional objects's free joints in target_jacp
            jac = np.array(target_jacp).reshape((3, self.robot.model.nv))[:, :self.n_dof]

            # compute torques with jacobian transpose method
            tau += jac.T @ (- self.masses[i] * self.g)

        return tau

