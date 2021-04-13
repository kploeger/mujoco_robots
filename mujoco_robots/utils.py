from mujoco_py import MjViewer
import glfw

import numpy as np
import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


open_viewers = []  # a static list to keep track of all viewers

class MjViewerExtended(MjViewer):
    """ An extension of mujoco-py's MjViewer. MjViewerExtended does not
        terminate all other viewers and the python interpreter when closeing.
    """
    def __init__(self, sim):
        glfw.init()  # make sure glfw is initialized
        super().__init__(sim)
        open_viewers.append(self)

    def close(self):
        """ Closes the viewers glfw window. To open a new one, create a new
            instance of MjViewerExtended
        """
        # MjViewer only calls glfw.terminate() here killing glfw entierly.
        if glfw.window_should_close(self.window):
            return
        try:
            glfw.set_window_should_close(self.window, 1)
            glfw.destroy_window(self.window)
        except Exception:
            pass

        open_viewers.remove(self)
        if len(open_viewers) == 0:
            glfw.terminate()

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE and key == glfw.KEY_ESCAPE:
            self.close()
        else:
            super().key_callback(window, key, scancode, action, mods)



class Wam4IK(Chain):
    """ A basic kinamatic model of the MjWAM4 """

    def __init__(self, active_joints=[1, 1, 1, 1],
                 base_translation=[0, 0, 0.84],        # x, y, z
                 base_orientation=[0, 0, np.pi/2],     # x, y, z
                 tool_translation=[0, 0, 0],
                 tool_orientation=[0, 0, 0]):

        links=[OriginLink(),
           URDFLink(name="wam/links/base",
                    translation_vector=base_translation,        # translation of frame
                    orientation=base_orientation,               # orientation of frame
                    rotation=[0, 0, 0]),              # joint axis [0, 0, 0] -> no joint
           URDFLink(name="wam/links/shoulder_yaw",
                    translation_vector=[0, 0, 0.16],
                    orientation=[0, 0, 0],
                    rotation=[0, 0, 1]),
           URDFLink(name="wam/links/shoulder_pitch",
                    translation_vector=[0, 0, 0.186],
                    orientation=[0, 0, 0],
                    rotation=[1, 0, 0]),
           URDFLink(name="wam/links/shoulder_roll",
                    translation_vector=[0, 0, 0],
                    orientation=[0, 0, 0],
                    rotation=[0, 0, 1]),
           URDFLink(name="wam/links/upper_arm",
                    translation_vector=[0, -0.045, 0.550],
                    orientation=[0, 0, 0],
                    rotation=[1, 0, 0]),
           URDFLink(name="wam/links/tool_base_wo_plate",
                    translation_vector=[0, 0.045, 0.350],
                    orientation=[0, 0, 0],
                    rotation=[0, 0, 0]),
           URDFLink(name="wam/links/tool_base_w_plate",
                    translation_vector=[0, 0, 0.008],
                    orientation=[0, 0, 0],
                    rotation=[0, 0, 0]),
           URDFLink(name="wam/links/tool",
                    translation_vector=tool_translation,
                    orientation=tool_orientation,
                    rotation=[0, 0, 0])
           ]

        self.all_joints = [False, False, True, True, True, True, False, False, False]
        self.active_joints = list(map(lambda x:x==1, active_joints))
        self.active_links = [False, False, *self.active_joints, False, False, False]
        Chain.__init__(self, name='wam4',
                       active_links_mask=self.active_links,
                       links=links)

    def fk(self, joints, full_kinematics=False):
        joints = np.array([0, 0, *joints, 0, 0, 0])
        return Chain.forward_kinematics(self, joints, full_kinematics)

    def ik(self, target_position=None, target_orientation=None, orientation_mode=None, **kwargs):
        full = Chain.inverse_kinematics(self, target_position, target_orientation, orientation_mode, **kwargs)
        active = self.joints_from_links(full)
        return active

    def joints_from_links(self, joints):
        return np.compress(self.all_joints, joints, axis=0)


