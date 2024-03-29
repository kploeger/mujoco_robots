from distutils.core import setup

setup(
    name='MuJoCo Robots',
    version='0.1dev',
    packages=['mujoco_robots',],
    install_requires=['pandas'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)
