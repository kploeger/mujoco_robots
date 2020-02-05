## MuJoCo Robots

A colltection of robots at IAS Lab modeled in MuJoCo with mujoco-py

### List of Robots:
- Barret WAM 7 DoF
- Barret WAM 4 DoF

### Tested with:
- Python 3.5, MuJoCo 2.00, mujoco-py 2.0.2.2


## Install instructions:
    
    git clone git@github.com:kploeger/mujoco_robots.git
    pip install -e mujoco_robots/ --user
    
## Usage
Take a look at the demos.

## Known Errors
### LD_LIBRARY_PATH in PyCharm:

    Missing path to your environment variable. 
    Current values LD_LIBRARY_PATH=
    Please add following line to .bashrc:
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kai/.mujoco/mujoco200/bin
    
PyCharm clears $LD_LIBRARY_PATH even if you add it to your .bashrc. Instead add it manually in PyCharm. To do so go to 
the drop down menu left of 'Run' button and select 'Edit Configurarions...' and paste 
';LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kai/.mujoco/mujoco200/bin'
in 'Environment variables'
