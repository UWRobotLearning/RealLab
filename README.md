## Installation
1. Use the instructions in the hound repo to install and build `hound_core` and all of its dependancies. [https://github.com/prl-mushr/hound_core](https://github.com/prl-mushr/hound_core)
2. Clone this repo into the src directory of your catkin workspace.
3. If you would like to run a policy that uses mocap, clone the odom branch of the mushr mocap repo with `git clone https://github.com/prl-mushr/mushr_mocap.git -b odom`. Then follow the [installation instructions](https://github.com/prl-mushr/mushr_mocap/tree/odom?tab=readme-ov-file#mushr-mocap) in mushr_mocap to get mocap set up.
4. Run `catkin_make` to build all packages
##

## Running the Vehicle
1. Upload your model weights into the `src/models` folder. For more information about training models refer to [https://github.com/UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab)
2. Go to `config/policies/<POLICY_TO_RUN>.yaml` and configure the model parameters
3. Use the following command to launch the policy:
```bash
roslaunch real_lab real_lab.launch policy:=<POLICY_TO_RUN> robot_name:=<ROBOT_NAME>
```
4. Arm the hound and put it into autonomous mode by pushing both of the right triggers all the way down.
5. Move the throttle passed the half way point to enable the policy to start moving the vehicle
6. To log data run the launch command with the `data:=True` arguement

## References

### This work

```
@misc{2502.07380,
Author = {Tyler Han and Preet Shah and Sidharth Rajagopal and Yanda Bao and Sanghun Jung and Sidharth Talia and Gabriel Guo and Bryan Xu and Bhaumik Mehta and Emma Romig and Rosario Scalise and Byron Boots},
Title = {Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics},
Year = {2025},
Eprint = {arXiv:2502.07380},
}
```

### Hound

[1] Sidharth Talia, Matt Schmittle, Alexander Lambert, Alexander Spitzer, Christoforos Mavrogiannis, and Siddhartha S. Srinivasa.Demonstrating HOUND: A Low-cost Research Platform for High-speed Off-road Underactuated Nonholonomic Driving, July 2024.URL http://arxiv.org/abs/2311.11199.arXiv:2311.11199 [cs].