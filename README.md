# WindAwareUAV-RL
This repository showcases the modifications I made to [Rotorpy](https://github.com/spencerfolk/rotorpy). All files included originate from the original Rotorpy repository, but this is not the complete Rotorpy framework. This repository is intended to highlight the changes I implemented for my research at GRASP Lab.


Compatible with gymnasium = 1.0.0 <br> 

Automate the evaluation <br>
- automated the periodic evaluation of model while training to see how our model reacts to different conditions without having to wait for training to complete.
- left to user discretion on how frequent the evaluation should be. sped up the auto evaluation by reducing the time for tracking 10 agents
  
Gif generation <br>
- because the auto-evaluation only lasts for 2 seconds every few epochs to not slow down training too much if user wants to check the performance, can view the generated gif.

animate now works for multiple agents <br>
- we have 10 agents running parallely in our environment

Tensorboard <br>
- integrated tensorboard summarywriter for our wind study

Integrate IMU <br>
- for real time RL integrated IMU

Study results of wind conditions on uav tracking - change reward functions <br>
- performance evaluations of integrating wind and imu during training.

