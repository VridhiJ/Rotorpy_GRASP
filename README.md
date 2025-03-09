# Wind Aware UAV 
This repository highlights he modifications made to [Rotorpy](https://github.com/spencerfolk/rotorpy) as part of my research at GRASP Lab. All files included originate from the original Rotorpy repository, but this is not the complete Rotorpy framework. This repository is intended to highlight the specific changes implemented.

**Key Modifications** <br>

Compatibility <br> 
- Now compatible with Gymnasium: Version 1.0.0 

Changes to Training script and quarotor environment <br>
- Training script should run for different scenarios:
  1. No wind
  2. Ground Truth Wind
  3. IMU
- Made quadrotor environment compatible to handle different training and evaluation scenarios.
  
Automated evaluation integrated into training: <br>
- Periodically assesses model performance under different conditions without waiting for training to complete to see how our model works under various conditions.
- Evaluation frequency is configurable by the user.
- Optimized evaluation by reducing tracking time for 10 parallel agents.

Visualization & Performance Tracking <br>

Gif generation <br>
- Auto-evaluation runs for only 2 seconds every few epochs to minimize training slowdowns.
- Users can view generated GIFs to assess model performance.
- Animation now supports multiple agents (10 agents running in parallel).

Tensorboard Integration <br>
- Implemented TensorBoard's SummaryWriter to log metrics for wind condition studies.

IMU Integration & Reward Function Adjustments <br>
- Integrated IMU for real-time reinforcement learning.
- Studied the impact of wind conditions on UAV tracking.
- Adjusted reward functions to improve performance when incorporating wind and IMU data.
