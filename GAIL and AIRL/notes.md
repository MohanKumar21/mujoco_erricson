# Notes on AIRL implementation

Trying to replicate the results of this [repo](https://github.com/toshikwa/gail-airl-ppo.pytorch/tree/master)

- So they try two envs, one mountain car and other hopper v2
- to get expert trajectories they use soft actor critic algorithm to train policy and get demostration from there.
- they implement both GAIL and AIRL.
- there is a file called train_imitation.py looks like they trained an imitation learning method (basically supervised ML algo)

code structure

- dir
  - train_imitation.py
  - train_expert.py
  - collect_demo.py
    - Algorithms/
      - SAC.py
      - AIRL.py

## Doubts

- how is train_expert different from collect_demo?
  - oo ok, i guess they train the expert first and then collect the demos.
  - int train_imitation he trained the agent through GAIL and AIRL, idk why he called it imitation learning, since he is explicity using an IRL algo he should have called it train_irl.py
- Why is the hidden units in network.py (64,64) shouldn't it be a single number?

## Task flow

- train the SAC algorithm on both envs.
- implement AIL
