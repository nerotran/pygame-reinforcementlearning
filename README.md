# pygame-reinforcementlearning
requires: pygame, tensorflow, numpy, pandas

Description: A python program where tensorflow is used to train learning agent to so that a black dot will recognize that its purpose is to follow a constantly-moving red dot.

How to use:\
step 1: run input.py\
step 2: a window will pop up with 2 dots: a red one and the black one. The goal is for the user to keep the black dot "chasing" the red dot by clicking on the red dot. The player performance data will be written into data.txt Notes: if the black dot is far away from the red dot, the user should click further up from the red dot as the travel distance increases according to how far away the click is.\
step 3: run train-model.py. Tensorflow will train the learning agent using the user input data in data.txt\
step 4: run RL.py and see how the learning agent performs. This time the black dot will moves on its own.
