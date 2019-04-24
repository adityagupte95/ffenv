from sympy import *
import numpy as np
from mpmath import *
import gym
from gym import spaces
from gym.utils import seeding
import itertools
import matplotlib
import pandas as pd
import sys
if "../" not in sys.path:
    sys.path.append("../")
from collections import defaultdict
from lib import plotting
import matplotlib as mp

class frictionFinger():
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space=spaces.Tuple(spaces.Box(low=np.array([0, 0]), high=np.array([12, 12]),spaces.Discrete(2)))
        self.d1_d=10
        self.d2_d=10
        self.fs_d=1
        self.state=[]

        self.actiondict={ 0: 0,
                          1: 1,
                          2: -1,
                          3: 1
                          }
    def reset(self):
        self.state=[1,10,1]

    def step(self,action):
        rew= lambda st: if st=[self.d1_d,self.d2_d,self.fs_d] 1 else 0
        if action== 0:
            self.state[2]==0
            self.reward=rew
        elif action== 1:
            self.reward=rew

        elif action== 2:
            if self.state




