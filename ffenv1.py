from sympy import *
import numpy as np
from mpmath import *
import gym
from gym import spaces
from gym.utils import seeding


class FrictionFinger :
    """

    state=(d1,d2,theta_obj,fs)
    action=(fs,theta_delta)
    """

    def __init__(self):
        self.sqrwidth=2.5
        self.d1 = 6
        self.t1 = 40 * np.pi / 180
        self.fngrwdth = 1.5
        self.palmwdth = 6.0
        self.d2 = 6
        self.t2 = 138 * np.pi / 180
        self.action_space =  spaces.Discrete(5)
        self.observation_space = spaces.Tuple((spaces.Box(low = np.array([0, 0]), high=np.array([12, 12]), dtype=np.int), spaces.Discrete(2)))
        self.state= None
        self.theta1_solution=None
        self.theta2_solution=None
        self.reward= None
        self.d1_d=10
        self.d2_d=10
    def translate_1(self, action):
        # Center Coordinates
        d2= self.state(2)
        t2=self.theta2_solution
        x_square = (d2 - self.sqrwidth / 2) * np.cos(t2) + (self.fngrwdth + self.sqrwidth / 2) * np.sin(t2)
        y_square = (d2 - self.sqrwidth / 2) * np.sin(t2) - (self.fngrwdth + self.sqrwidth / 2) * np.cos(t2)

        # Calculate theta2, d2
        d2v = np.array([d2 * np.cos(t2), d2 * np.sin(t2)])
        self.sqrwidthv = np.array([self.sqrwidth * np.sin(t2), -self.sqrwidth * np.cos(t2)])
        self.palmwdthv = np.array([self.palmwdth, 0])
        f1v = np.array([self.fngrwdth * np.sin(t2), -self.fngrwdth * np.cos(t2)])
        av = d2v + f1v + self.sqrwidthv - self.palmwdthv

        d1 = np.sqrt((av * av).sum() - self.fngrwdth * self.fngrwdth)
        t1 = np.arctan2(av[1], av[0]) - np.arctan2(self.fngrwdth, d1)
        return t1, d1


    def translate_2(self, action):
        d1 = self.state(1)
        t1 = self.theta1_solution + action(1) * pi / 36;
        # Center Coordinates of square
        x_square = self.palmwdth + (d1 - self.sqrwidth / 2) * np.cos(t1) - (self.sqrwidth / 2 + self.fngrwdth) * np.sin(t1)
        y_square = (d1 - self.sqrwidth / 2) * np.sin(t1) + (self.sqrwidth / 2 + self.fngrwdth) * np.cos(t1)
        # Calculate theta1, d1
        d1v = np.array([d1 * np.cos(t1), d1 * np.sin(t1)])
        self.sqrwidthv = np.array([self.sqrwidth * np.sin(t1), self.sqrwidth * np.cos(t1)])
        self.palmwdthv = np.array([self.palmwdth, 0])
        f2v = np.array([self.fngrwdth * np.sin(t1), self.fngrwdth * np.cos(t1)])
        av = d1v - self.sqrwidthv - f2v + self.palmwdthv
        d2 = np.sqrt((av * av).sum() - self.fngrwdth * self.fngrwdth)
        t2 = np.arctan2(av[1], av[0]) + np.arctan2(self.fngrwdth, d2)
        return t2,d2


    def step(self, action):
        if action[2]==1:
            self.t1,self.t2=self.calc_th1andth2left(self,action)
            self.t1,d1= self.translate_1(self, action)
            self.state=[d1,self.state[1],0,self.state[2]]
            self.reward=exp(-2.5*(abs(self.d1-self.d1_d)+abs(self.d2-self.d2_d)))
        elif action[2]==2:
            self.t1, self.t2=self.calc_th1andth2right(self,action)
            t2,d2 =self.translate_2(self,action)
            self.state = [d2, self.state[1], 0, self.state[2]]
            self.reward = exp(-2.5 * (abs(self.d1 - self.d1_d) + abs(self.d2 - self.d2_d)))

    def cacl_th1andth2left(self,action):
        assert self.state is not None, "the state space is none"
        (self.t1, self.t2) = variables
        if action == 0
            x=-1
        elif action == 1
            x=1
        eqn3 = self.fngrwdth * sin(self.t1) + self.fngrwdth * sin(self.t2) + (self.d1+x) * cos(
            self.t1) + self.sqrwidth * sin(
            self.t2) - self.palmwdth - self.d2 * cos(self.t2)
        eqn4 = -self.fngrwdth * cos(self.t1) - self.fngrwdth * cos(self.t2) + (self.d1+x) * sin(
            self.t1) - self.sqrwidth * cos(
            self.t2) - self.d2 * sin(self.t2)
        self.t1, self.t2 = opt.fsolve(f, (0.1, 1))
        print(self.t1, self.t2)


    def f_left(self):
        (self.t1, self.t2) = variables
        eqn3 = self.fngrwdth * sin(self.t1) + self.fngrwdth * sin(self.t2) + (self.d1 + x) * cos(
            self.t1) + self.sqrwidth * sin(
            self.t2) - self.palmwdth - self.d2 * cos(self.t2)
        eqn4 = -self.fngrwdth * cos(self.t1) - self.fngrwdth * cos(self.t2) + (self.d1 + x) * sin(
            self.t1) - self.sqrwidth * cos(
            self.t2) - self.d2 * sin(self.t2)
        return [eqn3 eqn4]

    def cacl_th1andth2right(self,action):
        assert self.state is not None , "the state space is none"
        if action == 0
            x=-1
        elif action == 1
            x=1
        (self.t1, self.t2) = variables
        eqn1 = self.fngrwdth * sin(self.t1) + self.fngrwdth * sin(self.t2) + self.d1 * cos(self.t1) + self.sqrwidth * sin(
            self.t1) - self.palmwdth - (self.d2 + x) * cos(self.t2)
        eqn2 = -self.fngrwdth * cos(self.t1) - self.fngrwdth * cos(self.t2) + self.d1 * sin(self.t1) - self.sqrwidth * cos(
            self.t1) - (self.d2+x) * sin(self.t2)
        self.t1,self.t2 = opt.fsolve(f, (0.1, 1))
        print(self.t1,self.t2)

    def reset(self):
        self.state=[6,6,0,1]
        return np.array(self.state)



import pdb

if __name__ == '__main__':
   ff = FrictionFinger()
   print('object created')
   state= ff.reset()
   print(state)
   print('state reset')
   #t1,t2=ff.cacl_th1andth2left(1)
