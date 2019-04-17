

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
import copy

class FrictionFinger :
    """

    state=[d1: [0:12] discrete position in left finger
            d2:[0:12] discrete position in right finger
            fs: 0: friction finger left high right low
                1: friction finger left low right high]

    ,d2,theta_obj,fs)
    action= [ 0: -1 position in the corresponding friction finger
              1: +1 in corresponding friction finger
              2: friction finger left high right low
              3: friction finger left low right high]

    Step: The env should step according to the action taken to the next step
    (Actually steps to the next stateas the action right now directly corresponds to making the state change and is
     completely deterministic and then calculates what action to take on the low level to actually make the block move
     the exact positon in the fingers to represent the state)

    Algo of step:


                    Check if the given desired combination is in the state space of the arm by checking the angles with the thresholds
                            |
                    Check the current state(fs variable)
                            |
                    check the action(0 or 1)
                            |
                     add or subtract position based on the action to the corresponding finger
                            |
                    calculate the actual angle needed to be moved to the position given by the action




                            """

    def __init__(self):
#self.sqrwidth=2.5


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Box(low=np.array([0, 0]), high=np.array([12, 12]), dtype=np.int), spaces.Discrete(2)))
        self.state = None
#       self.t1 = 40 * np.pi / 180 #       self.fngrwdth = 1.5
#       self.palmwdth = 6.0

#       self.t2 = 138 * np.pi / 180
        #self.theta1_solution=None
#       self.theta2_solution=None
        self.reward= None
        self.d1_d=7.0
        self.d2_d=10.0
        self.fs_d= 1
#       self.t1_max=140
#       self.t2_min=40

    """
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
        t1 = self.theta1_solution 
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
"""

    def reset(self):
        self.state = [10.0, 10.0, 1.0]
        return self.state

    def step(self, action):
        prev_state= copy.copy(self.state)
        rew= lambda st: 1 if st[0]==self.d1_d and st[1]==self.d2_d and st[2]==self.fs_d else 0
        #rew=lambda st: -np.linalg.norm([st[0]-self.d1_d,st[1]-self.d2_d])   /10
        #rew = lambda st: 10*np.exp((abs(st[0] - self.d1_d) + abs(st[1] - self.d2_d)))

        #rew= lambda st : 1 if st[0]==self.d1_d and st[1]==self.d2_d else -1
        # print('st:'+str(self.state)+' d:'+str([self.d1_d,self.d2_d]))
        # print(np.linalg.norm([self.state[0]-self.d1_d,self.state[1]-self.d2_d])    )
        #
        if self.state[2]== 0:
            if action == 0:
                if self.state[1] == 0:
                    # self.reward = -1
                    self.state = self.state
                    self.reward = rew(self.state)
                else :
                    self.state=[ self.state[0], self.state[1]-1 , self.state[2]]
                    self.reward =rew(self.state)
                """ self.t1,self.t2=self.calc_th1andth2left(self,action)
                self.t1,d1= self.translate_1(self, action)
                self.state=[d1,self.state[1],0,self.state[2]]"""
            elif action == 1:
                if self.state[1] == 12:
                    self. reward = rew(self.state)
                    self.state=self.state
                else:
                    self.state = [self.state[0] , self.state[1]+1, self.state[2]]
                    self.reward=rew(self.state)
                """self.t1, self.t2=self.calc_th1andth2right(self,action)
                t2,d2 =self.translate_2(self,action)
                self.state = [d2, self.state[1], 0, self.state[2]]
                self.reward = exp(-2.5 * (abs(self.d1 - self.d1_d) + abs(self.d2 - self.d2_d)))"""
            elif action == 2:
                self.state = self.state
                self.reward= rew(self.state)
            elif action == 3:
                self.state[2] = 1
                self.reward=rew(self.state)

        elif self.state[2] == 1:
            if action == 0:
                if self.state[0] == 0:
                    self.state = self.state
                    self.reward = rew(self.state)
                else:
                    self.state[0] = self.state[0]-1
                    self.reward=rew(self.state)
            elif action == 1:
                if self.state[0]==12:
                    self.reward = rew(self.state)
                    self.state=self.state
                else:
                    self.state[0] = self.state[0] + 1
                    self.reward=rew(self.state)
            elif action == 2:
                self.state[2] = 0
                self.reward= rew(self.state)
            elif action == 3:
                self.state = self.state
                self.reward= rew(self.state)

        if self.state[0:3] == [self.d1_d, self.d2_d,self.fs_d] :
            self.done =True
        else:
            self.done= False
        self.state=[max(min(self.state[0],12),0),max(min(self.state[1],12),0),max(min(self.state[2],1),0)]
        print((prev_state, self.state ,self.reward, self.done),)
        return prev_state, self.state ,self.reward, self.done
"""
    def cacl_th1andth2left(self):

        self.t1, self.t2 = opt.fsolve(f_left, (0.1, 1))
        print(self.t1, self.t2)
        return self.t1, self.t2

    def f_left(self,variables):
        (self.t1, self.t2) = variables
        eqn3 = self.fngrwdth * sin(self.t1) + self.fngrwdth * sin(self.t2) + (self.d1 ) * cos(
            self.t1) + self.sqrwidth * sin(
            self.t2) - self.palmwdth - self.d2 * cos(self.t2)
        eqn4 = -self.fngrwdth * cos(self.t1) - self.fngrwdth * cos(self.t2) + (self.d1 ) * sin(
            self.t1) - self.sqrwidth * cos(
            self.t2) - self.d2 * sin(self.t2)
        return [eqn3 eqn4]

    def f_right(self, variables):
        (self.t1, self.t2) = variables

        eqn1 = self.fngrwdth * sin(self.t1) + self.fngrwdth * sin(self.t2) + self.d1 * cos(self.t1) + self.sqrwidth * sin(
        self.t1) - self.palmwdth - (self.d2 + x) * cos(self.t2)
        eqn2 = -self.fngrwdth * cos(self.t1) - self.fngrwdth * cos(self.t2) + self.d1 * sin(self.t1) - self.sqrwidth * cos(
        self.t1) - (self.d2 + x) * sin(self.t2)
        return eqn 1
        print(self.t1, self.t2)
        return self.t1, self.t2

    def cacl_th1andth2right(self):
        self.t1, self.t2 = opt.fsolve(f_right, (0.1, 1))
        print(self.t1, self.t2)
        return self.t1, self.t2
"""





import pdb

if __name__ == '__main__':

   env = FrictionFinger()
   # #env.reset()
   # print(env.state)
   # st,r,done=env.step(2)
   # print(env.state)
   # print(r)

   def make_epsilon_greedy_policy(Q, epsilon, nA):
       """
       Creates an epsilon-greedy policy based on a given Q-function and epsilon.

       Args:
           Q: A dictionary that maps from state -> action-values.
               Each value is a numpy array of length nA (see below)
           epsilon: The probability to select a random action. Float between 0 and 1.
           nA: Number of actions in the environment.

       Returns:
           A function that takes the observation as an argument and returns
           the probabilities for each action in the form of a numpy array of length nA.

       """
       def policy_fn(observation):
           A = np.ones(nA, dtype=float) * epsilon / nA
           best_action = np.argmax(Q[tuple(observation)])
           A[best_action] += (1.0 - epsilon)
           return A
       return policy_fn



   def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.9, epsilon=0.1):
       """
       Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
       while following an epsilon-greedy policy

       Args:
           env: OpenAI environment.
           num_episodes: Number of episodes to run for.
           discount_factor: Gamma discount factor.
           alpha: TD learning rate.
           epsilon: Chance to sample a random action. Float between 0 and 1.

       Returns:
           A tuple (Q, episode_lengths).
           Q is the optimal action-value function, a dictionary mapping state -> action values.
           stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
       """

       # The final action-value function.
       # A nested dictionary that maps state -> (action -> action-value).
       Q = defaultdict(lambda: np.ones(env.action_space.n))

       # Keeps track of useful statistics
       stats = plotting.EpisodeStats(
           episode_lengths=np.zeros(num_episodes),
           episode_rewards=np.zeros(num_episodes))

       # The policy we're following
       policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

       for i_episode in range(num_episodes):
           # Print out which episode we're on, useful for debugging.
           # if (i_episode + 1) % 100 == 0:
           #     print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
           #     sys.stdout.flush()

           # Reset the environment and pick the first action
           state = env.reset()

           # One step in the environment
           # total_reward = 0.0
           for t in itertools.count():


               # Take a step
               action_probs = policy(state)
               action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
               # print("state_before", env.state)
               # print("action",action)
               prev_state,next_state, reward, done = env.step(action)
               # print("state_after", env.state)
               # if done: #or t==1000:
               #
               #     print('found the solution',state)
               #     #print (Q)
               #     break

               
               # Update statistics
               stats.episode_rewards[i_episode] += reward
               stats.episode_lengths[i_episode] = t

               # TD Update
               best_next_action = np.argmax(Q[tuple(next_state)])

               td_target = reward + discount_factor * Q[tuple(next_state)][best_next_action]


               td_delta = td_target - Q[tuple(state)][action]


               Q[tuple(state)][action] += alpha * td_delta


               #print('Q',Q[(1,10,1)])
               alpha= alpha**t

               if done: #or t==1000:
                   #if done:
                   print('found the solution',state,'prev',prev_state)
                   z=input()
                       #print (Q)
                   break

               state = next_state


       return Q, stats
   Q, stats = q_learning(env, 500)

   env.reset()
   def follow_greedy_policy(Q,start_state):
       env.reset()
       done1 =False
       print('start state1',start_state)
       while ( not done1):
           print('start state2',start_state)

           best_action1 = np.argmax(Q[tuple(start_state)])
           print('start state3', start_state)
           print('best action', best_action1)
           next_state1, reward1, done1 = env.step(best_action1)

           print ('nextstate',next_state1)
           start_state=next_state1
           print('state:',start_state)

   plotting.plot_episode_stats(stats)
   pdb.set_trace()
   follow_greedy_policy(Q, env.state)

