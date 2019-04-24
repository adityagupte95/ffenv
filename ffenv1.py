

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


class ReplayMemory:
    def __init__(self,n):
        self.size=n
        self.expBuffer=[]

    # Circular memory
    def push(self,exp):
        if len(self.expBuffer)<self.size:
            self.expBuffer=[exp]+self.expBuffer
        else:
            self.expBuffer.pop(0)
            self.expBuffer=[exp]+self.expBuffer

    # Check if buffer has sufficient experience
    def isReady(self):
        return len(self.expBuffer)>=32

    def sampleBatch(self,sz=None):
        if sz==None:
            sz=32
        idxs=[np.random.randint(0,len(self.expBuffer)) for _ in range(len(self.expBuffer))]
        return [self.expBuffer[idx] for idx in idxs]




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


        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((spaces.Box(low=np.array([0, 0]), high=np.array([2, 2]), dtype=np.int), spaces.Discrete(2)))
        self.state = None
#       self.t1 = 40 * np.pi / 180 #       self.fngrwdth = 1.5
#       self.palmwdth = 6.0

#       self.t2 = 138 * np.pi / 180
        #self.theta1_solution=None
#       self.theta2_solution=None
        self.reward= None
        self.d1_d=2
        self.d2_d=2
        self.fs_d= 1
#       self.t1_max=140
#       self.t2_min=40


    def reset(self):
        self.state = [1, 1, 1]
        return self.state

    def step(self, action):
        prev_state= copy.copy(self.state)
        rew= lambda st: 100 if st[0]==self.d1_d and st[1]==self.d2_d and st[2]==self.fs_d else 0
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

            elif action == 1:
                if self.state[1] == 12:
                    self. reward = rew(self.state)
                    self.state=self.state
                else:
                    self.state = [self.state[0] , self.state[1]+1, self.state[2]]
                    self.reward=rew(self.state)

            elif action == 2:
                self.state[2] = 1
                self.reward= rew(self.state)


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

        if self.state[0:3] == [self.d1_d, self.d2_d,self.fs_d] :
            self.done =True
        else:
            self.done= False
        self.state=[max(min(self.state[0],2),0),max(min(self.state[1],2),0),max(min(self.state[2],1),0)]
        print((prev_state,action, self.state ,self.reward, self.done),)
        return (prev_state,action, self.state ,self.reward, self.done)


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
       step_number=20
       Q = defaultdict(lambda: np.ones(env.action_space.n))
       exprep=ReplayMemory(2048)

       # Keeps track of useful statistics
       stats = plotting.EpisodeStats(
           episode_lengths=np.zeros(num_episodes),
           episode_rewards=np.zeros(num_episodes))

       # The policy we're following
       policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

       for i_episode in range(num_episodes):

           print ('episode no.',i_episode)

           # Reset the environment and pick the first action
           state = env.reset()

           # One step in the environment
           # total_reward = 0.0
           for t in itertools.count():


               # Take a step
               action_probs = policy(state)
               action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
               tup = env.step(action)
               # exprep.push(tup)

               
               # Update statistics
               stats.episode_rewards[i_episode] += tup[3]
               stats.episode_lengths[i_episode] = t
               # if exprep.isReady():
               #     B=exprep.sampleBatch(32)
               # for j in B:
                   # TD Update

                       print('Q for ',j[0],'before update',Q[tuple(j[0])])
                       prev_state,action, next_state, reward, done= j[0],j[1],j[2],j[3],j[4]
                       best_next_action = np.argmax(Q[tuple(next_state)])


                       td_target = reward + discount_factor * Q[tuple(next_state)][best_next_action] if not done else reward


                       td_delta = td_target - Q[tuple(prev_state)][action]


                       Q[tuple(prev_state)][action] += alpha * td_delta

                       print('Q for ', j[0],'after update',Q[tuple(j[0])])


                   #print('Q',Q[(1,10,1)])
                   #alpha= alpha**t

               if tup[-1]  :#or t==step_number:
                   if tup[-1]:
                       print('found the solution',tup[2],'prev',tup[0])
                   # z=input()
                       #print (Q)
                   break

               state = tup[2]


       return Q, stats
   Q, stats = q_learning(env, 100)

   env.reset()
   def follow_greedy_policy(Q,start_state):
       env.reset()
       done =False
       #print('start state1',start_state)
       step=0
       while ( not done and step<=10):
           print('start state2',start_state)

           best_action1 = np.argmax(Q[tuple(start_state)])
           # print('start state3', start_state)
           print('best action', best_action1)
           tup = env.step(best_action1)
           prev_state, action, next_state, reward, done = tup[0], tup[1], tup[2], tup[3], tup[4]
           print('Q',Q[tuple(prev_state)])
           print ('nextstate',next_state)
           start_state=next_state
           print('state:',start_state)
           step+=1
   plotting.plot_episode_stats(stats)

   follow_greedy_policy(Q, env.state)

