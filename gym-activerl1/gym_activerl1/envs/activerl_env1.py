import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding

class ActiverlEnv1(gym.Env):
  '''This is an environment to simulate application of reinforcement learning for active learning.
  In this environment there are 2 clusters which are separated by a theta. The agents job will
  be to predict this theta in less number of actions. The action is in a discrete action space
  and ranges between 0-1. The valid observation space is 2 dimensional and lies between 0-1 and
  the 1st element being smaller than the 2nd element. This environment can be visualized using the
  render function.
  '''
  metadata = {'render.modes': ['human']}
  
  def __init__(self):
    self.reward = int()
    self.theta_n = 0.5
    self.prev_theta = float()
    self.all_actions = list()
    self.theta = np.random.rand()
    self.State = np.array([0,1],dtype=np.float32)
    self.epsilon = 0.001
    self._actions = np.arange(0,1,0.001)
    high = np.array([1.,1.],dtype=np.float32)
    low = np.array([0.,1.],dtype=np.float32)
    self.action_space = spaces.Discrete(len(self._actions))
    self.observation_space = spaces.Box(low=low,high=high, dtype=np.float32)
    
  def _get_actions(self):
    actions=np.arange(0,1,0.001)
    return actions

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self, action):
    action = self._actions[action]
    self._take_action(action)
    reward = self._get_reward()
    current_State = self.getState()
    episode_over = self._episode_end(self.epsilon)
    return current_State,reward,episode_over,{'Theta':self.theta,'Predicted Theta':self.theta_n}
    
  def getState(self):
    return self.State

  def _episode_end(self,epsilon):
    if(np.absolute(self.theta-self.theta_n) <= epsilon)or(len(self.all_actions)>=1000):
      return True
    else:
      return False
  
  def reset(self):
    self.theta = np.random.rand()
    self.State = np.array([0,1])
    self.theta_n = 0
    self.prev_theta = float()
    self.all_actions=list()
    self.reward = 0
    return self.getState()

  def render(self, mode='human', close=False):
    y_array = [1 if i<self.theta else 0 for i in self.all_actions]
    x = list(self.State)
    plt.step(x=x,y=(1,0),where='mid',label='Theta hat')
    plt.step(x=[0,self.theta,1],y=(1,0,0),where='post',label='Theta')
    plt.plot(self.all_actions,y_array,'go')
    plt.legend()
    plt.show()
    plt.pause(0.1)
    plt.close('all')

  def _take_action(self,action):
    if action in self.all_actions:
      self.State = self.State
      self.theta_n = self.prev_theta
    else:
      self.all_actions.append(action)
      if (action < self.theta):
        if(action<self.State[0]):
          self.State = self.State
        else:
          self.State = np.array([action,self.State[1]],dtype=np.float32)
      else:
        if(action>self.State[1]):
          self.State = self.State
        else:
          self.State = np.array([self.State[0],action],dtype=np.float32)
      self.prev_theta = self.theta_n
      self.theta_n = (self.State[0] + self.State[1])/2

  def _get_reward(self):
    if (np.absolute(self.theta-self.prev_theta) > np.absolute(self.theta-self.theta_n)):
      self.reward += +1
    else:
      self.reward += -1
    return self.reward
	
      
