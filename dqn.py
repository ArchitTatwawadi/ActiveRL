from collections import deque
import numpy as np
import random
import keras

class DQNAgent:
  '''This class defines various functions required by the agent to implement a DQN algorithm.
     @param : env: environment to implement this agent.
     @param : model: The model which the agent will learn.
  '''
  def __init__(self,env,model):
    self.env = env
    self.model = model
    self.target_model = model
    self.memory = deque(maxlen=5000)
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.01
    self.tau = 0.05
    
  def act(self, state):
    #This implements epsilon greedy Q algorithm to choose action.
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)
    if np.random.random_sample() < self.epsilon:
      return self.env.action_space.sample()
    return np.argmax(self.model.predict(state))

  def remember(self, state, action, reward, new_state, done):
    #It stores the current (state,action) for replay
    self.memory.append([state, action, reward, new_state, done])
  
  def forget(self):
    #It flushes out the memory and reinitialize the memory block
    self.memory = deque(maxlen=5000)

  def replay(self):
    #It reiterates through previously remembered memories to update the Q table
    batch_size = 32
    if len(self.memory) < batch_size:
      return
    samples = random.sample(self.memory, batch_size)
    for sample in samples:
      state, action, reward, new_state, done = sample
      state = state.reshape(1,2)
      target = self.target_model.predict(state)
      if done:
        target[0][action] = reward
      else:
        Q_future = np.argmax(self.target_model.predict(new_state))
        target[0][action] = reward + Q_future * self.gamma
      self.model.fit(state, target, epochs=1, verbose=0)

  def target_train(self):
    #It trains the input model
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(target_weights)):
      target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    self.target_model.set_weights(target_weights)

  def save_model(self, filename):
    #It saves the learned model.
    self.model.save(filename)

  
