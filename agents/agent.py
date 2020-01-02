import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
from collections import namedtuple, deque
from Actor import Actor
from MyCritic import Critic
from Custom import Custom
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Noise import OUNoise

SEED = 123456
import random as rn
from tensorflow import set_random_seed

np.random.seed(SEED)
set_random_seed(SEED)
rn.seed(SEED)


# Transform train_on_batch return value
# to dict expected by on_batch_end callback
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        #self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.memory = [] # list
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def sortKey(self,e):
        return e.reward
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return rn.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, train=True):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        self.actor_lr =  1e-5 #.0001 
        self.critic_lr = 1e-4 #0.0000001 
        
        self.network = [128,256,128]
        
        self.train = train
        network = self.network
        actor_lr = self.actor_lr
        critic_lr = self.critic_lr 
        
        if(self.train):
            # Actor (Policy) Model
            self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, actor_lr, network)
            self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, actor_lr, network)

            # Critic (Value) Model
            self.critic_local = Critic(self.state_size, self.action_size, critic_lr, network)
            self.critic_target = Critic(self.state_size, self.action_size, critic_lr, network)

            # Initialize target model parameters with local model parameters
            self.critic_target.model.set_weights(self.critic_local.model.get_weights())
            self.actor_target.model.set_weights(self.actor_local.model.get_weights())

            # Noise process
            self.exploration_mu = 0 # Mean
            self.exploration_theta = 0.15 #.15 How fast variable reverts to mean
            self.exploration_sigma = 0.2 #.2 Degree of volatility
            self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

            # Replay memory
            self.buffer_size = 5000
            self.batch_size = 16
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
            self.targets = ReplayBuffer(self.buffer_size, self.batch_size)

            # Algorithm parameters
            self.gamma = 0.99  # discount factor
            self.tau = 0.01   # for soft update of target parameters
                    
            print("DDPG init", "Actor: ", actor_lr, "Critic: ", critic_lr)
            print("Tau: ", self.tau, "Sigma: ", self.exploration_sigma)
            print(self.actor_local.model.summary())
            print(self.critic_local.model.summary())

            # https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch?rq=1
            # Create the TensorBoard callback,
            # which we will drive manually
            self.tensorboard = keras.callbacks.TensorBoard(
              log_dir='logdir',
              histogram_freq=0,
              batch_size=self.batch_size,
              write_graph=True,
              write_grads=True
            )

            self.tensorboard.set_model(self.critic_local.model)
            self.summary_writer = tf.summary.FileWriter("scores")

            self.batch_id = 0
        
    def reset_episode(self):
        if(self.train):
            self.noise.reset()
            self.noise_arr = []
            self.noise_matrix = [0.,0.,0.,0.]
        
        state = self.task.reset()
        self.last_state = state
        return state
    
    def save_initial_weights(self):
        self.actor_local.model.save_weights('actor_local.h5')
        self.actor_target.model.save_weights('actor_target.h5')
        self.critic_local.model.save_weights('critic_local.h5')
        self.critic_target.model.save_weights('critic_target.h5')
                                              
    def load_initial_weights(self):
        self.actor_local.model.load_weights('actor_local.h5')
        self.actor_target.model.load_weights('actor_target.h5')
        self.critic_local.model.load_weights('critic_local.h5')
        self.critic_target.model.load_weights('critic_target.h5')
        
    def save_model(self):
        # Save the weights
        self.actor_local.model.save_weights('model_weights.h5')

    def load_weights(self, option=None):
        if(option==None):
            self.trained = Actor(self.state_size, self.action_size, self.action_low, 
                                 self.action_high, self.actor_lr, self.network)
            self.trained.model.load_weights('model_weights.h5')
        else:
            self.trained = Actor(self.state_size, self.action_size, self.action_low, 
                     self.action_high, self.actor_lr, self.network)
            self.trained.model.load_weights('weights-best.hdf5')
            print(self.trained.model.summary())
        
        
    def predict(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.trained.model.predict(state)[0]
        return action
        
    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if(len(self.memory) > self.batch_size*2):
            experiences = self.memory.sample()
            self.learn(experiences)
            
        if(len(self.memory) == self.buffer_size):
            self.memory.memory.clear()
            print("buffer cleared");
                
        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        noise = self.noise.sample()
        action = list(self.actor_local.model.predict(state)[0] + noise)
        
        return action, noise  # add some noise for exploration

    def learn(self,experiences): #experiences 
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        '''
        print("States", states.shape)
        print("actions", actions.shape)
        print("rewards", rewards.shape)
        print("dones", dones.shape)
        print("Next states", next_states.shape)
        '''
        # keep training actor local and critic local
        # use values from target model to update and train local
        # don't train target models, we soft update target

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        
        actions_next = self.actor_target.model.predict_on_batch(next_states) #target
                    
        #Actions predicted by target critic
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next]) #target

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        critic_loss = self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        actor_loss = self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        
        self.tensorboard.on_epoch_end(self.batch_id, named_logs(self.critic_local.model, [critic_loss]))
        self.batch_id += 1
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)