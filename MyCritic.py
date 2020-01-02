from keras import layers, models, optimizers, regularizers
from keras import backend as K
from keras.layers import Flatten, Concatenate, LeakyReLU
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import RandomUniform, Zeros
from keras_radam import RAdam
import numpy as np

SEED = 123456
np.random.seed(SEED)

def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': layers.Activation(mish)})

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr, network):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.network = network
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        #actions_norm = layers.BatchNormalization()(actions) , kernel_regularizer=layers.regularizers.l2(regularization)
        states_actions = Concatenate()([states, actions])
        net_SA = layers.BatchNormalization()(states_actions)
        
        alp = 0.1
                
        net = layers.Dense(units=self.network[0])(net_SA)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        net = layers.Dense(units=self.network[1])(net)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        net = layers.Dense(units=self.network[2])(net)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        #net = layers.Activation('Mish')(net) 

        # Combine state and action pathways
        # Add final output layer to produce action values (Q values)
        #Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=RandomUniform())(net)Zeros()
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=RandomUniform())(net)
        #Q_values = layers.LeakyReLU(alpha=alp)(Q_values)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        #optimizer = RAdam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)