from keras import layers, models, optimizers, regularizers
from keras import backend as K
from keras.layers import Flatten, Concatenate, LeakyReLU
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import RandomUniform, Zeros
from keras_radam import RAdam

def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': layers.Activation(mish)})

class Custom:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, lr, network):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.lr = lr
        self.network = [32,64,32]

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        alp = 0.1
        net_states = layers.BatchNormalization()(states)
        
        net = layers.Dense(units=self.network[0])(net_states)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        net = layers.Dense(units=self.network[1])(net)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        net = layers.Dense(units=self.network[2])(net)
        net = layers.LeakyReLU(alpha=alp)(net)
        
        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size,
            name='raw_actions', kernel_initializer=RandomUniform())(net)
        
        net = layers.Activation('sigmoid')(raw_actions)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states], outputs=actions)

        # Define loss function using action value (Q value) gradients
        #optimizer = RAdam()
        
        optimizer = optimizers.Adam()
        self.model.compile(loss='mse', optimizer=optimizer)
