import numpy as np
from physics_sim import PhysicsSim
import math

# Go to location and Hover

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_hover = np.array([0., 0., 0.])
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        
        # The Visual bounds
        env_bounds = 10
        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])
        
        
        # '
        self.reward_stat = []
        self.reward_matrix = []
        
        
    def sigmoid(self,x):
          return 1 / (1 + math.exp(-x))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # 300 m / 300 m / 300 m
        
        #Input: pose (6) for NN
        reward = 0
        euler_reward = 0
        proximity_reward = 0
        
        position = self.sim.pose[:3]
        euler_angles = self.sim.pose[3:]

        #element wise square
        f = np.subtract(position,self.target_pos[:3])
        
        x2 = -f**2
        pos = []
        
        for i, n in enumerate(self.target_pos):
            pos.append(max(1, int(abs(n))))
        
        res = np.divide(x2,pos)
        res = np.add(res, 3)
        parabolic_reward = res.sum()
        
        #degrees_angular = np.multiply(self.sim.angular_v, (180/3.14159))
        
        '''
        print("POS: ", pos,"\r")
        print("x2: ", self.target_pos[:3], x2,"\r")
        print("reward: ", reward,"\r")
        print("Angular: ", self.sim.angular_v,"\r")
        print("-----", "\r")
        '''
        
        #For hovering close
        squared_dist = np.sum((position-self.target_pos)**2, axis=0)
        dist = np.sqrt(squared_dist)
        
        if(dist <= 3):
            euler_reward = -((euler_angles).sum()) + 2
            proximity_reward = (5-dist)*10
            
        #Bounds Penalty
        '''
        for ii in range(3):
            if position[ii] <= self.lower_bounds[ii]:
                penalty += -1
            elif position[ii] > self.upper_bounds[ii]:
                penalty += -1
        '''
        
        reward = [parabolic_reward,proximity_reward,euler_reward]

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        
        for _ in range(self.action_repeat):
            done, bounds = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += np.array((self.get_reward())).sum() 
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        self.reward_stat.append(self.get_reward())
        
        if(done):
            matrix = np.asmatrix(np.array(self.reward_stat)) 
            self.reward_matrix = np.array([matrix[:,0],matrix[:,1],matrix[:,2]])
            self.mean_reward_matrix = np.array([matrix[:,0].mean(),matrix[:,1].mean(),matrix[:,2].mean()])

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)  
        #clear stat
        self.reward_stat = []
        return state