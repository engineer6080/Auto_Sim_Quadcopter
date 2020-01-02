import numpy as np
from physics_sim import PhysicsSim
from quadsim import QuadSim
import controller
import math
import random as rn

SEED = 123456
np.random.seed(SEED)
rn.seed(SEED)

class M_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, factor=None):
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
        #self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        # Define the quadcopters
        QUADCOPTER={'q1':{'position':init_pose[:3],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
        self.sim = QuadSim(QUADCOPTER)
        
        self.action_repeat = 1
        
        # Stat reporting
        self.reward_arr = []
        # Store Rotor behaviors
        self.rotor_arr = []
        # Noise tracking
        self.noise_arr = []
        
        # Action space
        self.action_low = 4000
        self.action_high = 9000 #5500 #9000
        self.action_size = 4
        
        # Variable factor for debug
        self.factor = factor
        self.randomize = False
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.state_size = self.action_repeat * (self.get_state()).shape[0]#6
        
        self.init_pose = init_pose
        self.runtime = runtime
        self.reached = False
        self.dist = 0
        
        self.agent_rotor_speeds = [0,0,0,0]
        self.pid_rotor_speeds = [0,0,0,0]
                
        # Controller parameters
        CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000], #9000
                        'Tilt_limits':[-10,10],
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':0.18,
                        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
                        }
        
        self.ctr = controller.Controller_PID_Point2Point(self.sim.get_state,self.sim.get_time,self.sim.set_motor_speeds,
                                                      params=CONTROLLER_PARAMETERS,quad_identifier='q1')
        
    def hover(self):
        self.ctr.update_target(self.target_pos)
        self.ctr.update_yaw_target(0.)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def tan_reward(self, x):
        return -np.tanh((10*x)-3)
    
    def get_reward(self):
        reward = 0                
        position = np.copy(self.sim.get_position('q1'))
        velocity = self.sim.get_linear_rate('q1') 
        euler_angles = self.sim.get_orientation('q1')
        
        # Main reward
        self.dist = np.linalg.norm(position-self.target_pos)        
        difference = abs(np.subtract(self.pid_rotor_speeds,self.agent_rotor_speeds))
        normalized = sum(np.divide(difference,[5000,5000,5000,5000]))/4
        reward = self.tan_reward(normalized)
        
        # Reached target
        if(self.dist <= 1):
            self.hover()
            self.reached = True

        return [reward]
    
    def pretty(self, obj, labels=None):
        if(labels != None):
            for i, l in zip(obj,labels):
                print(l + " {:4.3f} ".format(i), end=" ")
        else:
            for i, n in enumerate(obj):
                print(str(i) + ": {:4.3f} ".format(n), end=" ")
        print("\n")
        
    # Changing state space
    def get_state(self):
        dist_err = np.subtract(self.sim.get_position('q1'),self.target_pos)
        orientation = self.sim.get_orientation('q1')
        angular = self.sim.get_angular_rate('q1')
        linear = self.sim.get_linear_rate('q1')
        vec = [dist_err, orientation, angular, linear]
        out = []
        for i, active in enumerate(self.factor):
            if(active == 1):
                out = np.concatenate((out, vec[i]))
        return np.array(out)
        
    def step(self, agent_rotor_speeds, noise=None):
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        pose_all = []
        done = False
        self.pid_rotor_speeds = self.ctr.update()
        self.agent_rotor_speeds = agent_rotor_speeds
        
        for _ in range(self.action_repeat):     
            self.sim.set_motor_speeds('q1', (self.pid_rotor_speeds)) # update the sim pose and velocities
            bounds = self.sim.update(1/50.)
            rewards = self.get_reward()
            
            # Run out of time or out of bounds
            if (self.sim.time > self.runtime):
                if bounds:
                    print("Out of bounds: ", self.sim.pose[:3])
                done = True
                
            reward += sum(rewards)
            pose_all.append(self.get_state())
                
            # For plotting
            self.rotor_arr.append(agent_rotor_speeds)
            self.reward_arr.append(rewards)
            self.noise_arr.append(noise)
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()        
        self.reached = False
        # -10 to 10 (x,y)
        # 0 to 10 (z)
        if(self.randomize): # New start location
            rx = rn.randint(-1,2)
            ry = rn.randint(-1,2)
            rz = rn.randint(-1,2)
            new_start = np.add([rx,ry,rz], self.init_pose[:3])
            print("New start: ", new_start)
            self.sim.set_position('q1', new_start)
        else:
            self.sim.set_position('q1', self.init_pose[:3])
            
        self.sim.set_orientation('q1', self.init_pose[3:])
        self.sim.time = 0.
        state = np.concatenate([self.get_state()] * self.action_repeat)        
        # Refresh inner episodic stats
        self.reward_arr = []
        self.rotor_arr = []
        self.noise_arr = []
        return state