import gym
from gym import wrappers as w
from gym.spaces import Discrete, Box
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
from typing import List, Any

from policies import MLP_heb, CNN_heb, WLNHNN
from hebbian_weights_update import *
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame
from numba.typed import List

def reshape_coeffs(coeffs, env):
    rc  =None
    c = 0
    if env == 'AntBulletEnv-v0':
        size = (28+32+64+8, 5)
        rc = np.zeros(size, dtype=np.float32)
        for n in range(28):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c+1]
            rc[n, 3] = coeffs[c+2]
            rc[n, 4] = coeffs[c+3]
            c += 4
        for n in range(28, 64+28):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c + 1]
            rc[n, 2] = coeffs[c + 2]
            rc[n, 3] = coeffs[c + 3]
            rc[n, 4] = coeffs[c + 4]
            c += 5
        for n in range(28+128, 64+28+32):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c + 1]
            rc[n, 2] = coeffs[c + 2]
            rc[n, 3] = coeffs[c + 3]
            rc[n, 4] = coeffs[c + 4]
            c += 5
        for n in range(28+32+64, 32+28+64+8):
            rc[n, 0] = coeffs[c]
            rc[n, 2] = coeffs[c + 1]
            rc[n, 3] = coeffs[c + 2]
            rc[n, 4] = coeffs[c + 3]
            c += 4
    else:
        size = (648 + 128 + 64 + 3, 5)
        rc = np.zeros(size, dtype=np.float32)
        for n in range(648):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c + 1]
            rc[n, 3] = coeffs[c + 2]
            rc[n, 4] = coeffs[c + 3]
            c += 4
        for n in range(648, 128 + 648):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c + 1]
            rc[n, 2] = coeffs[c + 2]
            rc[n, 3] = coeffs[c + 3]
            rc[n, 4] = coeffs[c + 4]
            c += 5
        for n in range(648 + 128, 128 + 648 + 64):
            rc[n, 0] = coeffs[c]
            rc[n, 1] = coeffs[c + 1]
            rc[n, 2] = coeffs[c + 2]
            rc[n, 3] = coeffs[c + 3]
            rc[n, 4] = coeffs[c + 4]
            c += 5
        for n in range(648 + 128 + 64, 128 + 648 + 64 + 3):
            rc[n, 0] = coeffs[c]
            # rc[n, 1] = coeffs[c + 1]
            rc[n, 2] = coeffs[c + 1]
            rc[n, 3] = coeffs[c + 2]
            rc[n, 4] = coeffs[c + 3]
            c += 4
    return rc

def fitness_hebb(hebb_rule : str, environment : str, init_weights = 'uni' , *evolved_parameters) -> float:
    """
    Evaluate an agent 'evolved_parameters' controlled by a Hebbian network in an environment 'environment' during a lifetime.
    The initial weights are either co-evolved (if 'init_weights' == 'coevolve') along with the Hebbian coefficients or randomly sampled at each episode from the 'init_weights' distribution. 
    Subsequently the weights are updated following the hebbian update mechanism 'hebb_rule'.
    Returns the episodic fitness of the agent.
    """
    
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            if init_weights == 'xa_uni':  
                torch.nn.init.xavier_uniform(m.weight.data, 0.3)
            elif init_weights == 'sparse':  
                torch.nn.init.sparse_(m.weight.data, 0.8)
            elif init_weights == 'uni':  
                torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
            elif init_weights == 'normal':  
                torch.nn.init.normal_(m.weight.data, 0, 0.024)
            elif init_weights == 'ka_uni':  
                torch.nn.init.kaiming_uniform_(m.weight.data, 3)
            elif init_weights == 'uni_big':
                torch.nn.init.uniform_(m.weight.data, -1, 1)
            elif init_weights == 'xa_uni_big':
                torch.nn.init.xavier_uniform(m.weight.data)
            elif init_weights == 'ones':
                torch.nn.init.ones_(m.weight.data)
            elif init_weights == 'zeros':
                torch.nn.init.zeros_(m.weight.data)
            elif init_weights == 'default':
                pass
            
    # Unpack evolved parameters
    try: 
        hebb_coeffs, initial_weights_co = evolved_parameters
    except: 
        hebb_coeffs = evolved_parameters[0]

    # Intial weights co-evolution flag:
    coevolve_init = True if init_weights == 'coevolve' else False
    

    with torch.no_grad():
                    
        # Load environment
        try:
            env = gym.make(environment, verbose = 0)
        except:
            env = gym.make(environment)
            
        # env.render()  # bullet envs
        
        # For environments with several intra-episode lives -eg. Breakout-
        try: 
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireEpisodicLifeEnv(env)
        except: 
            pass

        # Check if selected env is pixel or state-vector 
        if len(env.observation_space.shape) == 3:     # Pixel-based environment
            pixel_env = True
            env = w.ResizeObservation(env, 84)        # Resize and normilise input   
            env = ScaledFloatFrame(env)
            input_channels = 3
        elif len(env.observation_space.shape) == 1:   
            pixel_env = False
            input_dim = env.observation_space.shape[0]
        elif len(env.observation_space.shape) == 0:   
            pixel_env = False
            input_dim = env.observation_space.n
            
        # Determine action space dimension
        if isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError('Only Box and Discrete action spaces supported')
        
        # Initialise policy network: with CNN layer for pixel envs and simple MLP for state-vector envs
        if pixel_env == True: 
            p = CNN_heb(input_channels, action_dim)      
        else:
            nl = List()
            for i in [input_dim, 128,64, action_dim]:
                nl.append(i)
            p = WLNHNN(nl) #MLP_heb(input_dim, action_dim)
            p.set_hrules(hebb_coeffs.flatten())
        
        
        # Initialise weights of the policy network with an specific distribution or with the co-evolved weights
        if coevolve_init:
            nn.utils.vector_to_parameters( torch.tensor (initial_weights_co, dtype=torch.float32 ),  p.parameters() )
        else:       
            # Randomly sample initial weights from chosen distribution
            #p.apply(weights_init)
            
             # Load CNN paramters
            if pixel_env:
                cnn_weights1 = initial_weights_co[:162]
                cnn_weights2 = initial_weights_co[162:]
                list(p.parameters())[0].data = torch.tensor(cnn_weights1.reshape((6,3,3,3))).float()
                list(p.parameters())[1].data = torch.tensor(cnn_weights2.reshape((8,6,5,5))).float()
        # p = p.float()
        
        # Unpack network's weights
        # if pixel_env:
        #     weightsCNN1, weightsCNN2, weights1_2, weights2_3, weights3_4 = list(p.parameters())
        # else:
        #     weights1_2, weights2_3, weights3_4 = list(p.parameters())
            
        
        # Convert weights to numpy so we can JIT them with Numba
        # weights1_2 = weights1_2.detach().numpy()
        # weights2_3 = weights2_3.detach().numpy()
        # weights3_4 = weights3_4.detach().numpy()
        
        observation = env.reset() 
        if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)       

        # Burnout phase for the bullet quadruped so it starts off from the floor
        if environment == 'AntBulletEnv-v0':
            action = np.zeros(8)
            for _ in range(40):
                __ = env.step(action)        
        
        # Normalize weights flag for non-bullet envs
        normalised_weights = False if environment[-12:-6] == 'Bullet' else True


        # Inner loop
        neg_count = 0
        rew_ep = 0
        t = 0
        while True:
            
            # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
            if isinstance(env.observation_space, Discrete): 
                observation = (observation == torch.arange(env.observation_space.n)).float()
            nl = List()
            for o in observation:
                nl.append(o)
            o3 = p.call(observation)
            o3 = torch.tensor(o3)
            # print(o3)
            o0 = []#o0.numpy()
            o1 = []#o1.numpy()
            o2 = []#o2.numpy()
            # print(0,o0)
            # print(1,o1)
            # print(2,o2)
            # print(3,o3)
            
            # Bounding the action space
            if environment == 'CarRacing-v0':
                action = np.array([ torch.tanh(o3[0]), torch.sigmoid(o3[1]), torch.sigmoid(o3[2]) ])
                o3 = action
            elif environment[-12:-6] == 'Bullet':
                o3 = torch.tanh(o3).numpy()
                action = o3
            else: 
                if isinstance(env.action_space, Box):
                    action = o3.numpy()                        
                    action = np.clip(action, env.action_space.low, env.action_space.high)  
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(o3).numpy()
                o3 = o3.numpy()

            
            # Environment simulation step
            # print(action)
            observation, reward, done, info = env.step(action)  
            if environment == 'AntBulletEnv-v0': reward = env.unwrapped.rewards[1] # Distance walked
            rew_ep += reward
            
            # env.render('human') # Gym envs
            
            if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)
                                       
            # Early stopping conditions
            if environment == 'CarRacing-v0':
                neg_count = neg_count+1 if reward < 0.0 else 0
                if (done or neg_count > 20):
                    break
            elif environment[-12:-6] == 'Bullet':
                if t > 200:
                    neg_count = neg_count+1 if reward < 0.0 else 0
                    if (done or neg_count > 30):
                        break
            else:
                if done:
                    break
            # else:
            #     neg_count = neg_count+1 if reward < 0.0 else 0
            #     if (done or neg_count > 50):
            #         break
            
            t += 1
            
            
            #### Episodic/Intra-life hebbian update of the weights
            if hebb_rule == 'A': 
                weights1_2, weights2_3, weights3_4 = hebbian_update_A(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_in_and_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in_and_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'NB':
                #rc = reshape_coeffs(hebb_coeffs, environment)
                weights1_2, weights2_3, weights3_4 = hebbian_update_NB(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule=='NU':
                pass
            else:
                raise ValueError('The provided Hebbian rule is not valid')
                

            # Normalise weights per layer
            if normalised_weights == True:
                (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
                list(p.parameters())[a].data /= list(p.parameters())[a].__abs__().max()
                list(p.parameters())[b].data /= list(p.parameters())[b].__abs__().max()
                list(p.parameters())[c].data /= list(p.parameters())[c].__abs__().max()
        
            
        env.close()
    print(rew_ep)
    return rew_ep
    # return max(rew_ep, 0)

