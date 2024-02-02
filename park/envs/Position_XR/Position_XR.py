import numpy as np
from collections import deque

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
# from park.envs.abr_sim.trace_loader import \
#     load_traces, load_chunk_sizes, sample_trace, get_chunk_time


import park.envs.Position_XR.ChannelGen as ChannelGen
import park.envs.Position_XR.VideoGen as VideoGen

class PositionXREnv(core.Env):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id = 3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id = Hyg1G2AqtQ

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id = 2787486
    """
    def __init__(self, BS_number=4, User_number = 10, bandwidth_total = 50e7,Radius = 1000, rmin = 30 ):
        # set up seed
        self.seed(config.seed)
        # load all trace files
        self.Radius=Radius

        self.BS_number = BS_number
        self.User_number = User_number       
        self.bandwidth_total = bandwidth_total


        self.Radius = Radius
        self.rmin = rmin # Min distance between user and BS
        self.Channel=ChannelGen.Channel(self.BS_number,self.User_number,self.bandwidth_total,self.Radius,self.rmin)
        self.channel_gain,self.SNR,self.User_sin_theta, self.User_cos_theta = self.Channel.GetChannelGain()
        
        self.position_requirement=np.ones((1,self.User_number))*0.1
        self.position_sensitivity=1

        self.Bitrate_is_discrete = False
        self.Bitrate_set = np.array([1e5,1e7])
        self.max_bitrate = 1e7
        self.Frame_rate = np.ones((1,self.User_number))*60
        self.Option = "GOP" # "slice" is not defined yet.
        self.Video = VideoGen.Video(User_number=self.User_number,Bitrate_is_discrete=self.Bitrate_is_discrete, Bitrate_set=self.Bitrate_set , Frame_rate = self.Frame_rate, Option = self.Option)
        self.video_bitrate_frame = np.zeros((1,self.User_number))

        self.time_len=1000
        self.video_bitrate_history=np.zeros((self.time_len,self.BS_number,self.User_number))
        self.time_cnt=0
        
        self.beta_1 = 0.1
        self.beta_2 = 0.4
        # observation and action space
        self.setup_space()


    def observe(self):
        # obs_arr=np.zeros((self.BS_number,self.User_number))
        obs_arr=np.log10(self.channel_gain.copy())/10

        return obs_arr.reshape(-1)

    def reset(self):
        self.Video.reset()
        self.Channel.reset()

        self.channel_gain, self.SNR, self.User_sin_theta, self.User_cos_theta = self.Channel.GetChannelGain()
        self.video_bitrate_frame=self.Video.GetBitratePerFrame()
        
        self.video_bitrate_history=np.zeros((self.time_len,self.BS_number,self.User_number))
        self.video_bitrate_history[0,:]=self.video_bitrate_frame[:,0]
        self.time_cnt=0
        self.last_bitrate=np.zeros((1,self.User_number))
        self.done=0
        return self.observe()
    

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * ( self.BS_number * self.User_number))
        self.obs_high = np.array([1]* ( self.BS_number * self.User_number))
        self.observation_space = spaces.Box(
            low = self.obs_low, high = self.obs_high, dtype = np.float32)

        self.action_low = np.array([0] * (self.User_number *2 + self.BS_number+self.User_number*self.BS_number))
        self.action_high = np.array([1]* (self.User_number *2 + self.BS_number+self.User_number*self.BS_number))               
        self.action_space = spaces.Box(
            low = self.action_low, high = self.action_high, dtype = np.float32)
        
    def get_action(self, action):
        association_num=self.User_number*self.BS_number
        action_association=np.zeros((self.BS_number,self.User_number))
        action_association=action[:association_num].reshape((self.BS_number,self.User_number))
        
        action_bitrate=action[association_num:association_num+self.User_number].reshape((1,self.User_number))
        action_BW_positioning=action[association_num+self.User_number:association_num+self.User_number+self.BS_number].reshape((self.BS_number,1))
        action_BW_commnunication=action[-self.User_number:].reshape((1,self.User_number))
        
        action_bitrate=np.clip(0,1,action_bitrate)
        action_BW_positioning=np.clip(0,1,action_BW_positioning)
        action_BW_commnunication=np.clip(0,1,action_BW_commnunication)
        
        return action_association,action_bitrate,action_BW_positioning,action_BW_commnunication
    
    def action_association_best(self, action):
        action_association=np.zeros((1,self.User_number))
        for i in range(self.User_number):
            action_association[0,i]=np.argmax(action[:,i])
            if (action_association[0,i]>=self.BS_number):
                print("action_association error")
        action_association=action_association.astype(np.int16)
        return action_association
            
    def BW_Norm(self,action_association,action_BW_positioning,action_BW_commnunication):
        for i in range(self.BS_number):
            ind=np.where(action_association[0,:]==i)
            sum_data=action_BW_positioning[i,0]+np.sum(action_BW_commnunication[0,ind])
            action_BW_commnunication[0,ind]=action_BW_commnunication[0,ind]/sum_data
            action_BW_positioning[i,0]=action_BW_positioning[i,0]/sum_data
            
            
            
    def step(self, action):

        action_association, action_bitrate,action_BW_positioning,action_BW_commnunication=self.get_action(action)
        
        action_association=self.action_association_best(self.SNR)
        # action_association=self.action_association_best(action_association)
        self.BW_Norm(action_association,action_BW_positioning,action_BW_commnunication)
        action_BW_positioning=action_BW_positioning*self.bandwidth_total
        
        
        self.Video.SetBitrate(action_bitrate*self.max_bitrate)
        
        
        
        position_error=np.zeros((1,self.User_number))
        for i in range(self.User_number):
            position_Jn=np.zeros((2,2))
            for j in range(self.BS_number):
                ctheta=self.User_cos_theta[j,i]
                stheta=self.User_sin_theta[j,i]
                G=np.array([[ctheta*ctheta, ctheta*stheta], [ctheta*stheta, stheta*stheta]])
                position_Jn += 8*np.pi*np.pi/3e8/3e8*(action_BW_positioning[j,0]*action_BW_positioning[j,0]*self.SNR[j,i]*G) 
            position_error[0,i]=np.trace( np.linalg.inv(position_Jn))
            
   
   
        PQoE=1/(1-np.exp(-self.position_sensitivity))*(1-np.exp(-self.position_sensitivity*self.position_requirement/position_error))
        
        #是否发送成功：
        SINR_communication=np.zeros((1,self.User_number))
        for i in range(self.User_number):
            SINR_communication[0,i]=self.SNR[action_association[0,i],i]
        UE_rate=1e-3*action_BW_commnunication*self.bandwidth_total*np.log2(1+SINR_communication)
        # print("SINR_communication",SINR_communication)
        # print("UE_rate",UE_rate)
        trans_success=np.zeros((1,self.User_number))
        trans_success[self.video_bitrate_frame-UE_rate<0]=1
        Tvalue=(action_bitrate-self.beta_1*np.abs(action_bitrate-self.last_bitrate))
        TQoE=trans_success*Tvalue
        
        reward=np.sum((1-self.beta_2)*PQoE+self.beta_2*TQoE)
        self.last_bitrate=1e5*np.ones((1,self.User_number))
        
 
 
        self.channel_gain, self.SNR, self.User_sin_theta, self.User_cos_theta = self.Channel.GetChannelGain()
        self.video_bitrate_frame=self.Video.GetBitratePerFrame()        
        self.last_bitrate=action_bitrate
        
        self.time_cnt +=1 
        if (self.time_cnt==self.time_len):    
            self.done=1
        return self.observe(), reward, self.done, \
               {'bitrate': action_bitrate,
                'reward': reward,
                'PQoE': PQoE,
                'TQoE':TQoE}
