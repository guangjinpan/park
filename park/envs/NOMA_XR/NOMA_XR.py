import numpy as np
from collections import deque

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.envs.abr_sim.trace_loader import \
    load_traces, load_chunk_sizes, sample_trace, get_chunk_time


import ChannelGen
import VideoGen

class NOMAXREnv(core.Env):
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
    def __init__(self,User_number = 4,channel_number = 10,bandwidth_total = 10e6,Radius = 1000, rmin = 30 ):
        # observation and action space
        self.setup_space()
        # set up seed
        self.seed(config.seed)
        # load all trace files



        self.User_number = User_number
        self.channel_number = channel_number        
        self.bandwidth_total = bandwidth_total


        self.Radius = Radius
        self.rmin = rmin # Min distance between user and BS
        self.Bw = np.ones((self.User_number,1))*self.bandwidth_total/self.channel_number
        Noise_Hz = 10**((-174-30)/10) # Noise value per Hertz = -174 dBm/Hz
        self.noise = np.ones((self.User_number,self.channel_number))*Noise_Hz*self.bandwidth_total/self.channel_number
        self.Channel=ChannelGen.Channel(self.User_number,self.channel_number,self.bandwidth_total,self.Radius,self.rmin)
        self.channel_gain = self.Channel.GetChannelGain()

        self.Bitrate_is_discrete = False
        self.Bitrate_set = np.array([1e5,1e7])
        self.Frame_rate = np.ones((self.User_number,1))*60
        self.Option = "GOP" # "slice" is not defined yet.
        self.Video = VideoGen.Video(User_number=self.User_number,Bitrate_is_discrete=self.Bitrate_is_discrete, Bitrate_set=self.Bitrate_set , Frame_rate = self.Frame_rate, Option = self.Option)
        self.video_bitrate_frame = np.zeros((self.User_number,1))


        self.time_len=1000
        self.video_bitrate_history=np.zeros((self.time_len,self.User_number))




    def observe(self):
        obs_arr=np.zeros((self.User_number,self.channel_number+1))
        obs_arr[:,:self.channel_number]=self.channel_gain
        obs_arr[:,-1]=self.video_bitrate_frame[:,0]

        return obs_arr.reshape(-1)

    def reset(self):
        self.Video.reset()
        self.Channel.reset()

        self.channel_gain = self.Channel.GetChannelGain()

        self.video_bitrate_history=np.zeros((self.time_len,self.User_number))
        self.video_bitrate_frame=self.Video.GetBitratePerFrame()
        self.video_bitrate_history[0,:]=self.video_bitrate_frame[:,0]
        
        self.last_bitrate=1e6
        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * (self.User_number+ self.channel_number * self.User_number))
        self.obs_high = np.array([1]* (self.User_number+ self.channel_number * self.User_number))
        self.observation_space = spaces.Box(
            low = self.obs_low, high = self.obs_high, dtype = np.float32)

        self.action_low = np.array([0] * (self.User_number+ self.channel_number * self.User_number))
        self.action_high = np.array([1]* (self.User_number+ self.channel_number * self.User_number))               
        self.action_space = spaces.Box(
            low = self.action_low, high = self.action_high, dtype = np.float32)

    def step(self, action):

        # 0 < =  action < num_servers
        # assert self.action_space.contains(action)
        action=action.reshape((self.User_number,self.channel_number+1))
        action_bitrate=action[:,-1]
        action_power=action[:,:self.channel_number]

        

        self.last_bitrate=action_bitrate
        return self.observe(), reward, done, \
               {'bitrate': self.bitrate_map[action],
                'stall_time': rebuffer_time,
                'bitrate_change': bitrate_change}
