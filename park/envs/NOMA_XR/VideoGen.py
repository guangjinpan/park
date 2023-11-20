# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:56:35 2016

@author: lsalaun

© 2016 - 2020 Nokia
Licensed under Creative Commons Attribution Non Commercial 4.0 International
SPDX-License-Identifier: CC-BY-NC-4.0

"""

import math
import numpy as np

import scipy.stats as stats


class Video():
    def __init__(self, User_number = 4, Bitrate_is_discrete = False, Bitrate_set = np.array([1e5,1e7]), Frame_rate = np.ones((4,1))*60, Option = "GOP"):
        self.User_number = User_number

        self.Bitrate_set = Bitrate_set
        if Bitrate_is_discrete:
            self.Bitrate_max = max(Bitrate_set)
            self.Bitrate_min = min(Bitrate_set)     
        else:
            self.Bitrate_min = Bitrate_set[0]
            self.Bitrate_max = Bitrate_set[1]

        self.Option = Option
        self.Frame_rate = Frame_rate

        # slice based model, according to 3gpp TR-38.838
        self.K = 8
        self.alpha = 2

        self.Bitrate = np.ones((self.User_number,1))*1e6
        self.Bitrate_per_frame = self.Bitrate/self.Frame_rate
        self.Frame_type = np.random.randint(0,self.K,(self.User_number,1))




    def generateFrame(self,):
        
        if self.Option == "GOP":
            for i in range(self.User_number):
                if self.Frame_type[i,0] == 0:
                    #I—frame
                    mu = 1e6*self.alpha/(self.K-1+self.alpha)*self.K/self.Frame_rate[i,0]
                else:
                    #P—frame                    
                    mu = 1e6*(self.K-1)/(self.K-1+self.alpha)*self.K/(self.Frame_rate[i,0]*(self.K-1))
                sigma = 0.105*1e6/self.Frame_rate[i,0]*self.alpha/(self.K-1+self.alpha)
                lower = 0.5*mu
                upper = 1.5*mu
                bitrate = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma).rvs(1)
                self.Bitrate_per_frame[i,0] = bitrate


                self.Frame_type[i,0]=(self.Frame_type[i,0]+1)%self.K

        

    def GetBitratePerFrame(self,):
        self.generateFrame()
        print(1e6/60,self.Bitrate_per_frame)
        Bitrate_per_frame =  self.Bitrate_per_frame/1e6*self.Bitrate
        return Bitrate_per_frame

    def SetBitrate(self,Bitrate_set):
        self.Bitrate=Bitrate_set

    def reset(self,):
        self.Frame_type = np.random.randint(0,self.K,(self.User_number,1))
      




# video = Video()
# ratesum=0
# for ii in range(100):
#     ratesum+=np.sum(video.GetBitratePerFrame())

# print(ratesum/100/4*60)
