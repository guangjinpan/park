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
    def __init__(self, User_number = 4, Bitrate_is_discrete = False, Bitrate_set = np.array([1e5,1e7]), Frame_rate = np.ones((1,10))*60, Option = "GOP"):
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

        self.Bitrate = np.ones((1,self.User_number))*1e7
        self.Bitrate_per_frame = self.Bitrate/self.Frame_rate
        self.Frame_type = np.random.randint(0,self.K,(1,self.User_number))




    def generateFrame(self,):
        
        if self.Option == "GOP":
            for i in range(self.User_number):
                if self.Frame_type[0,i] == 0:
                    #I—frame
                    mu = self.alpha/(self.K-1+self.alpha)*self.K/self.Frame_rate[0,i]
                else:
                    #P—frame                    
                    mu = (self.K-1)/(self.K-1+self.alpha)*self.K/(self.Frame_rate[0,i]*(self.K-1))
                sigma = 0.105/self.Frame_rate[0,i]*self.alpha/(self.K-1+self.alpha)
                lower = 0.5*mu
                upper = 1.5*mu
                bitrate = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma).rvs(1)
                self.Bitrate_per_frame[0,i] = bitrate


                self.Frame_type[0,i]=(self.Frame_type[0,i]+1)%self.K

        

    def GetBitratePerFrame(self,):
        self.generateFrame()
        # print(1e6/60,self.Bitrate_per_frame*self.Bitrate)
        # print(np.mean(self.Bitrate_per_frame*self.Bitrate))
        Bitrate_per_frame =  self.Bitrate_per_frame*self.Bitrate
        return Bitrate_per_frame

    def SetBitrate(self,Bitrate):
        self.Bitrate = Bitrate

    def reset(self,):
        self.Frame_type = np.random.randint(0,self.K,(1,self.User_number))
      




# video = Video()
# ratesum=0
# for ii in range(100):
#     ratesum+=np.sum(video.GetBitratePerFrame())

# print(ratesum/100/4*60)
