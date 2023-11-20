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


class Channel():
    def __init__(self, User_number = 4, channel_number = 10, bandwidth_total = 10e6, Radius = 1000, rmin = 30):
        self.User_number = User_number
        self.channel_number = channel_number        
        self.bandwidth_total = bandwidth_total
        self.Radius = Radius
        # Min distance between user and BS
        self.rmin = rmin
        self.Bw = np.ones((self.User_number,1))*self.bandwidth_total/self.channel_number
        self.BUdistance = self.rmin + math.sqrt(self.Radius**2-self.rmin**2)*np.sqrt(np.random.rand(self.User_number))

    def generateGains(self,):
        
        # Generate the position of self.User_number users in 1 cell
        # Uniform in a circle without r<self.rmin   
    #    print(BUdistance)

        # Compute the link gain including Rayleigh fading, path loss and shadowing
        rayleigh = np.random.randn(self.User_number,self.channel_number) + 1j*np.random.randn(self.User_number,self.channel_number)   # fast fading
        path_loss = -(128.1+37.6*np.log10(self.BUdistance/1000))   # path loss model : BUdistance/1000 in km
        path_loss = np.power(10,(path_loss/10))               # dB to scalar
        
        shadowing = -10*np.random.randn(self.User_number,self.channel_number)          # lognormal distributed with SD 10
        shadowing = np.power(10,(shadowing/10))       # dB to scalar
        
        BUlinkgain = np.array([[ path_loss[k] * np.power(np.absolute(rayleigh[k][l]),2) * shadowing[k][l] for l in range(self.channel_number)] for k in range(self.User_number)])
        
        '''
        print('BUdistance',BUdistance)   
        print('rayleigh',rayleigh) 
        print('path_loss',path_loss) 
        print('shadowing',shadowing) 
        print('BUlinkgain',BUlinkgain) 
        '''
        
        return BUlinkgain

    def GetChannelGain(self,):
        self.channel_gain=self.generateGains(self.User_number, self.channel_number, self.Radius,self.rmin)
        return self.channel_gain

    def reset(self,):
        self.BUdistance = self.rmin + math.sqrt(self.Radius**2-self.rmin**2)*np.sqrt(np.random.rand(self.User_number))