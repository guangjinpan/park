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
    def __init__(self, BS_number=4, User_number = 10, bandwidth_total = 10e6, Radius = 1000, rmin = 30):
        self.BS_number=BS_number
        self.User_number = User_number        
        self.bandwidth_total = bandwidth_total
        self.Radius = Radius
        # Min distance between user and BS
        self.rmin = rmin


        self.BS_position=np.array([[0,0],[self.Radius,0],[0,self.Radius],[self.Radius,self.Radius]])
        self.User_position=np.random.rand(self.User_number,2)*self.Radius
        diff = self.BS_position[:, np.newaxis, :] - self.User_position  # 计算坐标差
        dist = np.linalg.norm(diff, axis=2)  # 计算欧氏距离
        self.BUdistance = dist
        self.BUdistance[self.BUdistance < self.rmin] = self.rmin
     
        
        self.User_sin_theta=np.zeros((self.BS_number,self.User_number))
        self.User_cos_theta=np.zeros((self.BS_number,self.User_number))
        for i in range(self.User_number):
            for j in range(self.BS_number):
                x=self.User_position[i,:]-self.BS_position[j,:]
                xx=np.sqrt(x.dot(x))
                self.User_sin_theta[j,i]=x[1]/xx
                self.User_cos_theta[j,i]=x[0]/xx
        
        #生成初始信道和信噪比        
        Noise_Hz = 10**((-174)/10) # Noise value per Hertz = -174 dBm/Hz
        self.noise = np.ones((self.BS_number,self.User_number))*Noise_Hz  
        self.channel_gain=self.generateGains()
        self.SNR=self.generateSNR()
        
        # print("self.SNR",np.max(self.SNR,0))
        # print(self.BUdistance)
    def generateGains(self,):
        
        # Generate the position of self.User_number users in 1 cell
        # Uniform in a circle without r<self.rmin   
    #    print(BUdistance)

        # Compute the link gain including Rayleigh fading, path loss and shadowing
        rayleigh = np.random.randn(self.BS_number,self.User_number) + 1j*np.random.randn(self.BS_number,self.User_number)   # fast fading
        path_loss = -(128.1+37.6*np.log10(self.BUdistance/1000))   # path loss model : BUdistance/1000 in km
        path_loss = np.power(10,(path_loss/10))               # dB to scalar
        
        shadowing = -10*np.random.randn(self.BS_number,self.User_number)          # lognormal distributed with SD 10
        shadowing = np.power(10,(shadowing/10))       # dB to scalar
        BUlinkgain = np.array([[ path_loss[k][l] * np.power(np.absolute(rayleigh[k][l]),2) * shadowing[k][l] for l in range(self.User_number)] for k in range(self.BS_number)])
        
        
        # print('BUdistance',self.BUdistance)   
        # print('rayleigh',rayleigh) 
        # print('path_loss',path_loss) 
        # print('shadowing',shadowing) 
        # print('BUlinkgain',BUlinkgain) 
        
        
        return BUlinkgain

    def generateSNR(self,):
        SNR=np.zeros((self.BS_number,self.User_number))
        for i in range(self.BS_number):
            for j in range(self.User_number):
                interference=np.sum(self.channel_gain[:,j])-self.channel_gain[i,j]
                SNR[i,j]=self.channel_gain[i,j]/(interference+self.noise[i,j])
        return SNR


    def GetChannelGain(self,):
        self.channel_gain=self.generateGains()
        self.SNR=self.generateSNR()
        return self.channel_gain, self.SNR, self.User_sin_theta, self.User_cos_theta
    
    

    def reset(self,):
        self.BS_position=np.array([[0,0],[0,self.Radius],[self.Radius,0],[self.Radius,self.Radius]])
        # self.User_position=np.random.rand(self.User_number,2)*self.Radius
        diff = self.BS_position[:, np.newaxis, :] - self.User_position  # 计算坐标差
        dist = np.linalg.norm(diff, axis=2)  # 计算欧氏距离
        self.BUdistance = dist
        self.BUdistance[self.BUdistance < self.rmin] = self.rmin
        
        # print(self.User_position)