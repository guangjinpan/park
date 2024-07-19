import cvxpy as cp
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