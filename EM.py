# -*- coding: utf-8 -*-

import numpy as np
import math  
import copy  
import matplotlib.pyplot as plt  

isdebug = True

# 参考文献：机器学习TomM.Mitchell P.137
# 代码参考http://blog.csdn.net/chasdmeng/article/details/38709063

# 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差Sigma，均值分别为Mu1,Mu2。  
def init_data(Sigma,Mu1,Mu2,k,N):  
    global X  
    global Mu  
    global Expectations  
    X = np.zeros((1,N))  
    Mu = np.random.random(k)  
    Expectations = np.zeros((N,k))  
    for i in xrange(0,N):  
        if np.random.random(1) > 0.5:  
            X[0,i] = np.random.normal(Mu1, Sigma)
        else:  
            X[0,i] = np.random.normal(Mu2, Sigma)
    if isdebug:  
        print("***********")
        print("初始观测数据X：")
        print(X )
        
# EM算法：步骤1，计算E[zij]  
def e_step(Sigma, k, N):  
    global Expectations  
    global Mu  
    global X  
    for i in xrange(0,N):  
        Denom = 0 
        Numer = [0.0] * k
        for j in xrange(0,k):  
            Numer[j] = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)  
            Denom += Numer[j]
        for j in xrange(0,k):  
            Expectations[i,j] = Numer[j] / Denom  
    if isdebug:  
        print("***********")
        print("隐藏变量E（Z）：")
        print(Expectations)
        
# EM算法：步骤2，求最大化E[zij]的参数Mu  
def m_step(k,N):  
    global Expectations  
    global X  
    for j in xrange(0,k):  
        Numer = 0  
        Denom = 0  
        for i in xrange(0,N):  
            Numer += Expectations[i,j]*X[0,i]  
            Denom +=Expectations[i,j]  
        Mu[j] = Numer / Denom
        
# 算法迭代iter_num次，或达到精度Epsilon停止迭代  
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):  
    init_data(Sigma,Mu1,Mu2,k,N)  
    print("初始<u1,u2>:", Mu)
    for i in range(iter_num):  
        Old_Mu = copy.deepcopy(Mu)  
        e_step(Sigma,k,N)  
        m_step(k,N)  
        print(i,Mu)
        if sum(abs(Mu - Old_Mu)) < Epsilon:  
            break  

if __name__ == '__main__':
    sigma = 6   # 高斯分布具有相同的方差
    mu1 = 40    # 第一个高斯分布的均值 用于产生样本
    mu2 = 20    # 第二个高斯分布的均值 用于产生样本
    k = 2       # 高斯分布的个数
    N = 1000    # 样本个数
    iter_num = 1000 # 最大迭代次数
    epsilon = 0.0001    # 当两次误差小于这个时退出
    run(sigma,mu1,mu2,k,N,iter_num,epsilon)  
   
    plt.hist(X[0,:],50)  
    plt.show()
