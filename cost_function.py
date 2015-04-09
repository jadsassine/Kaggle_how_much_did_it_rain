# -*- coding: utf-8 -*-
"""
Created on Fri Apr 09 2015

@author: Jad Sassine
"""


import numpy as np

# to predict the correct output, we want to apply the sigmoid function to:
#
# [[(1-x11)b1+(1-x12)b2+(1-x13)b3+(x14)b4+...+(x1m)bm   ...   (70-x11)b1+(70-x12)b2+(70-x13)b3+(x14)b4+...+(x1m)bm
#   (1-x21)b1+(1-x22)b2+(1-x23)b3+(x24)b4+...+(x2m)bm   ...   (70-x21)b1+(70-x22)b2+(70-x23)b3+(x24)b4+...+(x2m)bm
#          .
#          .
#   (1-xN1)b1+(1-xN2)b2+(1-xN3)b3+(xN4)b4+...+(xNm)bm   ...   (70-xN1)b1+(70-xN2)b2+(70-xN3)b3+(xN4)b4+...+(xNm)bm

# this is why I defined: 

#feature = [np.arange(70)-x_train[:,0],np.arange(70)-x_train[:,1],
#np.arange(70)-x_train[:,3], x_train[:,4],x_train[:,5],x_train[:,6]]

# so that I can get the correct matrix by calling:

#pred = feature[0]*theta[0]
#for i in range(1,len(feature)):
#    pred += feature[i]*theta[i]




def sigmoid(theta, feature):
    pred = feature[0]*theta[0]
    for i in range(1,len(feature)):
        pred += feature[i]*theta[i]
    return 1./(1. + np.exp(-pred))



def cost(theta,feature,output):
    pred = sigmoid(theta, feature)
    return np.sum(np.square(pred-output))/(70*len(output))



def der_cost(theta,feature,output):
    pred = sigmoid(theta, feature)        
    der = []
    m_1 = 2*(pred-output)
    m_2 = np.multiply(m_1, np.multiply(pred,(1-pred)))
    for f in feature:       
        m = np.multiply(m_2, f)
        der.append(np.sum(m)/(70*len(output)))
    return np.array(der)


def verify_gradient(theta, feature, output, e=0.0001):
    eps = np.array([0.]*len(theta))
    est = []
    for i in range(len(theta)):
        eps[i] = e
        est.append((cost(theta+eps,feature,output)-
                        cost(theta-eps,feature,output))/(2*e))
        eps[i]= 0
    return np.sqrt(np.sum(np.square(est-der_cost(theta,feature,output))))


def optimize(init,feature,output,alpha,epsilon,maxiter):
    theta = init
    costs = []
    i=0
    while i < maxiter :
        costs.append(cost(theta,feature,output))
        der = der_cost(theta,feature,output)
        print costs[len(costs)-1], np.sum(np.square(der))
        if np.sum(np.square(der)) < epsilon:
            print 'reached minimum'
            print 'best parameters are: ', theta
            print 'cost is: ', costs[len(costs)-1]
            return theta, costs
        else:
            theta -= alpha*der
        i += 1
    print 'reached max number of iteration'
    print 'best parameters so far are :', theta
    return theta, costs









