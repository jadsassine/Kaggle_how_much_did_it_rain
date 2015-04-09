# -*- coding: utf-8 -*-
"""
Created on Fri Apr 09 2015

@author: Jad Sassine
"""


import numpy as np
import copy
import csv
from time import time
from scipy.optimize import minimize
import os
os.chdir('C:/Users/Pc-stock2/Desktop/rain') 

from cost_function import *
from process_csv2 import *

missing_value = 100

missing_data = ['-99900.0','-99901.0','-99903.0','999.0','nan']

hydro_chg = {'0.0': '0', '1.0': '1', '2.0': '1', '3.0': '2', '4.0': '3', '5.0': '4', 
'6.0': '5', '7.0': '6', '8.0': '0', '9.0': '0', '10.0': '7', '11.0': '8', '12.0': '9', 
'13.0': '10', '14.0': '10', '-99900.0': str(missing_value) ,'-99901.0': str(missing_value),
'-99903.0': str(missing_value), '999.0': str(missing_value) ,'nan':  str(missing_value)}

#prepare the training data
t0=time()
x_ref, outcome = prepare_data('train_2013.csv', missing_data, hydro_chg)
t1=time()
t1-t0

#replace all the missing values by the average of those that are not missing
x_train = copy.copy(x_ref)
for i in range(np.shape(x_ref)[1]):
    x_train[:,i][x_train[:,i] == 100]=np.mean(x_ref[:,i][x_ref[:,i] != 100])


#optimize
#we define feature as (#see cost_function.py for explanation):
feature = [np.arange(70)-x_train[:,0],np.arange(70)-x_train[:,1],
           np.arange(70)-x_train[:,2], x_train[:,3], x_train[:,4],
           x_train[:,5],x_train[:,6]]

np.random.seed(seed=0)

#we can either minimize using the optimize function I created
#theta0=np.random.random_sample((7,))
#best_param, costs = optimize(init=theta0,feature=feature,output=outcome,
#                             alpha=0.001,epsilon=0.01,maxiter=10)


#or we can use the built in (more advanced) L-BFGS-B method
#for the submission we use the built in method
best_param = []
min_cost = 0
for i in range(3):
    theta0=np.random.random_sample((7,))
    par = minimize(cost, theta0, method='L-BFGS-B', jac=der_cost, args =(feature,outcome))
    c = cost(par.x, feature, outcome)
    print 'last computed cost is ' + str(c)
    if (i == 0) | (c < min_cost): 
        min_cost = c
        best_param = par.x
        print 'curr min cost is ' + str(min_cost)



#load the test data
x_test, ind = prepare_data('test_2014.csv', missing_data, hydro_chg, train = False)
#correct using the same values as in the training set
for i in range(np.shape(x_ref)[1]):
    x_test[:,i][x_test[:,i] == 100]=np.mean(x_ref[:,i][x_ref[:,i] != 100])


#build the features and make the prediction
feature_test = [np.arange(70)-x_test[:,0],np.arange(70)-x_test[:,1],
                np.arange(70)-x_test[:,2], x_test[:,3], x_test[:,4],
                x_test[:,5],x_test[:,6]]

prediction = np.array(sigmoid(best_param,feature_test))


#write solution in a CSV file in the correct submission format
with open('test_output.csv', 'wb') as o:
    w=csv.writer(o)
    solution_header = ['Id']
    solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, 70)])  
    w.writerow(solution_header)
    for i in range(len(ind)):       
        new_p = ['{:.12f}'.format(x) for x in prediction[i]]
        w.writerow(np.append(ind[i],new_p))




