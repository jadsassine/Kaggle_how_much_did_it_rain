# -*- coding: utf-8 -*-
"""
Created on Fri Apr 09 2015

@author: Jad Sassine
"""

import csv
import copy
import numpy as np

def find_cut(time, missing):  
    "function used to separate each radar using TimeToEnd"  
    cut = []
    i=0; j=1
    while time[i] in missing: 
        i += 1
        j += 1
    while j < len(time):
        if time[j] not in missing:
            if float(time[j-1]) <= float(time[j]):
                cut.append((i,j))
                i = j          
        j+=1       
    cut.append((i,j))
    return cut   
    


def process_cut(x, qual, missing):
    """
    function used to select out missing data and weigh according to quality
    for the weight: 
    if all values are missing or all qualities have only 0 quality, we can't use it
    else if qualities are either 0 or missing, we use the missing with equal weight
    else if only some weights are missing, we set them equal to the minimum weight  
    """
    value, weight_qual = [], []
    for i in range(len(x)):
        if x[i] not in missing: 
            value = np.append(value, float(x[i]))              
            if qual[i] in missing: weight_qual = np.append(weight_qual, 100)
            else: weight_qual = np.append(weight_qual, float(qual[i]))
    
    if (len(weight_qual) == 0) | (np.sum(weight_qual) == 0): return 100  
    elif (np.sum(weight_qual)%100) == 0: weight_qual[weight_qual == 100] = 1.
    else: weight_qual[weight_qual == 100] = np.min(weight_qual[weight_qual != 100])/2.
    return np.dot(value, weight_qual)/np.sum(weight_qual)


def heavyside(value):
    """heavyside step function which will be used to compute the cost"""
    o=np.array([1]*70)
    o[:value+1] = 0
    return o


def prepare_data(data, missing_data, hydro_chg, train=True, stop=float('inf'), 
                 verify = False):
                     
    """
    this function is used to process the data
    
    each x can have observations/features of different radars
    so we separate each x into different cuts where each cut represents the 
    average observation/features of a radar
    then we take the average of all radars according to a certain weight
  
    after going though the file once, we will replace all the missing values
    by the average of those that are not missing
    """
    
    x, output, id_test = [], [], []
    
    with open(data) as o:
        reader = csv.reader(o)
        header = next(reader)
        
        feat_names = ['DistanceToRadar','RR1','RR2','RR3','HydrometeorType', 
        'MassWeightedMean', 'MassWeightedSD', 'LogWaterVolume']
        
        count = 0
        for row in reader:
            time = row[header.index('TimeToEnd')].split(' ')
            if verify: 
                print 'count', count
                print 'time', time
            cut = find_cut(time, missing_data)
            
            quality = row[header.index('RadarQualityIndex')].split(' ')
            if verify: print 'quality', quality
                                
            weight = np.array([])
            final_feat = []
            for name in feat_names:
                curr_feat = row[header.index(name)].split(' ')
                if verify: print name, curr_feat
                
                #we need to make changes to 'HydrometeorType' as there are doubles
                if name == 'HydrometeorType': 
                    curr_feat = [hydro_chg[i] for i in curr_feat]
                               
                feat_ave = np.array([])
                for c in cut:
                    #each rain rate estimate RR1, RR2 and RR3 is in mm/hour
                    #we compute the period assuming started 3 min before and 
                    #ended 3 min after - add 6 to the total and divide by 60 to get hour
                    p = time[c[0]:c[1]]
                                      
                    if name in ['RR1', 'RR2', 'RR3']: 
                        period = (float(p[0])-float(p[len(p)-1]) + 6.)/60 
                    else:
                        period = 1

                    feat_cut_ave = process_cut(curr_feat[c[0]:c[1]], 
                                               quality[c[0]:c[1]],missing_data)
                    if feat_cut_ave == 100: period = 1
                    feat_ave = np.append(feat_ave, feat_cut_ave*period)                                                                                                
                               
                if verify: print 'feat_ave', feat_ave
                
                valid = feat_ave != 100   
                #we use 'DistanceToRadar' as a weight
                if name == 'DistanceToRadar':
                    weight = copy.copy(feat_ave)
                    if len(weight[valid]) == 0: weight[weight == 100] = 1.
                    #when the distance is only 0's and 100's, give 2x more weight
                    #to a 0 distance than an unknown distance 
                    elif np.sum(weight)%100 == 0: 
                        weight[weight == 0] = 1.
                        weight[weight == 100] = .5
                    #else we replace the missing by the average of the non-missing
                    #we weigh using the inverse of the distance and give the 
                    #distance 0 2x more weight than the maximum
                    else: 
                        weight[weight == 100] = np.mean(feat_ave[valid])
                        weight[weight > 0] = 1./weight[weight > 0]
                        weight[weight == 0] = 2.*np.max(weight)
                    if verify: print 'weight', weight    
                else:
                    if len(feat_ave[valid]) == 0: final_feat.append(100)
                    elif len(feat_ave[valid]) == 1: 
                        final_feat.append(feat_ave[valid][0])                       
                    
                    else: final_feat.append(np.dot(feat_ave[valid], 
                                        weight[valid])/np.sum(weight[valid]))
            
            if verify: print 'final_feat', final_feat
            
            if str(np.mean(final_feat)) == 'nan':
                print 'count', count
                print 'time', time
                print 'quality', quality
                print name, curr_feat
                print 'weight', weight
                print 'feat_ave', feat_ave
                break

            x.append(final_feat)   
                 
            if train: output.append(heavyside(int(row[len(row)-1][0])))                
            else: id_test.append(int(row[0]))
            
            count += 1
            if count == stop: break
            if count%100000 == 0: print 'first ' + str(count) + ' rows done' 
            
    if train: return np.matrix(x), np.matrix(output)
    else: return np.matrix(x), id_test


