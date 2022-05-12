#Imports 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import itertools
import seaborn as sn
from sklearn.decomposition import PCA;
from datetime import datetime
from numpy.linalg import norm

#Preprocessing Code:

#Extract Data
spx_2018 = pd.read_csv('Batch_3TWA5YR9C2/SPX_2018.csv')
spx_2019 = pd.read_csv('Batch_3TWA5YR9C2/SPX_2019.csv')
spx_2020 = pd.read_csv('Batch_3TWA5YR9C2/SPX_2020.csv')

#Concatenate the dataframes
spx_total = pd.concat([spx_2018,spx_2019,spx_2020], ignore_index = True)

#Extract relevant fields
spx_total = spx_total[['underlying_last',
                       'type',
                       'expiration',
                       'quotedate',
                       'strike',
                       'impliedvol',
                       'delta']]

#Calculate Moneyness
spx_total['moneyness'] = spx_total['strike']/spx_total['underlying_last']

#Extract Out of the money Calls and Puts
spx_otm_calls = spx_total[spx_total['type'] == 'call']
spx_otm_calls = spx_otm_calls[spx_otm_calls['moneyness'] > 1]
spx_otm_puts = spx_total[spx_total['type'] == 'put']
spx_otm_puts = spx_otm_puts[spx_otm_puts['moneyness'] < 1]

relevant_options = pd.concat([spx_otm_calls,spx_otm_puts], ignore_index = True)

#Function to calculate time to expiry in years
def add_time_to_exp(df):
    df["time_to_expiry_yrs"] = 0.0
    for i in range(len(df.index)):
        if (i%1000) == 0:
            print(i/len(df.index))
        df['expiration'][i] = pd.Timestamp(df['expiration'][i])
        df['quotedate'][i] = pd.Timestamp(df['quotedate'][i])
        df['time_to_expiry_yrs'][i] = int(str(df['expiration'][i] \
                                             - df['quotedate'][i]).split()[0])/252
    
    return df

#Run this to calculate all the expiry times, takes awhile
total_chain = add_time_to_exp(relevant_options)

def get_surface(day):
    #Returns the surface first column is moneyness, then time to expiry, then impvol
    dummydf = day.copy()
    surface = pd.DataFrame(columns=['moneyness','time_to_expiry_yrs','impliedvol','quotedate'])
    while len(dummydf.index) > 0:
        one_exp = dummydf[dummydf['expiration'] == dummydf['expiration'][0]]
        dummydf = dummydf[dummydf['expiration'] != dummydf['expiration'][0]].reset_index()
        dummydf = dummydf[['moneyness','time_to_expiry_yrs','impliedvol','expiration','quotedate']]
        smile_slice = one_exp[['moneyness','time_to_expiry_yrs','impliedvol','quotedate']]
        surface = pd.concat([surface, smile_slice], ignore_index = True)
    return surface.to_numpy()

#Get surfaces for all days, input full chain here
def get_all_surfaces(options):
    dummydf = options.copy()
    surfaces = []
    while len(dummydf.index) > 0:
        day = dummydf[dummydf['quotedate'] == dummydf['quotedate'][0]]
        dummydf = dummydf[dummydf['quotedate'] != dummydf['quotedate'][0]].reset_index()
        dummydf = dummydf[['moneyness','time_to_expiry_yrs','impliedvol','expiration','quotedate']]
        surfaces.append(get_surface(day))
    return surfaces

#Smoothing Surface Code 
def smooth_surface(points, h1,h2):
    
    def apply_estimator(moneyness, exp, points, h1, h2):
        up_sum = 0
        down_sum = 0
        for p in points:
            gk = gaussian_kernel(moneyness - p[0],exp-p[1],h1,h2)
            up_sum += gk*p[2]
            down_sum += gk
        return up_sum/down_sum
            
    def gaussian_kernel(x,y, h1, h2):
        return (1/2*math.pi)*math.exp((-x**2)/(2*h1))*math.exp((-y**2)/(2*h2))
    
    moneyness = np.linspace(0.6,1.4,50)
    expiry = np.linspace(0,2,50)
    date = points[0][3]
    surface = []
    
    for i in range(len(moneyness)):
        for j in range(len(expiry)):
            smoothed_point = apply_estimator(moneyness[i],expiry[j],points, h1, h2)
            surface.append([moneyness[i],expiry[j],smoothed_point])
    return [np.array(surface),date]
        
def smooth_all(all_surfaces):
    smoothed = []
    count = 0
    for surface in all_surfaces:
        if count%1 == 0:
            print(count/len(all_surfaces))
        count += 1
        smoothed.append(smooth_surface(surface,0.01,0.05))
    return smoothed


#Space of Variations
def get_delta_surfaces(surfaces):
    del_X = []
    moneyness = surfaces[0][0][:,0]
    exp = surfaces[0][0][:,1]
    for i in range(1,len(surfaces)):
        del_imp = np.log(surfaces[i][0][:,2]) - np.log(surfaces[i-1][0][:,2])
        del_surface = np.array([moneyness,exp,del_imp]).T
        del_X.append([del_surface,surfaces[i][1]])
    return del_X

#Util to calculate the mean surface of a group of surfaces 
def calculate_mean_surface(surfaces):
    example = surfaces[0][0]
    mon = example[:,0]
    exp = example[:,1]
    means = []
    for i in range(len(example)):
        dels = []
        for d_surf in surfaces:
            dels.append(d_surf[0][i][2])
        means.append(np.mean(np.array(dels)))
    
    mean_surf = np.array([mon,exp,means]).T
    return mean_surf

#Compute PCA over surfaces
def pca_over_surfaces(dsurfs, num_components):
    
    moneyness = dsurfs[0][0][:,0]
    exp = dsurfs[0][0][:,1]
    X = []
    for surf in dsurfs:
        vec = surf[0][:,2]
        X.append(vec)
    X = np.array(X)
    
    meanPoint = X.mean(axis = 0)
    X -= meanPoint
    
    pca = PCA(n_components=num_components)
    pca.fit(X)
    
    pc_surfaces = []
    for component in pca.components_:
        pc_surfaces.append(np.array([moneyness, exp, component]).T)
    return pc_surfaces, pca.explained_variance_ratio_
    




