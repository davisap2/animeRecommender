# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

np.seterr(divide='ignore', invalid='ignore')

#Calculates the dot poduct subtracted by row mean
def matmult (a,amean,b,bmean):
    total = 0
    for i,j in zip(a,b):
        if (j == 0 or i==0 or np.isnan(j) or np.isnan(i) ):
            continue
        total += (i - amean)  * (j - bmean)
    return total


#Calculates the cosine simularity
def sim (a,amean,b,bmean):

    top = matmult(a,amean,b,bmean)
    bottoma = matmult(a,amean,a,amean)
    bottomb = matmult(b,bmean,b,bmean)
    if (top==0 or bottoma ==0 or bottomb==0):
        value = 0
    else:
        value = top/np.sqrt(bottoma*bottomb)
        
    return value

#The shows to evaluate estimated values
evalshow_list = ['Death Note','FLCL','Naruto','Highschool of the Dead','Vampire Knight']
#The users to evaluate
evaluser_list = [51,196,256,365,591,607,672,700,751,955]

#load evaluation data
ratings_list  = pd.read_csv(r'rating.csv')
show_list = pd.read_csv(r'anime.csv')

#Create user list
userlist = ratings_list['user_id'].drop_duplicates().head(10000).tolist()

#Merge tables and sort movie list by volume of users that have rated it
top_list = show_list[['name','anime_id','members']].sort_values('members',ascending= False).drop('members',axis=1)
top_list = pd.merge(top_list,ratings_list[ratings_list.user_id.isin(userlist)], on='anime_id')
top_list = top_list.pivot_table(index = 'name', columns = 'user_id', values = 'rating')

#calculate the row mean
top_list = top_list.replace(-1,np.nan)
top_list = top_list.assign(rmean=lambda x: x.mean(axis=1))
top_list = top_list.assign(simval=0)

output = []
for evalshow in evalshow_list:
    for evaluser in evaluser_list:
        #Store the currently rated value of the show and remove it from the matrix
        if (not np.isnan(top_list.at[evalshow,evaluser])):
            original_rate = top_list.at[evalshow,evaluser]
            top_list.at[evalshow,evaluser] = np.nan
        else:
            continue

        
        #Calculate the cosine simularity for each row
        a = top_list.loc[evalshow].drop(['rmean','simval'])
        amean = top_list.at[evalshow,'rmean']
        
        #Filter un ranked shows
        rowlist = top_list[top_list[evaluser]>0][evaluser].index.tolist()
        
        for row in rowlist:
            bmean = top_list.at[row,'rmean']
            if (bmean == 0):
                val = 0
            else:
                val = sim(a,amean,top_list.loc[row].drop(['rmean','simval']),bmean)
            top_list.loc[row,'simval'] = val
        
        
        pred_list = top_list[top_list[evaluser] != 0].sort_values('simval',ascending=False).head(5)
        
        n = 0
        s = 0
        
        for sx,nx in zip(pred_list[evaluser].tolist(),pred_list['simval'].tolist() ):
            if (not np.isnan(sx) and not np.isnan(nx)):
                s += sx * nx
                n += nx 
        if (n==0 or s==0):
            r = 0
        else:
            r = s/n
        
       
        output.append([evaluser,evalshow,r,original_rate])
        top_list.at[evalshow,evaluser] = original_rate
       
 
output = pd.DataFrame(output,columns=['User','Title','Estimated Rating','Actual Rating']) 
rms = sqrt(mean_squared_error(output['Actual Rating'],output['Estimated Rating']))
print(output.sort_values(['User','Title']))
print ('The root mean squeard error (RMSE) is ' + str(rms))
output.to_csv('Estimated_Ratings_Comparison.csv')