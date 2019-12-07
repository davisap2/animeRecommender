# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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


def check_genre(genre_list,string):
    if any(x in string for x in genre_list):
        return True
    else:
        return False

#The users to evaluate
evaluser_list = [51,196,256,3657,5915,6076,6727,7004,7511,9558]

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

#Identify possible shows
ratings = pd.DataFrame(top_list['rmean'])
ratings['num of ratings'] = pd.DataFrame( top_list.drop('rmean',axis = 1).count(axis='columns'))
ratings = ratings[ratings['num of ratings']>ratings['num of ratings'].mean()]
ratings = ratings.sort_values(['rmean','num of ratings'],ascending=(False,False)).head(20)

genre_dict = pd.DataFrame(show_list[['name','genre']])
genre_dict.set_index('name',inplace=True)
filtered_dict = pd.DataFrame(show_list[show_list.name.isin(ratings.index.tolist())][['name','genre']])
filtered_dict = filtered_dict.reset_index()

output = pd.DataFrame()

for evaluser in evaluser_list:
    #Get Top 5 rated shows for user
    user_top_rated = top_list[evaluser].where(lambda x: x != 0 ).sort_values(ascending = False).head(5)
    user_rated = user_top_rated.index.tolist()
    user_top_rated = user_top_rated.index.tolist()
    
    reclist = list()
    for item in user_top_rated:
        glist = genre_dict.loc[item].values[0].split(', ')
        reclist.extend(filtered_dict[filtered_dict['genre'].apply(lambda x: check_genre(glist,str(x)))]['name'].tolist())
        
    reclist = set(reclist)
    reclist = list(reclist)
    
    #Check for already reviewed items
    for u in user_top_rated:
        if u in reclist:
            reclist.remove(u)
            
    print('\n User number ' + str(evaluser) + ' highest rated shows:')
    print (user_top_rated)
    
    final_list = pd.DataFrame(columns=['User','Name','Estimated Rating'])
    
    for evalshow in reclist:
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
        
        #Using Nearest neighbor value for head selection
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
            
        final_list = final_list.append({'User':evaluser,'Name': evalshow,'Estimated Rating':r}, ignore_index = True)
        
    output = output.append(final_list.sort_values('Estimated Rating',ascending=False).head(5), ignore_index=True)

print('\nShow Reccomendations')
print(output)
output.to_csv('User_Reccomendations.csv')

