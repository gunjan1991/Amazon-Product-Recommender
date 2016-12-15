
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn import *
import gzip
import math
import warnings 
warnings.filterwarnings('ignore')

#function to find rated and unrated item by a user
def getUnrated(nonan_test,u):
    rated = np.where(nonan_test[u,:] != 0)[0]
    
	#unrateditems = np.nonzero(nonan_test[u,:])[1]
    unrateditems = np.where(nonan_test[u,:] == 0)[0]
    return list(unrateditems), list(rated)

#function to find similarity between two users
def find_sim(nonan_test,u1,u2) :  
   
   cosine_sim = metrics.pairwise.cosine_similarity(nonan_test[u1], nonan_test[u2])# returns only a number
   return cosine_sim

#function used to find n similar users
def getUser(dat, user, n, similarity=find_sim):
   simUsers = []
   scores = []
   #print(user)
   for other in range(len(dat)):
       #print(other)
       if other != user:
           scores.append([similarity(dat, user, other), other])
# sort the list
   scores.sort()
   scores.reverse()
   #print(scores)
   for item in scores[:n]:
       simUsers.append(item[1])
   return simUsers	
   
#function to find similarity between two items
def getItem(dat, item, similarity = find_sim):
   simItems = []
   scores = []
   #print(user)
   for other in range(len(dat.T)):
       #print(other)
       if other != item:
           scores.append([similarity(dat.T, item, other), other])
# sort the list
   scores.sort()
   scores.reverse()
   
   return scores

#function to find k similar users based on precalculated user similarity matrix
def getUserPC(user_sim_mat,u,k):
    score=[]
    simUsers = []
    for others in range(len(user_sim_mat)):
        if others != u:
            score.append([user_sim_mat[u][others], others])
    score.sort()
    score.reverse()
    for item in score[:k]:
        simUsers.append(item[1])
    return simUsers
#return score[:k]

#function to serve user based recommendations
def userBasedRecs(nonan_test, simUsers, unrated, numOfRecs, rated, unique_products, metadata):
    all_simusers_productsReviews = {}
    user_based_recs = []
    final_recs = []
    for user in simUsers:
        simUsers_productreviews = {}
        rated_items = np.where(nonan_test[user] != 0)[0]
        rated_item_reviews = np.extract(nonan_test[user] != 0, nonan_test[user])
        for i in list(zip(rated_items, rated_item_reviews)):
            for item in unrated:
                if item == i[0]:
                    simUsers_productreviews[i[0]] = i[1]
        #for item in unrated:
        #   if item not in simUsers_productreviews: del simUsers_productreviews[item]
        #print(simUsers_productreviews)
        for key in simUsers_productreviews:
            #If this product is not already in the master sim_users dictionary ...
            #... create a key-value pair for that product in the master list: make sure the value is a list
            if key not in all_simusers_productsReviews:
                all_simusers_productsReviews[key] = [simUsers_productreviews[key]]
            else:
                #If the user's product is already in the master sim_users dictionary
                #Find that product in the master dict and append the user's review to the existing list of reviews for that product
                all_simusers_productsReviews[key].append(simUsers_productreviews[key])
    #print(all_simusers_productsReviews)            
    for key in all_simusers_productsReviews:
        user_based_recs.append([np.mean(all_simusers_productsReviews[key]), key])
    user_based_recs.sort()
    user_based_recs.reverse()
    for item in user_based_recs[:numOfRecs]:
        final_recs.append(item[1])

    #print(user_based_recs)
    #print(final_recs)

    all5 = []
    for item in user_based_recs:
        if item[0] == 5.0:
            all5.append(item[1])

    for item in rated:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("User Rated: " + product)
    print('\n')
    for item in final_recs:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("Recommendation: " + product)
        
    return final_recs
        
#Item-based recommender
#All-in-one: most similar items --> recommended items
def itemSim(nonan_test, u, num_recs, unique_products, metadata, itm_sim_mat, unRate, rate):
    
    
    #unRate, rate =  getUnrated(nonan_test,u)
    user_rev_list = []
    simItems = []
    #find their top rated item
    for prod in rate:
        user_rev = nonan_test[u,prod]
        user_rev_list.append([user_rev,prod])
        user_rev_list.sort()
        user_rev_list.reverse()

    most_liked_prod = user_rev_list[0][1]
    asin_pd = unique_products[most_liked_prod]
    product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
    product = str(product)
    print("Most Liked Product: " + product+ '\n')
    
    score=[]
    simItems = []
    for others in range(len(itm_sim_mat)):
        if others != most_liked_prod:
            score.append([itm_sim_mat[most_liked_prod][others], others])
    score.sort()
    score.reverse()
    #print(score[:20])
    for item in score[:20]:
        if item[1] in rate:
            #print(item[1])
            score = list(filter((item).__ne__, score))
    #print(score)
    
    for item in score[:num_recs]:
        simItems.append(item[1])
    #print(simItems)
    for item in simItems:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("Recommended Product: " + product)
    #return score[:k]
 
#Function that takes the col indices of items rated by a user (aka one of the outputs of getUnrated) ...
# .. and returns a list of unique product categories (our user's 'content profile' for hybrid method) 
def getCate(rated, unique_products, metadata):
    listn = []
    for item in rated:
        #print(item)
        asin_pd = unique_products[item]
        #print(asin_pd)
        cat = metadata[metadata['asin']==asin_pd].categories.item()# 'categories']]
        for i in cat:
            for j in i:
            #print(j)
                listn.append(j)
    unl = list(set(listn))
    return unl
	
	
#Hybrid recommender
def hybridRec(nonan_test, simUsers, unrated, numOfRecs, rated, unique_products, metadata):
    all_simusers_productsReviews = {}
    user_based_recs = []
    final_recs = []
    
    catUser = getCate(rated, unique_products, metadata)
    
    for user in simUsers:
        simUsers_productreviews = {}
        rated_items = np.where(nonan_test[user] != 0)[0]
        rated_item_reviews = np.extract(nonan_test[user] != 0, nonan_test[user])
        for i in list(zip(rated_items, rated_item_reviews)):
            for item in unrated:
                if item == i[0]:
                    simUsers_productreviews[i[0]] = i[1]
        #for item in unrated:
        #   if item not in simUsers_productreviews: del simUsers_productreviews[item]
        #print(simUsers_productreviews)
        for key in simUsers_productreviews:
            #If this product is not already in the master sim_users dictionary ...
            #... create a key-value pair for that product in the master list: make sure the value is a list
            if key not in all_simusers_productsReviews:
                all_simusers_productsReviews[key] = [simUsers_productreviews[key]]
            else:
                #If the user's product is already in the master sim_users dictionary
                #Find that product in the master dict and append the user's review to the existing list of reviews for that product
                all_simusers_productsReviews[key].append(simUsers_productreviews[key])
    #print(all_simusers_productsReviews)            
    for key in all_simusers_productsReviews:
        user_based_recs.append([np.mean(all_simusers_productsReviews[key]), key])
    #print("Original user-based recs:\n", user_based_recs)
    for item in user_based_recs:
        #print([item[1].item()])
        catItem = getCate([item[1].item()], unique_products, metadata)
        match = list(set(catUser).intersection(catItem))
        weight = len(match)/len(catItem)
        item[0] = item[0]*weight
    
    user_based_recs.sort()
    user_based_recs.reverse()
    
    for item in user_based_recs[:numOfRecs]:
        final_recs.append(item[1])
    
    
    #print("With content-based weighting:\n", final_recs)
    
    all5 = []
    for item in user_based_recs:
        if item[0] == 5.0:
            all5.append(item[1])
    
    for item in rated:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("User Rated: " + product)      
    
    print('\n')
        
    for item in final_recs:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("Recommendation: " + product)
		
    return final_recs

#Hybrid based Recall
def hybrid_recall(nonan_test, ratio, numSimUsers, numRecs, unique_products, metadata, user_sim_mat):
   #get a subset of users
   num_users = np.shape(nonan_test)[0]
   list_of_users = list(range(0,num_users))
   np.random.shuffle(list_of_users)
   subset_size = int(num_users * ratio)
   subset_users = list_of_users[0:subset_size]
  
   user = 0
   user_recall = 0
   
   #for each index in that subset_users, pull the user at that row, and do process of getting similar users
   for wuser in subset_users:
       user+=1
       #simUsers = getUser(nonan_test, wuser, numSimUsers, similarity=find_sim)
       simUsers = getUserPC(user_sim_mat,wuser,numSimUsers)
       unrated, rated = getUnrated(nonan_test, wuser)
       wuser_reviews_list = []
       #print("Items that user rated: ", rated)
       for product in rated:
           wuser_review = nonan_test[wuser, product]
           wuser_reviews_list.append([wuser_review, product])
       wuser_reviews_list.sort()
       wuser_reviews_list.reverse()
       
       #We now have the users product reviews and similar users.
       #We want to add validation prods to 'unrated' list, and serve recommendations to user
       validation_prods = wuser_reviews_list[0][1]
       #print("Product pulled: ", validation_prods)
       unrated.append(validation_prods)
       #print(unrated)
       recommendations = hybridRec(nonan_test, simUsers, unrated, numRecs, rated, unique_products, metadata)
       recommendations = map(int,recommendations)
       validation_prods = int(validation_prods)
       count = 0
       for rec in recommendations:
           if rec == validation_prods:
               count+=1
       user_recall += count/1
       #print(count)
       #print(user_recall)
       #print(user)
   
   final_recall = user_recall/user
   print("\n")
   print("Final recall: ", final_recall)


        
def user_based_recall(nonan_test, ratio, numSimUsers, numRecs,  unique_products, metadata, user_sim_mat):
    #get a subset of users
    num_users = np.shape(nonan_test)[0]
    list_of_users = list(range(0,num_users))
    np.random.shuffle(list_of_users)
    subset_size = int(num_users * ratio)
    subset_users = list_of_users[0:subset_size]
   
    user = 0
    user_recall = 0
    
    #for each index in that subset_users, pull the user at that row ...
	# ... and do process of getting similar users
    for wuser in subset_users:
        user+=1
        simUsers = getUserPC(user_sim_mat, wuser, numSimUsers)
        unrated, rated = getUnrated(nonan_test, wuser)
        wuser_reviews_list = []
        for product in rated:
            wuser_review = nonan_test[wuser, product]
            wuser_reviews_list.append([wuser_review, product])
        wuser_reviews_list.sort()
        wuser_reviews_list.reverse()
        
        #We now have the users product reviews and similar users.
        #We want to add validation prods to 'unrated' list, and serve recommendations to user
        validation_prods = wuser_reviews_list[0][1]
        #print(validation_prods)
        unrated.append(validation_prods)
        #print(unrated)
        recommendations = userBasedRecs(nonan_test, simUsers, unrated, numRecs, rated, unique_products, metadata)
        recommendations = map(int,recommendations)
        validation_prods = int(validation_prods)
        count = 0
        for rec in recommendations:
            if rec == validation_prods:
                count+=1
        user_recall += count/1
        #print(count)
        #print(user_recall)
        #print(user)
    
    final_recall = user_recall/user
    print("\n")
    print("Final recall: ", final_recall)
    
def item_based_recall(nonan_test, ratio, numSimUsers, numRecs, unique_products, metadata, itm_sim_mat):
    
    num_users = np.shape(nonan_test)[0]
    list_of_users = list(range(0,num_users))
    np.random.shuffle(list_of_users)
    subset_size = int(num_users * ratio)
    subset_users = list_of_users[0:subset_size]
   
    user = 0
    user_recall = 0
    for wuser in subset_users:
        user+=1
        unRate, rate =  getUnrated(nonan_test,wuser)
        user_rev_list = []

        #find their top rated item
        for prod in rate:
            user_rev = nonan_test[wuser,prod]
            user_rev_list.append([user_rev,prod])
            user_rev_list.sort()
            user_rev_list.reverse()
        #print(user_rev_list)	
        if len(user_rev_list) < 2:
            print("Can't perform recall!")
        else:
            most_liked_prod = user_rev_list[0][1]
            #validation_prod = user_rev_list[0][1]
            unRate.append(most_liked_prod)
            asin_pd = unique_products[most_liked_prod]
            product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
            product = str(product)
            #print("Most Liked Product:" + product)
            
            score=[]
            simItems = []

            for item in rate:
                if item != most_liked_prod:
                    #score.append([find_sim(nonan_test.T, most_liked_prod, item), item])
                    score.append([itm_sim_mat[most_liked_prod][item], item])
            score.sort()
            score.reverse()
            
            rate = []

            for item in score:
                rate.append(item[1])
            #print(rate)
            
            
            simItems = []
            
            prod_to_use = score[0][1]
            
            #print(prod_to_use)
            
            score=[]
            
            for others in range(len(itm_sim_mat)):
                if others != prod_to_use:
                    score.append([itm_sim_mat[prod_to_use][others], others])
            score.sort()
            score.reverse()
            
            #print(score[:20])
            for item in score[:20]:
                if item[1] in rate:
                    score = list(filter((item).__ne__, score))
    #print(score)
    
            for item in score[:numRecs]:
                simItems.append(item[1])
    #print(simItems)
            '''
            for item in simItems:
                asin_pd = unique_products[item]
                product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
                product = str(product)
                print("Recommended Product: " + product)
                '''      
            recommendations = map(int,simItems)
            validation_prods = int(most_liked_prod)
            count = 0
            for rec in recommendations:
                if rec == validation_prods:
                    count+=1
            user_recall += count/1
            #print(count)
            #print(user_recall)
            #print(user)
   
    final_recall = user_recall/user
   
    print("Final recall: ",final_recall)

#MAE for item-based
#We need to:
#Get a user's product reviews
#Pop one of their products and find most similar products to it (that the user has reviewed!)
#Get the average rating that the user has given to the most similar k items

def itemRec_evaluator(user, k, nonan_test, itm_sim_mat):
    
    score = []
    simuser = []
    simu = []
    simItems = []
    users_products = list(np.nonzero(nonan_test[user]))
    the_prod = users_products[0][0]  #arbitrary product from their list of products - popped product
    #print(the_prod)
    
    users_review = nonan_test[user, the_prod]
    #print(users_review)
    
    #find the products rated by user and its similarity to poped product
    unRate, rate =  getUnrated(nonan_test,user)
    #print(rate)
    for item in rate:
        if item != the_prod:
            score.append([itm_sim_mat[the_prod][item], item])
    score.sort()
    score.reverse()
    #print(score, '\n')
    for item in score[:k]:
        simItems.append(item[1])
    
    itmRatings = 0
    
    for item in simItems:
        itmRatings += nonan_test[user][item]
    pred_rating = itmRatings/k
    #print(pred_rating)
    #print(users_review)
    return pred_rating, users_review

def calc_mae_itm(nonan_test, ratio, k, itm_sim_mat):
    
    #Get the subset of users
    num_users = np.shape(nonan_test)[0]
    list_of_users = list(range(0,num_users))
    np.random.shuffle(list_of_users)
    subset_size = int(num_users * ratio)
    subset_users = list_of_users[0:subset_size]
    #print(subset_users)
    abserror = 0
    count = 0
    for wuser in subset_users:
        pred, userr = itemRec_evaluator(wuser, k, nonan_test, itm_sim_mat)
        abserror += abs(pred-userr)
        #print(abserror)
        count += 1
    MAE = abserror/count
    print("MEAN ABSOLUTE ERROR: ", MAE)

#Given a user, find the most similar users who have also reviewed that user's 1st reviewed product ...
#Get the avg rating of that product from the k most similar users, that's our prediction
#Returns avg rating and actual rating ... this function is used for calculating MAE
def collabo_evaluator(user, k, nonan_test, user_sim_mat):
   
    simuser = []
    simu = []
    users_products = list(np.nonzero(nonan_test[user]))
    the_prod = users_products[0][0]  #arbitrary product from their list of products
    #print(the_prod)
    users_review = nonan_test[user, the_prod]
    #print(users_review)
       #find all users who have reviews 'the_prod'
    users_with_prod = np.where(nonan_test[:, the_prod] != 0)[0]
    for i in users_with_prod:
        if i != user:
            sim = user_sim_mat[i][user]
            simuser.append([sim, i])
    simuser.sort()
    simuser.reverse()
    for item in simuser[:k]:
        simu.append(item[1])
    #print(simu)
    rat = 0
    for kuser in simu:
        rat += nonan_test[kuser][the_prod]
    pred_rating = rat/k
    #print(pred_rating)
    return pred_rating, users_review

def collabo_evaluator_hyb(user, k, nonan_test, unique_products, metadata, user_sim_mat):
   
    simuser = []
    simu = []
    rated_prod = np.where(nonan_test[user,:] != 0)[0]
    #print(rated_prod)
    users_products = list(np.nonzero(nonan_test[user])) #get products rated by user
    #print(users_products)
    catUser = getCate(rated_prod[1:], unique_products, metadata)                           #categories of products rated by user expect the product we are predicting for
    
    the_prod = users_products[0][0]                 #arbitrary product from their list of products
    users_review = nonan_test[user, the_prod]       #Users review for that product
    
    catItem = getCate([the_prod], unique_products, metadata)
    match = list(set(catUser).intersection(catItem))
    weight = len(match)/len(catItem)
    #print(weight)
    users_review = users_review*weight  #So we have weighted user review based on how similar that item is to all rated items
    #print(the_prod)
    
    #print(users_review)
       #find all users who have reviews 'the_prod'
    users_with_prod = np.where(nonan_test[:, the_prod] != 0)[0]
    for i in users_with_prod:
        if i != user:
            sim = user_sim_mat[i][user]
            simuser.append([sim, i])
    simuser.sort()
    simuser.reverse()
    
    for item in simuser[:k]:
        simu.append(item[1])
    
    #print(simu)
    
    rat = 0
    othusers_review = 0
    for kuser in simu:
        rat = nonan_test[kuser][the_prod]
        othrated_prod = np.where(nonan_test[kuser,:] != 0)[0]
        othrated_prod = list(filter((the_prod).__ne__, othrated_prod))
        #othusers_products = list(np.nonzero(nonan_test[kuser]))  #list of products rated by similar user
        othusers_cat = getCate(othrated_prod, unique_products, metadata)
        match = list(set(othusers_cat).intersection(catItem))
        #print(match)
        #print(othusers_cat)
        #print(catItem)
        weight = len(match)/len(catItem)
        #print(weight)
        othusers_review += rat*weight
        
    pred_rating = othusers_review/k
    #print(pred_rating)
    #print(users_review)
    return pred_rating, users_review

#performs many hybrid runs, using collabo_evaluator, returns mae
def calc_mae_hyb(nonan_test, ratio, k, unique_products, metadata, user_sim_mat):
    
    #Get the subset of users
    num_users = np.shape(nonan_test)[0]
    list_of_users = list(range(0,num_users))
    np.random.shuffle(list_of_users)
    subset_size = int(num_users * ratio)
    subset_users = list_of_users[0:subset_size]
    #print("Length is:",len(subset_users))
    abserror = 0
    count = 0
    for wuser in subset_users:
        pred, userr = collabo_evaluator_hyb(wuser, k, nonan_test, unique_products, metadata, user_sim_mat)
        abserror += abs(pred-userr)
        #print(abserror)
        count += 1
        #print(wuser)
    MAE = abserror/count
    print("MEAN ABSOLUTE ERROR: ", MAE)

#performs many hybrid runs, using collabo_evaluator, returns mae    
def calc_mae_user(nonan_test, ratio, k, user_sim_mat):
    
    #Get the subset of users
    num_users = np.shape(nonan_test)[0]
    list_of_users = list(range(0,num_users))
    np.random.shuffle(list_of_users)
    subset_size = int(num_users * ratio)
    subset_users = list_of_users[0:subset_size]
    #print(subset_users)
    abserror = 0
    count = 0
    for wuser in subset_users:
        pred, userr = collabo_evaluator(wuser, k, nonan_test, user_sim_mat)
        abserror += abs(pred-userr)
        #print(abserror)
        count += 1
    MAE = abserror/count
    print("MEAN ABSOLUTE ERROR: ", MAE)

#If there is ever a time where NO recommendations are given to a user...
#We can use this function to get the best-reviewed items
def getBestItems(nonan_test):
   best_items = []
   ranked = []
   for item in range(50):#nonan_test.shape[1]):
       item_review = nonan_test[:,item]
       #print(item_review)
       nonzero_review = np.where(item_review != 0)
       len_nz = len(nonzero_review[0])
       item_review = np.extract(nonzero_review, item_review[nonzero_review])
       #print(item_review)
       mean_review = np.mean(item_review)
       best_items.append([mean_review, item, len_nz])
   best_items.sort()
   best_items.reverse()
   print(best_items)
   print('/n')
   for item in best_items:
       if item[0] == 5.0:
           ranked.append([item[2],item[1]])
   
   return ranked