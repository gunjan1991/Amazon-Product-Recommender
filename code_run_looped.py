#Read the data
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn import *
import final_functions as our_funcs


print("\n\n*****--------------*****\n")
logo = '''
 /$$$$$$  /$$$$$$/$$$$   /$$$$$$  /$$$$$$$$  /$$$$$$  /$$$$$$$ 
 |____  $$| $$_  $$_  $$ |____  $$|____ /$$/ /$$__  $$| $$__  $$
  /$$$$$$$| $$ \ $$ \ $$  /$$$$$$$   /$$$$/ | $$  \ $$| $$  \ $$
 /$$__  $$| $$ | $$ | $$ /$$__  $$  /$$__/  | $$  | $$| $$  | $$
|  $$$$$$$| $$ | $$ | $$|  $$$$$$$ /$$$$$$$$|  $$$$$$/| $$  | $$
 \_______/|__/ |__/ |__/ \_______/|________/ \______/ |__/  |__/
                                                                
'''
print(logo)
print("Recommender systems \nLOADING DATA (may take several minutes)...\n\n")
# !! Just read in data_df as array or dataframe
data_df = pd.read_csv("Clothing_Shoes_Jewelry_5.csv")
data_df = data_df.drop('Unnamed: 0', 1)

#Getting unique items
unique_products = data_df.asin.unique()

#Get metadata
metadata = pd.read_csv("clothing_shoes_jewelry_metadata.csv")
metadata = metadata.drop('Unnamed: 0', 1)
metadata['flag'] = metadata['asin'].isin(unique_products)
metadata = metadata[metadata['flag'] == True]

#data_df_small = data_df.ix[0:4000,:]

#Read in the user-item matrix SUBSET (a smaller version for easier runtimes)
nonan_test = np.loadtxt("smallUserItem/smallUserItem.txt")

# !! Just read in nonan_test as a numpy array (possible?)
#If not, here's the work:
#data_df_slim = data_df_small[['asin', 'overall', 'reviewerID']]

#Creating user similarity matrix
user_sim_mat = metrics.pairwise.cosine_similarity(nonan_test)

#Creating item similarity matrix
nonan_test_t = nonan_test.T
itm_sim_mat = metrics.pairwise.cosine_similarity(nonan_test_t)

print("\n\nWelcome to the Amazon clothing, shoes & jewelry recommender")
print('\n')
loopkey = "Yes"
while loopkey == "Yes":
    random_or_input = input("\n\nWould you like to review some products and get recommendations, or see recommendations for a random user? \n (type 'User input' or 'Random user'): ")
    print('\n')
    print("You selected: ", random_or_input)

    if random_or_input == 'User input':
        user_rev = [0]*np.shape(nonan_test)[1]
        print("Please rate the following items (1-5):\n")
        for i in range(5):
            i += 1
            random_prod = np.random.randint(0, np.shape(nonan_test)[1])
            asin_pd = unique_products[random_prod]
            product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
            product = str(product)
            user_review = float(input("Please rate " + product + ": "))
            user_rev[random_prod] = user_review
            #print(random_prod)

        #np.insert(nonan_test, len(nonan_test), user_rev, 0)
        #serve up random rows and get 5 reviews for 5 prods
        #result: a list of same length as array's width: w/ reviews at the correct spot based on column index
        u1 = np.shape(nonan_test)[0]
        nonan_test = np.append(nonan_test, np.array([user_rev]), axis = 0)                               

    else:
        random_row = np.random.randint(0, np.shape(nonan_test)[0])
        u1 = random_row

    print('\n')
    unrated, rated = our_funcs.getUnrated(nonan_test,u1)
    #print info about the user: "product name: review"
    for item in rated:
        asin_pd = unique_products[item]
        product = metadata[metadata['asin']==asin_pd].title.item()# 'categories']]
        product = str(product)
        print("User Rated:" + product)



    ##############################	
    #Let responder give rec method
    ##############################
	
    rec_method = input("\n\nSelect recommendation method \n(type 'User-based', 'Item-based', 'Hybrid'): ")
    print('\n')
    num_recs = int(input("\nNumber of products to recommend? (Input an integer:)\n"))
    print('\n')
    #Three branches, based on input
    #User-based branch
    if rec_method == 'User-based' or rec_method == 'Hybrid':
        n = int(input("How many similar users to base rec's off of? (Input an integer):\n"))
        if random_or_input == 'Random user':
            simUsers = our_funcs.getUserPC(user_sim_mat,u1,n)
        else:
            simUsers = our_funcs.getUser(nonan_test, u1, n, similarity=our_funcs.find_sim)
            
        if rec_method == 'User-based':

            print('\n')

            #Serve up the rec's

            our_funcs.userBasedRecs(nonan_test, simUsers, unrated, num_recs, rated, unique_products, metadata)

        #Hybrid branch
        elif rec_method == 'Hybrid':

            our_funcs.hybridRec(nonan_test, simUsers, unrated, num_recs, rated, unique_products, metadata)

    #Item-based branch
    elif rec_method == 'Item-based':

        our_funcs.itemSim(nonan_test, u1, num_recs, unique_products, metadata, itm_sim_mat, unrated, rated)


    #########################
    # Validation of results
    #########################
    validation_key=input("\nDo you want to see how we validate our recommenders? (type 'Yes' or 'No'):\n ")
    if validation_key == 'Yes':
	    #Now that we've reached the end of the loop, delete the input row if the user made one        
        if random_or_input == 'User input':
            np.delete(nonan_test, u1, 0)
        print("\n\n -- Recommender evaluator -- \n\n")
        print("Now that you've seen some examples, let's talk about performance \n")

        rec_method = input("Which recommendation method do you want to evaluate? \n(type 'User-based', 'Item-based', 'Hybrid'): ")
        ratio = float(input("How much of the data do you want to validate on? \nInput a decimal value between 0 and 1 \nFor reference '0.01' would perform about 80 recommendations: \n"))
        if rec_method == 'User-based' or rec_method == 'Hybrid':
            n = int(input("How many similar users to base rec's off of? (type an integer)\n"))
            eval_method = input("How do you want to evaluate the recommender system? \n(type 'MAE' or 'Recall')")
            num_recs = int(input("How many recommendations to give?(type an integer)\n"))
            print('\n')
            #User branch
            if rec_method == "User-based":
                if eval_method == 'MAE':
                    our_funcs.calc_mae_user(nonan_test, ratio, n, user_sim_mat)
                elif eval_method == 'Recall':
                    # !! needs work 
                    our_funcs.user_based_recall(nonan_test, ratio, n, num_recs, unique_products, metadata, user_sim_mat)

            #Hybrid branch
            elif rec_method == 'Hybrid':
                if eval_method == 'MAE':
                    our_funcs.calc_mae_hyb(nonan_test, ratio, n, unique_products, metadata, user_sim_mat)
                elif eval_method == 'Recall':
                    # !! needs work 
                    our_funcs.hybrid_recall(nonan_test, ratio, n, num_recs, unique_products, metadata, user_sim_mat)

        elif rec_method == 'Item-based':
            eval_method = input("How do you want to evaluate the recommender system? \n(type 'MAE' or 'Recall')")
            num_recs = int(input("How many recommendations to give? \n(type an integer): "))

            if eval_method == 'MAE':
                our_funcs.calc_mae_itm(nonan_test, ratio, n, itm_sim_mat)

            elif eval_method == 'Recall':
                our_funcs.item_based_recall(nonan_test, ratio, n, num_recs, unique_products, metadata, itm_sim_mat)

    print("\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    loopkey = input("Do you want to start over again? (Type 'Yes' or 'No':\n")
print("\n\nTHANK YOU FOR USING: GOODBYE!! \n")
byebye = '''
 ____ _  _,____,  ____ _  _,____, 
(-|__|-\_/(-|_,  (-|__|-\_/(-|_,  
 _|__) _|, _|__,  _|__) _|, _|__, 
(     (   (      (     (   (  
'''
print(byebye)

