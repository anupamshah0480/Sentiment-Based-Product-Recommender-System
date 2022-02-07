import matplotlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)

df = pd.read_csv("capstone_data.csv", encoding='utf8')

#Lets fetch out the columns required for creating similarity matrices
df_main = df[['id', 'reviews_username', 'reviews_rating']]

#Renaming the columns for ease of understanding and coding
df_main = df_main.rename(columns= {'id':"product_id", 'reviews_username':"user", 'reviews_rating':"ratings"})

#Lets split the data into train and test before jumping into building different type of matrices required for recommendation.
train, test = train_test_split(df_main, test_size=0.30, random_state=42)

#Lets create a pivot table for user vs product rating to get a basic idea
df_pivot = train.pivot_table(
    index = 'user',
    columns = "product_id",
    values = 'ratings'
).fillna(0)

# Creating the dummy_train dataset
# 1. take a copy of train data in a new df
# 2. The products not rated by user is marked as 1
# 3. Convert the dummy train dataset into matrix format

dummy_train = train.copy()

dummy_train['ratings'] = dummy_train['ratings'].apply(lambda x: 0 if x>=1 else 1)

dummy_train = dummy_train.pivot_table(
    index='user',
    columns='product_id',
    values='ratings'
).fillna(1)

# Creating the User Similarity Matrix using pairwise_distance function and metric will be COSINE.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0

# Now lets create a user similarity matrix with Adjusted Cosine Similarity approach

""" 
- Here we will be needing NaNs as we will first first find the mean which will store the values of means corresponding to each 
  product_id in the dataset for all the users 

- Once we have the mean array, we need to subtract each user's rating from the mean [for this we will be creating a new pivot table 
  ‘df_subtracted’ using ‘df_subtracted = (df_pivot.T-mean).T](Normalizing the user rating around mean 0)

- The ‘df_pivot’ has users in rows and products in columns, and you need to subtract each column from the data available in the 
  ‘mean’ array (1D array), and this is why you need to do ‘df_pivot.T’

- Then finally we  need to transpose the (df_pivot.T - mean) to finally get ‘df_substracted’

"""

df_pivot = train.pivot_table(
    index='user',
    columns='product_id',
    values='ratings'
)

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T - mean).T

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0

#Doing the prediction for the users which are positively related with other users, and not the users which are negatively
# related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for
# values less than 0.

user_correlation[user_correlation<0]=0

#getting predictions for the movies rated and not rated by the user
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

# Lets get final pivot
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

user_final_rating_df = pd.DataFrame(user_final_rating)

# pickeling the user_final_rating matrix to use in the main py file to get recommendations
user_final_rating_df.to_pickle("./final_rating.pkl")

