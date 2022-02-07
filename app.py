import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, url_for, request, redirect
import pickle
app = Flask(__name__)

# Loading/unpickeling final user rating dataframe
final_rating = pd.read_pickle("./final_rating.pkl")

# Loading/unpickeling sentiment analysis model
with open('sentiment_analysis_model.pkl' , 'rb') as f:
    sentiment_model = pickle.load(f)

# Loading the TFIDF vectorizer
with open('vectorizer.pkl' , 'rb') as f:
    vectorizer = pickle.load(f)

# Loading the main data that will be used for mapping products
df_map= pd.read_csv('capstone_data.csv')
df_map = df_map[['id', 'name', 'reviews_text']]



# Creating a function to get 5 recommendations for user
def get_recommendation(user_input):

    #getting top 20 recommendations for the user
    #top20_recommendations_list = (final_rating.loc[user_input].sort_values(ascending=False)[:20]).tolist()
    top20_recommendations = (final_rating.loc[user_input].sort_values(ascending=False)[:20])

    #mapping the recommendation to mapping file to get review_text for sentiment analysis
    df_final = pd.merge(top20_recommendations, df_map, left_on='product_id', right_on='id', how='left')
    df_20_prods = pd.merge(top20_recommendations, df_map, left_on='product_id', right_on='id', how='left')

    #dropping the duplicate products
    df_final = df_final.drop_duplicates('id')

    #Vectorizing the text using TF_IDF
    text = vectorizer.transform(df_final['reviews_text'])

    #Predicting the sentiment of the reviews
    preds = sentiment_model.predict(text)
    df_final["model_pred_sentiment"] = preds

    # Filtering out the top 5 products based on sentiment of reviews
    result = df_final[df_final['model_pred_sentiment'] == 1][0:5]
    products_to_recommend = result['name'].tolist()
    top20_recommendations_list = list(set(df_20_prods['name']))[0:20]

    return top20_recommendations_list, products_to_recommend

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/', methods=['POST', 'GET'])
def recommend_products():
    if request.method == "POST":
        #Get user Input
        user_input = request.form['user_inp']

        # Get recommendations for the inputed user
        top_20_products, products = get_recommendation(user_input)
        print(top_20_products)
        # Sending recommendations to the webpage
        return render_template("home.html", username=user_input,
                               P1 = top_20_products[0], P2 = top_20_products[1], P3 = top_20_products[2],
                               P4=top_20_products[3], P5 = top_20_products[4], P6 = top_20_products[5],
                               P7=top_20_products[6], P8 = top_20_products[7], P9 = top_20_products[8],
                               P10=top_20_products[9], P11 = top_20_products[10], P12 = top_20_products[11],
                               P13=top_20_products[12], P14 = top_20_products[13], P15 = top_20_products[14],
                               P16=top_20_products[15], P17 = top_20_products[16], P18 = top_20_products[17],
                               P19 = top_20_products[18], P20 = top_20_products[19],
                               Prod_1=products[0], Prod_2=products[1], Prod_3=products[2],
                               Prod_4=products[3], Prod_5=products[4])

# Flask application configuration
if __name__ == '__main__':
    app.run(debug=True, port=5388)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

