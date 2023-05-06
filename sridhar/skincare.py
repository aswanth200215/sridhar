import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
#computes the similarity between the enterd value and the valus in our df
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request

skincare = Flask(__name__)

@skincare.route('/')
def home():
    return render_template('home.html')

@skincare.route('/search' , methods=['POST'])
def search():
    keyword = request.form['keyword']
    df = pd.read_csv('C:/Users/Admin/Desktop/skindataall.csv')
    df.head()
    #removing unwanted col
    df.drop('Skin_Tone', inplace=True, axis=1)
    df.drop('Eye_Color', inplace=True, axis=1)
    df.drop('Hair_Color', inplace=True, axis=1)
    df.drop('Product_Url', inplace=True, axis=1)
    df.drop('Username', inplace=True, axis=1)
    df.drop('Good_Stuff', inplace=True, axis=1)
    df.drop('Ingredients', inplace=True, axis=1)
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df.drop('Review_Cleaned', inplace=True, axis=1)
    df.drop('Price', inplace=True, axis=1)
    df.drop('User_id', inplace=True, axis=1)
    df.drop('Review', inplace=True, axis=1)
    df.drop('Skin_Type', inplace=True, axis=1)

    #drop the face mask 
    df = df[~df.Category.str.match('Face Mask')]

    #checking duplicates
    df.duplicated(subset='Product_id').sum() 

    #removing them
    df = df.drop_duplicates(subset='Product_id')

    #checking if they were removed
    df.duplicated(subset='Product_id').sum() 

    df.reset_index(drop=True)

    #clean Ingredients_Cleaned more
    df.replace(',','', regex=True, inplace=True)
    df.head()

    #remove zeros and change one to col names in new col 

    # changing data types of compatibility columns from int to str so we can manipulate them
    df = df.astype({'Combination':'string','Dry':'string','Normal':'string','Oily':'string','Sensitive':'string'})

    #   replacing all 1's with col name 
    # #Combination	Dry	Normal	Oily	Sensitive
    df['Combination'] = df['Combination'].str.replace('1','combinational')
    df['Combination'] = df['Combination'].str.replace('0',' ')

    df['Dry'] = df['Dry'].str.replace('1','dry')
    df['Dry'] = df['Dry'].str.replace('0',' ')


    df['Normal'] = df['Normal'].str.replace('1','normal')
    df['Normal'] = df['Normal'].str.replace('0',' ')


    df['Oily'] = df['Oily'].str.replace('1','oily')
    df['Oily'] = df['Oily'].str.replace('0',' ')


    df['Sensitive'] = df['Sensitive'].str.replace('1','sensitive')
    df['Sensitive'] = df['Sensitive'].str.replace('0',' ')

    #Adding Skin sutibility column to combine what the product is good for as a feature 
    df['Skin_sutibility'] = df['Combination'].map(str) + ' ' + df['Dry'].map(str) + ' ' + df['Normal'].map(str)+ ' ' + df['Oily'].map(str) + ' ' +df['Sensitive'].map(str)

    #making the product name in lower case because the user will probably enter it that way
    df['Product'] = df['Product'].str.lower()

    #now we remove the extra columns does it have to be 
    df.drop(['Rating','Combination','Dry','Normal','Oily','Sensitive','Ingredients_Cleaned'],inplace=True,axis=1)

    df.info()

    #checking fo null values
    df.isnull().values.any()

    #changing the index of some columns so that we dont add them in the combined data col
    df = df [['Product_id','Product','Rating_Stars','Brand', 'Category','Skin_sutibility','Ing_Tfidf']]

    df.sample(5)

    #creating a column with the important features 
    df['combined_data'] = df[df.columns[4:]].apply( 
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
    )

    df.sample(10)

    def recommender (product_name):
    
        #turning our combined data col to a vector to measure its similarity against other vectors
        vec = CountVectorizer()
        vectorized = vec.fit_transform(df['combined_data'])

        #calculate cosine sim matrix from the vectors 
        cs = cosine_similarity(vectorized)
        #get the name of product 
        #product_name = 'cleanser'
        #finding the product id
        product_id = df[df.Product == product_name]['Product_id'].values[0]

        #create enumerations for similarity scores and sort it 
        score = list(enumerate(cs[product_id]))
        sorted_score = sorted(score, key = lambda x:x[1], reverse= True)[1:]
        i = 0
        print ('the 5 most similar product to your choice are:\n ')
        for item in sorted_score:
            results = []
            product_namef = df[df.Product_id == item[0]]['Product'].values[0]
            results.append(str(product_namef)+' ' +str(sorted_score[i][1])+' ' )
            i = i+1
            if i == 5:
                break
            return results
    result=[]
    result=recommender(keyword)
    return render_template('result.html',result=result)


if __name__=='__main__':
    skincare.run(debug=True)