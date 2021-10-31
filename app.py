## import necessary library
from flask import Flask,request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

## init Flask App
app = Flask(__name__)

# Applying Tfidf Vectorizer 
tfidf = TfidfVectorizer(stop_words='english',max_features=5000,max_df=0.7,ngram_range=(1,3))

# Load Pickle model
loaded_model = pickle.load(open('deploy.pkl', 'rb'))



## Read the dataset, convert it into dataframe
df = pd.read_csv('DataSet/train.csv')
df = df.dropna()
df.reset_index(inplace=True)
x= df['text']
y = df['label']

 
#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


def fake_news_det(news):
    tfid_x_train = tfidf.fit_transform(x_train)
    tfid_x_test = tfidf.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfidf.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print("prediction is :",prediction)
    return prediction

@app.route('/')
def home():
    return "<h1>Deployed sucessfully make POST request at /predict and send a json object as 'news' key </h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.json["news"]
        print(message)
        pred = fake_news_det(message).tolist()
        print(type(pred))
        print(pred[0])
        response = {"prediction":pred[0]}
    else:
        response = {"prediction":"Something went wrong"}
    return response
        

if __name__ == '__main__':  
    app.run(debug=True)