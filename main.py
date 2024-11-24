from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    text : object
    

# loading the saved model
classification_model = pickle.load(open('LinearSVC.pkl','rb'))
tfidf_vector = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))


@app.post('/cyberbullying_detection')
def cb_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    content = input_dictionary['text']
    
    input_list = [content]

    vectorized_data = tfidf_vector.fit_transform(input_list)
    prediction = classification_model.predict(vectorized_data)
    
    if prediction[0] == 0:
        return 'Cyberbullying'
    else:
        return 'Non-cyberbullying'

