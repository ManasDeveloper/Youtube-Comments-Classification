import streamlit as st 
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

stemmer = PorterStemmer()

model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))

st.title("Youtube comments classification")

input_comment = st.text_input("Enter the message")



#1. preprocess the text

def transform_text(text):
  text = text.lower()

  # tokenize
  text = nltk.word_tokenize(text)

  y = []
  # remove special characters
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  #remove the stopwords
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  # stemming
  for i in text:
    y.append(stemmer.stem(i))






  return " ".join(y)

if st.button("Predict"):

   transform_comment = transform_text(input_comment)

   #2. Vectorize
   vector_input = tfidf.transform([transform_comment])


   #3. Predict
   prediction = model.predict(vector_input)[0]

   if prediction == 1:
     st.header("Spam")
   if prediction == 0:
     st.header("Legitimate")