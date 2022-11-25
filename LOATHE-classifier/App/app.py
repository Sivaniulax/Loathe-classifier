
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
image = Image.open('hateimage.png')


pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def prediction(temp):
	prediction = classifier.predict([temp])
	if prediction == [0]:
		print("HATE SPEECH 😠😔")
		X = "HATE SPEECH 😠😔"

	elif prediction == [1]:
		print("OFFENSIVE LANGUAGE 🤮😨😱")
		X = "OFFENSIVE LANGUAGE 🤮😨😱"
	
	elif prediction == [2]:
		print("NOT HATE SPEECH 🤗😮😂")
		X = "NOT HATE SPEECH 🤗😮😂"
		
	return X

def main():
	# giving the webpage a title
	st.title("LOATHE - DETECTOR")
	st.header("This is a Hate-speech detection app")
	st.image(image)
	
    
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    
	temp = st.text_input("Enter your data:", "Type Here")
	result =""
	
	if st.button("Predict"):
		result = prediction(temp)
	st.success('The output is {}'.format(result))
       


if __name__=='__main__':
    main()