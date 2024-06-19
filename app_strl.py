import streamlit as st
#pip install streamlit  <------ to install 
#streamlit run app_strl.py <--- to run

import numpy as np
import pickle
model = pickle.load(open('lrg.pkl','rb'))

st.title("TECHNOLOGIES")
st.header("House Price Predictor")
def main():
    bedroom = st.number_input("Enter Number of Bed Rooms",min_value =1,max_value=20,value=1,step=1)
    bathrooms = st.number_input("Enter Number of Bath Rooms",min_value =1,max_value=20,value=1,step=1)
    sqft = st.number_input("Enter Number Squre Feets",min_value =200,max_value=10000,value=200,step=200)
    
    bt = st.button('Calculate')
    if bt: 
        arr = np.array([bedroom,bathrooms,sqft])
        arr = np.float64(arr)
        arr=arr.reshape(1, -1)
        pred = model.predict(arr) 
        st.write("")
        st.write('Predicted House Price:', pred)
        st.write("")

if __name__=="__main__":
    main()