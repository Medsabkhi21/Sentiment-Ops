import streamlit as st
import pandas as pd
import json

import inference




# defines an h1 header
text = st.text_input("Insert tweet here")


st.subheader("Sentiment:")

#st.write( query_endpoint(app_name=app_name, input_json=input_df))
if st.button('predict'):
    
    pred = inference.prediction(text)
    st.write('result: %s' % pred)


    