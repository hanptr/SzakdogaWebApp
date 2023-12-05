import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow as tf

transformer_model=pickle.load(open('transformer_model.sav', 'rb'))
bidirectional_model=pickle.load(open('bidirectional_model.sav', 'rb'))
 

with st.sidebar:
   
   selected = option_menu('Running intensity classification',
                         
                         ['Data analytics',
                          'Classification',
                          'Upload CSV'],
                         icons=['clipboard_data','heart','cloud_upload'],
                         default_index=0)
   
   if (selected == 'Data analytics'):
       # page title
       st.title('Analyzing the data')
       
   if (selected == 'Data Classification'):
       # page title
       st.title('Classifying the data')
       
   if (selected == 'Upload CSV'):
       # page title
       st.title('Upload the desired CSV')
