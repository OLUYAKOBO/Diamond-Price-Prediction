import pandas as pd
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

scaler = pickle.load(open('scal.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
#model = pickle.load(open('diamond.pkl','rb'))
st.title(" *Diamond Price Prediction Application*")
st.write("*This app predicts the market value of **Diamonds***")

st.header("*Enter the details of the Diamond below*")

def user_input():
    c1,c2 =  st.columns(2)
    with c1:
        
        carat = st.number_input('*How many carat is the Diamond*',0.2,3.65,1.3)
        cut  = st.selectbox('*What type of cut is the Diamond*',(['Ideal','Premium','Very Good','Good','Fair']))
        color = st.selectbox('*What is the color type of the Diamond*',(['D','E','F','G','H','I','J']))
        
    with c2:
        
        clarity = st.selectbox('*Select the clarity of the Diamond*', (['SI1', 'VS2','SI2','VS1','VVS2','VVS1','IF','I1']))
        depth = st.number_input("*What is the depth of the Diamond*", 43.0,79.0, 46.7)
        table = st.number_input("*What is the table value of the Diamond*", 43.0,79.0, 46.7)
        
    feat = np.array([carat,cut,color,clarity,depth,table]).reshape(1,-1)
    cols = ['carat','cut','color','clarity','depth','table']
    feat1 = pd.DataFrame(feat, columns=cols)
    return feat1
    
    #data = {
        #'carat':carat,
        #'cut':cut,
        #'color':color,
        #'clarity':clarity,
        #'depth':depth,
        #'table':table}
    #features = pd.DataFrame(data, index = [0])
    #return features

df = user_input()
#st.write(df)

num_features = df.select_dtypes(exclude = 'object').columns
cat_features = df.select_dtypes(include = 'object').columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
def prepare(df):
    
    enc_data =pd.DataFrame(encoder.transform(df[['cut','color','clarity']]))#.toarray())
    #enc_data.columns = encoder.get_feature_names_out()
    enc_data.columns = encoder.get_feature_names_out(['cut','color','clarity'])
    df = df.join(enc_data)

    df.drop(['cut','color','clarity'],axis=1,
           inplace = True)
    
    cols = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df,columns=cols)
    return df
df = prepare(df)

#st.write(df)                                                                      
model = pickle.load(open('cat_diamond.pkl','rb'))
predictions = model.predict(df)

st.subheader('*Diamond Price*')
if st.button('*Click here to get the price of the **Diamond***'):
    st.write(predictions)
    

                                                                                                                                  
