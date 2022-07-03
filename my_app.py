from re import A
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


#image
img=Image.open("autos.jpg")
#st.image(img, caption="catie", width=800)
st.image(img)


html_temp = """
<div style="background-color:purple;padding:1.5px">
<h1 style="color:black;text-align:center;">Sell your car at its value!</h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

html_temp1 = """
<div style="background-color:cyan;">
<text style="color:purple;text-align:center;font-size:140%;">All you need to give your car's general properties and get the optimal price!</text>
</div><br>"""
st.markdown(html_temp1,unsafe_allow_html=True)

html_temp = """
<div style="background-color:green;">
<p style="color:white;text-align:center;font-size:160%;">Your Car's Properties</p>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

power=st.sidebar.slider("Select your car's power in kw", 50, 300, step=1)
age=st.sidebar.select_slider("Select your car's age", [0,1,2,3,4])
km_stand=st.sidebar.slider("Write your car's km stand",0,300000,step=1000)
brand_model=st.sidebar.selectbox("Select your car's brand and model", ['Audi A1','Audi A3','Opel Insignia','Opel Astra','Opel Corsa','Renault Clio','Renault Espace','Renault Duster'])
gearing_type=st.sidebar.selectbox("Select your car's gearing type", ('Manual', 'Automatic','Semi-automatic'))
def your_car():
    my_dict = {"Power" :power,
        "Age":age,
        "KM Stand": km_stand,
        "Brand Model": brand_model ,
        "Gearing Type": gearing_type}
    df_sample = pd.DataFrame.from_dict([my_dict])
    return df_sample
df = your_car()
st.table(df)

#using trained models
import pickle


final_model=pickle.load(open("final_model.pkl","rb"))
final_scaler=pickle.load(open("final_scaler.pkl","rb"))
my_dict = {
    "hp_kW": power,
    "age": age,
    "km": km_stand,
    "make_model": brand_model,
    "Gearing_Type": gearing_type
}
my_dict = pd.DataFrame([my_dict])

my_dict = pd.get_dummies(my_dict)
Xcolumns=['hp_kW', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
       'make_model_Opel Astra', 'make_model_Opel Corsa',
       'make_model_Opel Insignia', 'make_model_Renault Clio',
       'make_model_Renault Duster', 'make_model_Renault Espace',
       'Gearing_Type_Automatic', 'Gearing_Type_Manual',
       'Gearing_Type_Semi-automatic']

my_dict = my_dict.reindex(columns = Xcolumns, fill_value=0)

my_dict = final_scaler.transform(my_dict)

if st.button("To get your car's price, press this button"): 
    price=int(final_model.predict(my_dict)[0])
    st.markdown(f"""
      #### <span style="background-color:yellow;color:red;font-size:32px;border-radius:2%;text-align:center"> Your car's actual price is ${str(price)}  </span>
      # """,unsafe_allow_html=True)










