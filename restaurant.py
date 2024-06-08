import pickle
import streamlit as st
import numpy as np
import pandas as pd
import json
from streamlit_lottie import st_lottie
import plotly.express as px

st.title("Restaurant ratings prediction app")
df = pd.read_csv("dataset.csv")


with open("Label_encoder.pkl", "rb") as file:
    Label_encoder = pickle.load(file)

with open("Training_data.pkl", "rb") as file:
    X = pickle.load(file)

with open("restaurant_Name.pkl", "rb") as file:
    restaurant_name = pickle.load(file)

with open("Cuisines.pkl", "rb") as file:
    Cuisines = pickle.load(file)

with open("City.pkl", "rb") as file:
    City = pickle.load(file)

with open("Currency.pkl", "rb") as file:
    Currency = pickle.load(file)

with open("Has_table.pkl", "rb") as file:
    Has_table = pickle.load(file)

with open("Has_online.pkl", "rb") as file:
    Has_online = pickle.load(file)

with open("Is_delivery.pkl", "rb") as file:
    Is_delivery = pickle.load(file)

with open("Rating_text.pkl", "rb") as file:
    Rating_text = pickle.load(file)

with open("Restaurant_scaling.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("Restaurant_model.pkl", "rb") as file:
    rfr = pickle.load(file)

with st.sidebar:

    @st.cache_data(ttl=60 * 60)
    def load_lottie_file(filepath : str):
        with open(filepath, "r") as f:
            gif = json.load(f)
        return gif

    gif = load_lottie_file("restaurant.json")

    # Display the animation with a placeholder image while loading
    with st.spinner("Loading animation..."):
        st_lottie(gif, speed=1, width=350, height=250)

    City_val = st.selectbox("Select city name: ",City)
    lat = df[df["City"] == City_val]["Latitude"].tolist()
    lat_val = st.selectbox("Latitude values of selected city: ",lat)
    lon = df[df["City"] == City_val]["Longitude"].tolist()
    lon_val = st.selectbox("Longitude values of selected city: ",lon)
    Price_range = st.selectbox("Select price range: ",df["Price range"].unique())
    st.warning("**Note**: 1:**Low**,2:**Medium**,3:**High**,4:**Premium**")
    Average_Cost_for_two = st.text_input("Enter price of two items: ",key=1)
    Votes = st.text_input("Enter reviews count: ",key=2)
    restaurant_name_val = st.selectbox("Select restaurant name: ",restaurant_name)
    Cuisines_val = st.selectbox("Select Cuisine type: ",Cuisines)
    Currency_val = st.selectbox("Select currency used: ",Currency)
    Has_table_val = st.selectbox("Select table booking status: ",Has_table)
    Has_online_val = st.selectbox("Select it has online facility: ",Has_online)
    Is_delivery_val = st.selectbox("Select it has online delivery: ",Is_delivery)
    Rating_text_val = st.selectbox("Select rating text: ",Rating_text)
    but = st.button("Click here to predict",use_container_width=True)

if but:
    sample = {
        "Longitude": lat_val,
        "Latitude": lon_val,
        "Price_range_log":np.log(int(Price_range)) ,
        "Average_Cost_for_two_log" : np.log(int(str(Average_Cost_for_two))) ,
        "Votes_log" : np.log(int(Votes)) ,
        "Restaurant Name" : restaurant_name_val ,
        "Cuisines" : Cuisines_val,
        "City" : City_val ,
        "Currency" : Currency_val,
        "Has Table booking" : Has_table_val ,
        "Has Online delivery" : Has_online_val,
        "Is delivering now" : Is_delivery_val ,
        "Rating text" : Rating_text_val
    }

    sample_df = pd.DataFrame([sample])

    sample_df["Restaurant Name"] = Label_encoder.transform(sample_df["Restaurant Name"])

    binary_columns = ['Has Table booking','Has Online delivery','Is delivering now']

    for col in binary_columns:
        sample_df[col] = sample_df[col].map({"Yes":1,"No":0}) # map() is used to map 1 with each Yes class value and 0 with each No class

    one_hot_columns = ["City","Currency","Rating text"]
    sample_df = pd.get_dummies(sample_df,columns=one_hot_columns,dtype=int)

    # to handle categorical classes of encoding which is not there in our training
    for col in X.columns:
        if col not in sample_df.columns:
            sample_df[col] = 0

    # Target encoding
    target_mean = df.groupby("Cuisines")["Aggregate_rating_squared"].mean()
    sample_df["Cuisine_target_encoded"] = sample_df["Cuisines"].map(target_mean)

    sample_df.drop(columns=["Cuisines"],inplace=True)

    sample_df = sample_df[X.columns] # arranging columns based on training columns


    scaler.fit(X) # here we again fit the scaling to training because , in previous i use fit_transform at a time so that's why my new df is not transformd properly

    new_scaled_sample_df = scaler.transform(sample_df) # transform the sample_df based on above fitted training scaler

    predict = rfr.predict(new_scaled_sample_df) # we are predicting sample by using RandomForestRegressor model

    



    # Adding custom CSS styles using HTML within Markdown
    st.markdown(
        """
        <style>
            .grid-container {
                display: grid; /* Setting display property to grid for grid layout */
                grid-template-columns: repeat(3, 1fr); /* Setting grid to have 3 columns */
                grid-gap: 20px; /* Adding gap between grid items */
            }
            .grid-item {
                background-color: #f9f9f9; /* Setting background color for grid items */
                padding: 20px; /* Adding padding around grid items */
                border-radius: 5px; /* Adding border radius to grid items */
                margin-bottom: 20px; /* Adding margin to create space between rows */
            }
            .grid-item h2 {
                color: #333333; /* Setting color for h2 headings */
                margin-bottom: 10px; /* Adding margin below h2 headings */
            }
            .grid-item h3 {
                color: #ffA500;  /* Setting color for h3 headings */
                margin-bottom: 10px; /* Adding margin below h3 headings */
                font-size: 40px;
            }
            .grid-item p {
                color: black; /* Setting color for paragraph text */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Round the prediction
    rounded_prediction = np.round(np.sqrt(*predict), 2)
    # Creating HTML elements with extracted data and adding them to the columns list
    st.markdown(f"<div class='grid-item'><h2 style='color: #581845;'>🌟 Restaurant predicted ratings:<h3>{rounded_prediction}</h3></h2></div>",unsafe_allow_html=True)

    st.subheader("Map view of selected city restaurants")
    st.warning("**Note**: Bubble size is based on ratings of the restaurant")
    geo_df = df[df["City"] == City_val]
    st.dataframe(geo_df.head())
    fig = px.scatter_mapbox(
        geo_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Restaurant Name",
        hover_data=["Cuisines", "Aggregate rating", "City"],
        color="Restaurant Name",
        size='Aggregate rating',  # Adjust the size as needed
        zoom=10,  # Increase the zoom level
        center={"lat": geo_df["Latitude"].mean(), "lon": geo_df["Longitude"].mean()},  # Center the map on the average location
        mapbox_style="carto-positron",  # Specify a different map style if needed
        height=600
    )


    st.plotly_chart(fig)
