import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.express as px
import plotly.graph_objs as go
from plots import *
from pipeline import *


st.title('Online Retail Customer Segmentation')
st.write("EDA, Clustering and RFM Analysis of online retail data.")

bar=st.sidebar.selectbox(
    "Sections",
    ("EDA", "Clustering & RFM")
)

def find_outliers(data, column, coef):
    q1, q3=data[column].describe()[['25%', '75%']]
    iqr=q3-q1
    outliers=data[(data[column]<=(q1-coef*iqr)) | (data[column]>=(q3+coef*iqr))]
    return outliers


def read_and_prep_data(path):
    data=pd.read_csv(path, encoding="ISO-8859-1")
    data.dropna(subset=['CustomerID'], inplace=True)
    out_quantity=find_outliers(data, 'Quantity', 3)
    out_price=find_outliers(data, 'UnitPrice', 3)
    out_quantity_idx, out_price_idx=set(out_quantity.index), set(out_price.index)
    all_outliers_idx=out_quantity_idx | out_price_idx
    data.drop(all_outliers_idx, inplace=True)
    cancelled_idx=data[data['InvoiceNo'].apply(lambda x: 'C' in x)].index
    data.drop(cancelled_idx, inplace=True)
    return data


df=read_and_prep_data('data/data.csv')

if bar=='EDA':
    st.title('Exploratory Data Analysis')
    with st.container():
        st.header("Plot One: Number of Orders by Country")
        plot_map(df, 'Country')

    with st.expander("Abotu Orders by Countries"):
        st.write('''
        As the data comes from UK-based store, it's no surprise that vast majority of orders are from the UK itself (90%+).
        ''')

    with st.container():
        st.header("Plot Two: Price Distribution")
        plot_price_dist(df)

    with st.expander("About Price Distribution"):
        st.write('''
        Price distribution of orders shows that
        people tend to buy pretty cheap (0.5-2.5 dollars) items. 
        What about Quantities? Maybe it's common to buy cheap things, but in bulk?
        ''')

    with st.container():
        st.header("Plot Three: Quantity Distribution")
        plot_quantity_dist(df)

    with st.expander("About Quantity Distribution"):
        st.write('''
        No, people just buy 1-4 cheap items in most cases, that's all.
        ''')

    with st.container():
        st.header("Plot Four: Most Popular Items")
        plot_popular_items(df)

    with st.expander("About Most Popular Items"):
        st.write('''
        As the store specializes in small cheap accessories, we see that their bestselllers 
        are things like t-light holders, bags and buntings.
        ''')

elif bar=='Clustering & RFM':
    uploaded_file=st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df_test=pd.read_csv(uploaded_file)
        data, matrix, prod_description, price_basket=prepare_test_data_for_clustering(df_test)
        pred, top_products=clustering(data, matrix, prod_description, price_basket)
        st.title('Customer Segmentation')
        st.success(f'The customer is from cluster: {pred}')
        cluster_data=pd.read_csv('data/customer_cluster_data.csv')
        title=f'Product Category Distribution For Cluster {pred}'
        plot_radar(cluster_data, pred, title)
        st.write('Most Popular Products In Customer\'s Top Category:')
        for i in top_products:
            st.markdown("- "+i)

        st.title('RFM Analysis')
        rfm_data=prepare_test_data_for_rfm(df_test)
        rfm_score=rfm_scoring(rfm_data)
        st.success(f"Customer\'s RFM Score: {rfm_score}")
        segment, description, marketing=evaluate_rfm(rfm_score)
        st.write(f"Customer\'s Segment: {segment}.")
        st.write(f"Customer\'s Description: {description}.")
        st.write(f"Marketing Strategy: {marketing}.")
    else:
        st.warning("You need to upload a CSV file.")
