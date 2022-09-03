import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

def plot_map(data, column):
    countries=data[column].value_counts()

    map_data=dict(type='choropleth',
                    locations=countries.index,
                    locationmode='country names', 
                    z=countries,
                    colorscale=[[0, 'rgb(214,255,255)'],
                        [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
                        [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
                        [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
                        [1, 'rgb(227,26,28)']], reversescale=False)

    layout=dict(geo=dict(showocean=True, 
                    oceancolor="LightBlue"),
                )

    choromap=go.Figure(data=map_data, layout=layout)
    choromap.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    choromap.update_traces(showscale=False)
    st.plotly_chart(choromap)


def plot_price_dist(data):
    plt.figure(figsize=(10, 6))

    fig=px.histogram(data, x="UnitPrice", nbins=20,
                        labels={
                        "UnitPrice": "Price "},
                        color_discrete_sequence=['darkred']).update_layout(yaxis_title="Price")

    fig.update_traces(marker_line_width=1,marker_line_color="white")

    fig.update_layout(
        margin=dict(l=20, r=40, t=40, b=20),
    )
    st.plotly_chart(fig)


def plot_quantity_dist(data):
    fig=px.histogram(data, x="Quantity", nbins=20,
                    labels={
                    "Quantity": "Quantity "},
                    title="Quantity Distribution",
                    color_discrete_sequence=['darkblue']).update_layout(yaxis_title="Price")

    fig.update_traces(marker_line_width=1,marker_line_color="white")
    fig.update_layout(
        margin=dict(l=20, r=40, t=40, b=20),)
    st.plotly_chart(fig)


def plot_popular_items(data):
    items=pd.DataFrame(data['Description'].value_counts()).reset_index()
    items.rename(columns={'index':'Item', 'Description':'No. of Orders'}, inplace=True)
    fig=px.bar(items.head(10), x='Item', y='No. of Orders',
            color='No. of Orders')
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(showlegend=False)
    st.plotly_chart(fig)

def plot_radar(data, cluster_num, title):
    categs=['categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(
        theta=categs,
        r=data[categs].iloc[cluster_num, :].values,
        fill='toself'))
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)