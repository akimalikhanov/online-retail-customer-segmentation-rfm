import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime as dt
import joblib

def prepare_test_data_for_clustering(data):
    data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
    data['InvoiceDate']=data['InvoiceDate'].astype('int64')
    data['TotalPrice']=data['UnitPrice']*data['Quantity']

    price_basket=data.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].agg('sum')
    price_basket.rename(columns={'TotalPrice': 'BasketPrice'}, inplace=True)
    time_mean=data.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].agg('mean')
    price_basket['InvoiceDate_mean']=time_mean['InvoiceDate']
    price_basket=price_basket[price_basket['BasketPrice']>0]
    
    list_prod_names=pd.read_csv('data/list_prod_list.csv').columns

    prod_description=data['Description'].unique()

    X=pd.DataFrame()
    for key in list_prod_names:
        X[key]=list(map(lambda x:int(key.upper() in x), prod_description))

    threshold=[0, 1, 2, 3, 5, 10]
    label_col=[]

    for i in range(len(threshold)):
        if i==len(threshold)-1:
            col=f'{threshold[i]}+'
        else:
            col=f'{threshold[i]}-{threshold[i+1]}'
        label_col.append(col)
        X[col]=0

    for i, prod in enumerate(prod_description):
        prix=data[data['Description']==prod]['UnitPrice'].mean()
        j=0
        while prix>threshold[j]:
            j+=1
            if j==len(threshold):
                break
        X.loc[i, label_col[j-1]]=1

    matrix=X.to_numpy()

    return data, matrix, prod_description, price_basket


def clustering(data, mat, prod_desc, price_basket):
    kmeans_prod=joblib.load('models/kmeans_products.sav')
    categ_pred=kmeans_prod.predict(mat)

    corresp=dict()
    for key, val in zip (prod_desc, categ_pred):
        corresp[key]=val 
    
    data['categ_product']=data.loc[:, 'Description'].map(corresp)

    for i in range(kmeans_prod.n_clusters):
        col=f'categ_{i}'
        df_temp=data[data['categ_product']==i]
        price_temp=df_temp['TotalPrice']
        price_temp=price_temp.apply(lambda x:x if x > 0 else 0)
        data[col]=price_temp
        data[col].fillna(0, inplace=True)

    for i in range(kmeans_prod.n_clusters):
        col=f'categ_{i}'
        temp=data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()[col]
        price_basket[col]=temp

    transactions_per_user=price_basket.groupby(by=['CustomerID'])['BasketPrice'].agg(['count','min','max','mean','sum'])

    for i in range(5):
        col=f'categ_{i}'
        transactions_per_user[col]=price_basket.groupby(by=['CustomerID'])[col].sum()/transactions_per_user['sum']*100
    
    transactions_per_user.drop(columns='sum', inplace=True)
    scaler=joblib.load('models/std_scaler.bin')

    scaled_matrix=scaler.transform(transactions_per_user.to_numpy())

    kmeans_cust=joblib.load('models/kmeans_customers.sav')
    cust_pred=kmeans_cust.predict(scaled_matrix)

    return cust_pred[0]


def prepare_test_data_for_rfm(data):
    data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
    data['TotalPrice']=data['UnitPrice']*data['Quantity']
    data['Date']=data['InvoiceDate'].dt.date

    today_date = dt.datetime(2011,12,11)

    rfm = data.groupby('CustomerID').agg({'InvoiceDate': lambda invoice_date: (today_date - invoice_date.max()).days,
                                        'InvoiceNo': lambda invoice: invoice.nunique(),
                                        'TotalPrice': lambda total_price: total_price.sum()})

    rfm = rfm.reset_index()
    rfm.rename(columns={'InvoiceDate':'Recency',
                        'InvoiceNo':'Frequency',
                        'TotalPrice':'Monetary'}, inplace=True)

    return rfm

def r_score(x, p, d):
    if x<=d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

def f_s_score(x, p, d):
    if x<=d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

def rfm_scoring(rfm_data):
    quantiles_dict={'Recency': {0.25: 19.0, 0.5: 52.0, 0.75: 143.5},
                    'Frequency': {0.25: 1.0, 0.5: 2.0, 0.75: 5.0},
                    'Monetary': {0.25: 307.245, 0.5: 674.45, 0.75: 1661.64}}
    rfm_data['R_Score']=rfm_data['Recency'].apply(lambda x: r_score(x, 'Recency', quantiles_dict))
    rfm_data['F_Score']=rfm_data['Frequency'].apply(lambda x: f_s_score(x, 'Frequency', quantiles_dict))
    rfm_data['M_Score']=rfm_data['Monetary'].apply(lambda x: f_s_score(x, 'Monetary', quantiles_dict))

    rfm_data['RFM_Score']=rfm_data['R_Score'].astype(str)+rfm_data['F_Score'].astype(str)+rfm_data['M_Score'].astype(str)

    return rfm_data['RFM_Score'].values[0]

def evaluate_rfm(rfm_score):
    segment, description, marketing='', '', ''
    if rfm_score=='111':
        segment='Best Customers'
        description='Bought most recently and most often, and spend the most'
        marketing='No price incentives, new products, and loyalty programs'
    elif rfm_score[1]=='1':
        segment='Loyal Customers'
        description='Buys most frequently'
        marketing='Use R and M to further segment'
    elif rfm_score[2]=='1':
        segment='Big Spender'
        description='Spends the most'
        marketing='Market your most expensive products'
    elif rfm_score=='311':
        segment='Almost Lost'
        description='Haven\'t purchased for some time, but purchased frequently and spend the most'
        marketing='Aggressive price incentives'
    elif rfm_score=='411':
        segment='Lost Customer'
        description='Haven\'t purchased for some time, but purchased frequently and spend the most'
        marketing='Aggressive price incentives' 
    elif rfm_score=='444':
        segment='Lost Cheap Customer'
        description='Last purchased long ago, purchased few, and spent little'
        marketing='Don\'t spend too much trying to re-acquire'
    else :
        return 'No Specific Description'
    return segment, description, marketing