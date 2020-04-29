#!/usr/bin/env python
# coding: utf-8

# COVID-19 Web APP

import streamlit as st
import requests

import plotly.express as px
import plotly 

import pandas as pd
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

#### Data ####

url_c = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_d = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_r = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
#time_series_covid19_confirmed_global.csv
#time_series_covid19_deaths_global.csv
#time_series_covid19_recovered_global.csv

@st.cache
def loading_covid_data():
    #COVID-19 API
    r = requests.get('https://api.covid19api.com/summary')
    j = r.json() 
    #CSSEGISandData
    df_c = pd.read_csv(url_c,delimiter = ',')
    df_d = pd.read_csv(url_d,delimiter = ',')
    df_r = pd.read_csv(url_r,delimiter = ',')
    return df_c,df_d,df_r,j

df_c,df_d,df_r,j = loading_covid_data()

df_c.rename(columns={'Lat':'lat', 'Long':'lon'}, inplace=True)
df_d.rename(columns={'Lat':'lat', 'Long':'lon'}, inplace=True)
df_r.rename(columns={'Lat':'lat', 'Long':'lon'}, inplace=True)

####### APP #######

st.title(' COVID-19 APP 1.0')

#### App pages ####

st.sidebar.markdown(
"""

### Data

""")

page = st.sidebar.radio('Select data to display   ', ['Global and country comparison data', 'One country data and predictions'])

st.sidebar.markdown(
"""

### About

This web app gets **actual COVID-19 data** from [CSSE at Johns Hopkins University data repository](https://github.com/CSSEGISandData/COVID-19) 
and from an [API](https://covid19api.com/) related to this data developed by [Kyle Redelinghuys](https://twitter.com/ksredelinghuys).

On this sidebar you can **select the type of data** you want to consult (**Global** or per **country** data)
. In the one country data option there is also showed a **time series forecasting based on neural networks (LTSM)**.

**All plots are interactive** allowing zooming, scrolling, **hovering on data** for additional information and **image saving**.

Source code can be found at [GitHub](https://github.com/Sampayob).

**Author**: Sergio Sampayo

""")

if page == 'Global and country comparison data':
#### Global Data ####

    st.markdown(
    """
    ## Global resume
    """)

    ### actual global info tables ###

    data_json = j['Global']
    global_info = pd.DataFrame.from_dict(data_json, orient='index', columns=['Global data']).T

    global_info.rename(columns={'NewConfirmed':'New confirmed','TotalConfirmed':'Total confirmed',
                              'NewDeaths':'New deaths','TotalDeaths':'Total deaths',
                                'NewRecovered':'New recovered','TotalRecovered':'Total recovered'}, inplace=True)
    
    ### Top 10 confirmed cases countries ###
    
    df_global = pd.DataFrame(columns=['Country','New confirmed', 'Total confirmed',
               'New deaths', 'Total deaths', 'New recovered', 'Total recovered'])

    for i in range(0,len(j['Countries'])):
        name = j['Countries'][i]['Country'] 
        nc = j['Countries'][i]['NewConfirmed'] 
        tc = j['Countries'][i]['TotalConfirmed'] 
        nd = j['Countries'][i]['NewDeaths'] 
        td = j['Countries'][i]['TotalDeaths'] 
        nr = j['Countries'][i]['NewRecovered'] 
        tr = j['Countries'][i]['TotalRecovered'] 

        df_global = df_global.append({'Country': name,'New confirmed' : nc, 'Total confirmed' :tc,
                   'New deaths':nd, 'Total deaths' :td, 'New recovered' :nr, 'Total recovered' :tr}, ignore_index=True)
        
    top10 = df_global.sort_values(by='Total confirmed', ascending=False).head(10)
    top10 = pd.concat([top10, global_info], axis = 0).sort_values(by='Total confirmed', ascending=False)
    top10['Country'][0] = 'Global'
    top10[' '] = list(range(0,11))
    st.table(top10.set_index(' '))
    
    check_global_df = st.checkbox('Show global dataset')
    
    if check_global_df:
        
        st.dataframe(df_global.sort_values(by='Total confirmed', ascending=False))
    
    check_global_bubble = st.checkbox('Show global confirmed cases bubble chart')
    
    if check_global_bubble:

        ### Scatterplot ###

        ## Select info ##

        df_selected = st.radio('Select data to display', ['Confirmed cases', 'Deaths', 'Recovered'])

        if df_selected == 'Confirmed cases':
            df = df_c

        elif df_selected == 'Deaths':
            df = df_d

        elif df_selected == 'Recovered':
            df = df_r

        df_scatter = df

        df_scatter[df_scatter.columns[-1]].replace({-1: 0}, inplace=True)

        df_scatter = df_scatter.sort_values(by=df_scatter.columns[-1], ascending=False)

        df_scatter = df_scatter.groupby(["Country/Region"])[df_scatter.columns[-1]].sum().reset_index()

        fig = px.scatter(df_scatter, x="Country/Region" , y = df_scatter.columns[-1] , color = "Country/Region" ,  
                         size= df_scatter.columns[-1], size_max = 60, hover_name = 'Country/Region')
        fig.update_yaxes(title_text='Date: '+df.columns[-1])

        st.plotly_chart(fig, use_container_width=True)

    #### Multiple countries plot #####
    
    st.markdown(
    """    
    ## Country comparison
    
    ### Line chart
    """)
    
    country_list = df_c['Country/Region'].unique().tolist()
    
    country = st.multiselect('Select two or more countries', country_list )
    
    dates_list = df_c.T.reset_index()['index'][4:].tolist()
    date_selected = st.selectbox('Select start date', dates_list )

    data_oc_select = st.radio('Select data to display ', ['Confirmed cases', 'Deaths', 'Recovered'])
    
    if len(country) > 0:
        
        df_oc_c = []
        df_oc_d = []
        df_oc_r = []

        df_multicountry = pd.DataFrame(columns=[])

        for n in country:
            country_data = df_c[df_c['Country/Region'] == n]

            if len(country_data)>1:
                country_data1 = pd.DataFrame(df_c[df_c['Country/Region'] == n].sum(), columns=['sum']).T
                country_data2 = pd.DataFrame(df_d[df_d['Country/Region'] == n].sum(), columns=['sum']).T
                country_data3 = pd.DataFrame(df_r[df_r['Country/Region'] == n].sum(), columns=['sum']).T
            else:
                country_data1 = pd.DataFrame(df_c[df_c['Country/Region'] == n])
                country_data2 = pd.DataFrame(df_d[df_d['Country/Region'] == n])
                country_data3 = pd.DataFrame(df_r[df_r['Country/Region'] == n])

            x1 = country_data1.loc[:,date_selected:].T
            x2 = country_data2.loc[:,date_selected:].T
            x3 = country_data3.loc[:,date_selected:].T

            df_oc_c = x1[x1.columns[0]].tolist() 
            df_oc_d = x2[x2.columns[0]].tolist()
            df_oc_r = x3[x3.columns[0]].tolist()

            #Fixing date format because there is a streamlit axis representation problem if not

            dates = x1.reset_index()['index'].tolist()

            new_dates_column = []

            for i in dates:
                if len(i) == 7:
                    new_dates_column.append(i[-2:]+'/'+i[0:4]) 
                elif len(i) == 8:
                    new_dates_column.append(i[-2:]+'/'+i[0:5]) 
                elif len(i) == 6:
                    new_dates_column.append(i[-2:]+'/'+i[0:3]) 
            dates = new_dates_column

            if data_oc_select == 'Confirmed cases':
                df_oc_c = np.asarray(df_oc_c)
                df_multicountry[n] = df_oc_c
                df_multicountry['date'] = dates
            elif data_oc_select == 'Deaths':
                df_oc_r = np.asarray(df_oc_r)
                df_multicountry[n] = df_oc_r
                df_multicountry['date'] = dates
            elif data_oc_select == 'Recovered':
                df_oc_d = np.asarray(df_oc_d)
                df_multicountry[n] = df_oc_d
                df_multicountry['date'] = dates

            df_multicountry["date"] = pd.to_datetime(df_multicountry['date'],  yearfirst = True)

        #plot

        st.line_chart(df_multicountry.rename(columns={'date':'index'}).set_index('index'))

elif page == 'One country data and predictions':
    #### Country Data ####  

    ###General data

    st.markdown(
    """
    ## Country data
    """)

    country_list = df_c['Country/Region'].unique().tolist()
    country = st.selectbox('Select country data to display', country_list )

    dates_list = df_c.T.reset_index()['index'][4:].tolist()
    date_selected = st.selectbox('Select start date', dates_list )
    data_oc_select = st.radio('Select data to display', ['All','Confirmed cases', 'Deaths', 'Recovered'])
    
    st.info('To show a **growth factor plot** and a **prediction plot** select the "Confirmed cases", "Deaths" or "Recovered" option.')

    df_oc_c = []
    df_oc_d = []
    df_oc_r = []

    country_data = df_c[df_c['Country/Region'] == country]

    if len(country_data)>1:
        country_data1 = pd.DataFrame(df_c[df_c['Country/Region'] == country].sum(), columns=['sum']).T
        country_data2 = pd.DataFrame(df_d[df_d['Country/Region'] == country].sum(), columns=['sum']).T
        country_data3 = pd.DataFrame(df_r[df_r['Country/Region'] == country].sum(), columns=['sum']).T
    else:
        country_data1 = pd.DataFrame(df_c[df_c['Country/Region'] == country])
        country_data2 = pd.DataFrame(df_d[df_d['Country/Region'] == country])
        country_data3 = pd.DataFrame(df_r[df_r['Country/Region'] == country])

    x1 = country_data1.loc[:,date_selected:].T
    x2 = country_data2.loc[:,date_selected:].T
    x3 = country_data3.loc[:,date_selected:].T

    df_oc_c = x1[x1.columns[0]].tolist() 
    df_oc_d = x2[x2.columns[0]].tolist()
    df_oc_r = x3[x3.columns[0]].tolist()

    st.subheader(country)   
    
    if data_oc_select == 'All':
        st.write('Confirmed cases, Deaths and Recovered')
    else: 
        st.write(data_oc_select)

    #Fixing date format because there is a streamlit axis representation problem if not

    dates = x1.reset_index()['index'].tolist()

    new_dates_column = []

    for i in dates:
        if len(i) == 7:
            new_dates_column.append(i[-2:]+'/'+i[0:4]) 
        elif len(i) == 8:
            new_dates_column.append(i[-2:]+'/'+i[0:5]) 
        elif len(i) == 6:
            new_dates_column.append(i[-2:]+'/'+i[0:3]) 
    dates = new_dates_column

    if data_oc_select == 'Confirmed cases':
        df_oc_final = pd.DataFrame({'date': dates,data_oc_select: df_oc_c})
    elif data_oc_select == 'Deaths':
        df_oc_final = pd.DataFrame({'date': dates,data_oc_select: df_oc_d})
    elif data_oc_select == 'Recovered':
        df_oc_final = pd.DataFrame({'date': dates,data_oc_select: df_oc_r})
    elif data_oc_select == 'All':   
        df_oc_final = pd.DataFrame({'date': dates,'Confirmed cases': df_oc_c, 'Deaths': df_oc_d, 'Recovered': df_oc_r})

    df_oc_final["date"] = pd.to_datetime(df_oc_final['date'],  yearfirst = True)

    #plot
    st.line_chart(df_oc_final.rename(columns={'date':'index'}).set_index('index'))

    ### Growth factor
    
    df_oc_final_gf = df_oc_final.copy()
    
    if data_oc_select != 'All':
        st.write(data_oc_select + ' growth factor')

    if data_oc_select != 'All':
        
        gf = []
                                 
        for i in range(0,len(df_oc_final_gf[data_oc_select])):
            if i == 0:
                gf.append(1)
            else:
                number=(df_oc_final_gf[data_oc_select][i])/(df_oc_final_gf[data_oc_select][i-1])
                if np.isinf(number) == True or np.isnan(number) == True:
                    number= 1
                    gf.append(number)
                else:
                    gf.append(number)
    
        df_oc_final_gf[data_oc_select] = gf
             
        st.line_chart(df_oc_final_gf.rename(columns={'date':'index'}).set_index('index'))
        
    ### LTSM Time series forecasting
    
    if data_oc_select != 'All':

        df_oc_final = df_oc_final.rename(columns={'date':'Date', data_oc_select:'Confirmed'}).set_index('Date')

        df = df_oc_final

        # Train and Test split

        train = df[:len(df)-5]
        test = df[len(df)-5:]

        #Normalize data

        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)

        #timesseries generator object and model

        n_input = 5 #number of steps
        n_features = 1 # number of features you want to predict (univariate time series = 1)

        generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features))) # optimum neurons = 3/2*hiddenLayers*n_input 
        model.add(Dense(75, activation = 'relu')) #a second layer improve accuracy and model robustness
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit_generator(generator,epochs=20)

        # Forecast

        test_prediction = []

        # last n points from training set
        first_eval_batch = scaled_train[-n_input:]
        current_batch = first_eval_batch.reshape(1,n_input,n_features)

        #forecasting next 7 days
        for i in range (len(test)+7): 
            current_pred = model.predict(current_batch)[0]
            test_prediction.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis = 1)

        true_prediction  = scaler.inverse_transform(test_prediction)

        ## Generating index for prediction df

        time_series_array = test.index
        for k in range(0,7):
            time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

        #forecast dataframe and plot

        df_forecast = pd.DataFrame(index=time_series_array)
        df_forecast.loc[:,"Confirmed"] = test['Confirmed']
        df_forecast.loc[:,"Predicted"] = true_prediction[:,0]

        #plot1

        st.write(data_oc_select + ' prediction for next 7 days')
         
        st.line_chart(df_forecast)

        #plot2
        df_oc_final['Predicted'] = df_oc_final[df_oc_final.columns[0]]
        df1 = df_oc_final.rename(columns={df_oc_final.columns[0]: "Confirmed"})[:len(df_oc_final)-5]
        df2 = df_forecast.iloc[:,0:2]
        df_final_ltsm = pd.concat([df1, df2], axis=0)

        st.line_chart(df_final_ltsm)

        #Accuracy
        MAPE = np.mean(np.abs(np.array(df_forecast["Confirmed"][:5]) - np.array(df_forecast["Predicted"][:5]))/np.array(df_forecast["Confirmed"][:5]))
        
        st.write("**MAPE **:" + str(round((MAPE*100),2)) + " %")
        st.write("**Model Accuracy**: " + str(round((1-MAPE),2)))
        st.write('*Please take into account that a good amount of data is needed for an accurate prediction. Moreover, the neural network algorithm could be improved*')
      



