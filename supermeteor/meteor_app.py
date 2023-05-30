import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import datetime
import streamlit.components.v1 as components


st.set_page_config(layout="wide")

st.title('Meteor Scattering')


col1, col2 = st.columns([2,2], gap="medium")



DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            #'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
DATA_URL = ("https://raw.githubusercontent.com/snowformatics/SuperMeteor/master/supermeteor/table.csv")
today = date.today()

f = '%Y%m%d'
f2 = '%H%M%S'
f3 = f + f2

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, delimiter='\t')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    #print (data)
    data['meteor start date (yyyy/mm/dd)'] = pd.to_datetime(data['meteor start date (yyyy/mm/dd)'], format=f)
    data['meteor start time (hh:mm:ss)'] = pd.to_datetime(data['meteor start time (hh:mm:ss)'], format=f2)
    data['meteor start date (yyyy/mm/dd)'] = data['meteor start date (yyyy/mm/dd)'].astype(str)
    data['meteor start time (hh:mm:ss)'] = data['meteor start time (hh:mm:ss)'].astype(str)
    data['meteor start time (hh:mm:ss)'] = data['meteor start time (hh:mm:ss)'].str.slice(10)
    data[DATE_COLUMN] = pd.to_datetime(data['meteor start date (yyyy/mm/dd)'].astype(str) +
                                          data['meteor start time (hh:mm:ss)'].astype(str))
    data.to_csv('out.csv')
    return data

def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

data_load_state = st.text('Loading data...')
data = load_data(10000)





with col1:
    date_input1 = st.date_input(
        "Choose a date",
        datetime.date(today.year, today.month,today.day))
    #st.write('Your birthday is:', date_input)
    #print (date_input)
    # Filter by date
    #data2 = data[data['meteor start date (yyyy/mm/dd)'] == "2023-01-03"]
    data2 = data[data['meteor start date (yyyy/mm/dd)'] == str(date_input1)]
    df_stats2 = data2[['peak meteor signal (float. db)', 'peak meteor noise (float. db)',
                     'peak meteor signal to noise ratio (float. db)', 'peak meteor frequency (integer. hz)',
                     'meteor duration (float. seconds)']].describe(include="all").transpose()

    if st.checkbox('Show raw data', key=2):
        st.subheader('Raw data')
        st.write(data2)
    if st.checkbox('Show statistics', key=3):
        st.subheader('Statistics')
        st.write(df_stats2)

    st.subheader('Number of meteors by hour')
    hist_values1 = np.histogram(data2[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values1)
    # st.image(
    #     "https://ik.imagekit.io/nb4gbrqqe/event20230225_133517_27.jpg",
    #     width=400,  # Manually Adjust the width of the image as per requirement
    # )

with col2:
    date_input2 = st.date_input(
        "Choose a date",
        datetime.date(today.year, today.month, today.day), key=1)
    # st.write('Your birthday is:', date_input)
    # print (date_input)
    # Filter by date
    # data2 = data[data['meteor start date (yyyy/mm/dd)'] == "2023-01-03"]
    data3 = data[data['meteor start date (yyyy/mm/dd)'] == str(date_input2)]
    df_stats3 = data3[['peak meteor signal (float. db)', 'peak meteor noise (float. db)',
                       'peak meteor signal to noise ratio (float. db)', 'peak meteor frequency (integer. hz)',
                       'meteor duration (float. seconds)']].describe(include="all").transpose()

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data3)
    if st.checkbox('Show statistics', key=4):
        st.subheader('Statistics')
        st.write(df_stats3)

    st.subheader('Number of meteors by hour')
    hist_values2 = np.histogram(data3[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values2)

ChangeWidgetFontSize('Show raw data', '22px')
ChangeWidgetFontSize('Show statistics', '22px')
ChangeWidgetFontSize('Choose a date', '22px')

