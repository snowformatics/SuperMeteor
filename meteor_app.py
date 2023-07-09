import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import datetime
import streamlit.components.v1 as components
import calendar

st.set_page_config(layout="wide")

st.title('Meteor Scattering')

# txt = """DIAMOND D-130NJ DISCONE-ANTENNE with RTL-SDR Blog V3 R860 RTL2832U.
#          Recorded with SpectrumLab and a periodic action (screenshot every 90 seconds).
#          Objects were segmented with a Mask-C-RNN model.
#          Special thanks to Wilhelm, who provided the SpectrumLab settings and the segmenation model."""
#st.write('Setup', txt)
#st.subheader(txt)


col1, col2 = st.columns([2,2], gap="medium")



DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            #'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
DATA_URL = ("https://raw.githubusercontent.com/snowformatics/SuperMeteor/master/supermeteor/out_all.csv")
DATA_URL_top5 = ("https://raw.githubusercontent.com/snowformatics/SuperMeteor/master/supermeteor/image_out.csv")

#DATA_URL = ("test.csv")

today = date.today()

f = '%Y-%m-%d'
f2 = '%H%M%S'
f3 = f + f2
pd.set_option('display.max_columns', None)


def get_top5_meteors(data):
    data['h'] = pd.to_numeric(data['h'], errors='coerce')
    data['w'] = pd.to_numeric(data['w'], errors='coerce')

    # Group the DataFrame by 'date'
    grouped_df = data.groupby('date')

    # Extract the two largest 'w' and 'h' values per date
    largest_objects_per_day = grouped_df.apply(lambda x: x.nlargest(5, ['w', 'h'])).reset_index(drop=True)
    return largest_objects_per_day


@st.cache_data(ttl=datetime.timedelta(hours=24))
def load_data(nrows):
    data = pd.read_csv(DATA_URL, delimiter='\t', dtype={'time':str})



    data['date'] = data["timestemp"].str.slice(stop=10)
    data['date'] = pd.to_datetime(data['date'], format=f)
    data['time'] = pd.to_datetime(data['time'], format=f2)
    data['date'] = data['date'].astype(str)
    data['time'] = data['time'].astype(str)
    data['time'] = data['time'].str.slice(10)

    data[DATE_COLUMN] = pd.to_datetime(data['date'].astype(str) +
                                          data['time'].astype(str))



    return data


def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


#data_load_state = st.text('Loading data...')
data = load_data(100000)
largest_objects_per_day = get_top5_meteors(data)

#print (largest_objects_per_day)

with col1:
    date_input1 = st.date_input(
        "Choose a date",
        datetime.date(today.year, today.month, today.day-1))
    # Filter by date
    st.subheader('Number of meteors by hour')

    data2 = data[data['date'] == str(date_input1)]
    df_stats2 = data2[['h', 'w']].describe(include="all").transpose()
    if st.checkbox('Show raw data', key=2):
        st.subheader('Raw data')
        st.write(data2)
    if st.checkbox('Show statistics', key=3):
        st.subheader('Statistics')
        st.write(df_stats2)
    hist_values1 = np.histogram(data2[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values1)

    #data_images = largest_objects_per_day[largest_objects_per_day['date'] == str(date_input1)]
    # https://drive.google.com/file/d/18moiUN7JsWHVjLfTCGEnyghYLKFz3PTF/view?usp=drive_link

    #st.write(data_images)
    #print (data_images)



    # st.subheader('Number of meteors by month')
    # hist_values2 = np.histogram(data[DATE_COLUMN].dt.month, bins=12, range=(0, 12))[0]
    # st.bar_chart(hist_values2)


with col2:
    st.subheader('Top 5 Meteor images')
    top_meteors = pd.read_csv(DATA_URL_top5, header=None)
    top_meteors.sort_values()
    top_meteors.to_csv('out.csv')
    print (top_meteors)
    #data2['id_all'] = data2['image_file'].str[:25]
    #top_meteors['id_top5'] = top_meteors[0].str.split('/').str[4].str[:25]
    #print (top_meteors['id_top5'])

    #top_meteor_list = data2[data2['id_all'].isin(top_meteors['id_top5'])]['id_all'].unique().tolist()
#     top_meteors.columns = ['url']
#     top_meteor_list = []
#     for index, row in data2.iterrows():
#         id_all = row['image_file'][0:25]
#         for index1, row1 in top_meteors.iterrows():
#             id_top5 = row1['url'].split('/')[4][0:25]
#             if id_all == id_top5:
#                 if id_top5 not in top_meteor_list:
#                     top_meteor_list.append(id_top5)
#                     st.image(row1['url'],width=600)
    #print (top_meteor_list)


    # st.image(
    #     "https://i
    #     k.imagekit.io/nb4gbrqqe/GRAVES-XYmVV_230607172530_1_org.jpg",
    #     width=600,
    # )
    # st.image(
    #     "https://ik.imagekit.io/nb4gbrqqe/GRAVES-XYmVV_230607172530_1_org.jpg",
    #     width=600,
    # )
#     date_input2 = st.date_input(
#         "Choose a date",
#         datetime.date(today.year, today.month, today.day), key=1)
#     # st.write('Your birthday is:', date_input)
#     # print (date_input)
#     # Filter by date
#     # data2 = data[data['meteor start date (yyyy/mm/dd)'] == "2023-01-03"]
#     data3 = data[data['meteor start date (yyyy/mm/dd)'] == str(date_input2)]
#     df_stats3 = data3[['peak meteor signal (float. db)', 'peak meteor noise (float. db)',
#                        'peak meteor signal to noise ratio (float. db)', 'peak meteor frequency (integer. hz)',
#                        'meteor duration (float. seconds)']].describe(include="all").transpose()
#
#     if st.checkbox('Show raw data'):
#         st.subheader('Raw data')
#         st.write(data3)
#     if st.checkbox('Show statistics', key=4):
#         st.subheader('Statistics')
#         st.write(df_stats3)
#
#     st.subheader('Number of meteors by hour')
#     hist_values2 = np.histogram(data3[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
#     st.bar_chart(hist_values2)

ChangeWidgetFontSize('Show raw data', '22px')
ChangeWidgetFontSize('Show statistics', '22px')
ChangeWidgetFontSize('Choose a date', '22px')
#ChangeWidgetFontSize('Setup', '18px')

