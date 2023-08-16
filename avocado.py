# import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title('Avocado Analytics Web App')

# Data Description
st.header('Data Description')
st.markdown('''
            - It is a well known fact that Millenials LOVE Avocado Toast. It's also a well known fact that all Millenials live in their parents basements.
            - Clearly, they aren't buying home because they are buying too much Avocado Toast!
            - But maybe there's hope... if a Millenial could find a city with cheap avocados, they could live out the Millenial American Dream.
            ''')

st.markdown('''
            The table below represents weekly 2018 retail scan data for National retail volume (units) and price. Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. Starting in 2013, the table below reflects an expanded, multi-outlet retail data set. Multi-outlet reporting includes an aggregation of the following channels: grocery, mass, club, drug, dollar and military. The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.
            ''')

# load data
data = pd.read_csv('avocado.csv')

# drop the first column
data = data.drop(['Unnamed: 0'], axis=1)

# get user input on how many rows to display
st.subheader('Sample Data')
rows = st.slider('How many rows of data would you like to see?', 0, 20, 5)

# display data
st.dataframe(data.head(rows))

# top regions visualization
st.header('Top 5 Regions')

# get top 5 regions
top_regions = data.value_counts('region').head(5)

# plot top 5 regions
# set darkgrid style
sns.set_style('darkgrid')
fig, ax = plt.subplots()
ax = sns.barplot(x=top_regions.index, y=top_regions.values)
ax.set_title('Top 5 Regions')
ax.set_xlabel('Region')
# rotate x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Count')
st.pyplot(fig)

# subheader
sns.set_style('darkgrid')
fig1, ax = plt.subplots()
# change figure size
fig1.set_size_inches(10, 5)
ax = sns.histplot(data['Total Volume'], kde=True)
ax.set_title('Total Volume Distribution')
st.pyplot(fig1)

# Machine Learning part
st.header('Machine Learning')

# create my features and the target variable (target = type)
y = data['type']
# select only the numeric columns
X = data[['Total Bags', 'Total Volume', 'AveragePrice']]
# import my ml model
from sklearn.ensemble import RandomForestClassifier
# instantiate my model
rf = RandomForestClassifier()
# fit my model
rf.fit(X, y)

# show progress bar for model training
import time
with st.spinner('Training Model...'):
    time.sleep(5) # wait 5 seconds
    st.success('Done')

# print model accuracy
st.subheader('Model Accuracy')
st.write(f'Model Accuracy: {rf.score(X, y)}')

# get user input
Total_bags = st.number_input('Total Bags: ')
Total_volume = st.number_input('Total Volume: ')
Average_price = st.number_input('Average Price: ')

# create a new dataframe 
X_new = pd.DataFrame({'Total Bags': Total_bags, 
                      'Total Volume': Total_volume, 
                      'AveragePrice': Average_price}, 
                      index=[0])

# print the user input
st.subheader('User Input')
st.dataframe(X_new)

# make predictions and print them 
prediction = rf.predict(X_new)
st.subheader('Prediction')
st.write(f'type: {prediction}')

# print parts of the dataset that are almost equal to the user input
data_comp = data[(data['Total Bags'] >= Total_bags - 20) & 
                 (data['Total Bags'] <= Total_bags + 20) |
                (data['Total Volume'] >= Total_volume - 20) &
                (data['Total Volume'] <= Total_volume + 20) |
                (data['AveragePrice'] >= Average_price - 0.2) &
                (data['AveragePrice'] <= Average_price + 0.2)]
st.subheader('Similar Data Points')
st.dataframe(data_comp)




