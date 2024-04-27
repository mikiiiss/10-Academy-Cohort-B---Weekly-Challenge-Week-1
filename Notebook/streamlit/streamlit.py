import streamlit as st
# my_notebook.ipynb

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



# Load the saved CSV file into a new DataFrame
new_df = pd.read_csv('C:\\Users\\Nole\\Desktop\\Teleco\\Notebook\\Cleaned_data_csv\\cleaned_data.csv')

# Now you can use the new_df DataFrame for further analysis in pandas
new_df['session frequency'] = new_df['Avg RTT DL (ms)'] + new_df['Avg RTT UL (ms)']
new_df['session frequency']

new_df['session total traffic'] = new_df['Total DL (Bytes)'] + new_df['Total UL (Bytes)']
new_df['session total traffic']

# Assuming user_engagement_table is your DataFrame
new_df['mean_engagement'] = new_df[['session total traffic', 'session frequency', 'Dur. (ms)']].mean(axis=1)


user_engagement_columns = new_df[['MSISDN/Number', 'session frequency', 'Dur. (ms)', 'Total UL (Bytes)','session total traffic' , 'mean_engagement']]
user_engagement_table = user_engagement_columns.copy()
user_engagement_table

# Create the Streamlit app
st.title("User Engagement Analysis")

# Display the user_engagement_table
st.subheader("User Engagement Table")
st.dataframe(user_engagement_table)

# Plot the data
st.subheader("Plots")

# Session Frequency vs Duration
fig, ax = plt.subplots()
ax.scatter(user_engagement_table['session frequency'], user_engagement_table['Dur. (ms)'])
ax.set_xlabel('Session Frequency')
ax.set_ylabel('Duration (ms)')
st.pyplot(fig)

# Session Total Traffic vs Total UL Bytes
fig, ax = plt.subplots()
ax.scatter(user_engagement_table['session total traffic'], user_engagement_table['Total UL (Bytes)'])
ax.set_xlabel('Session Total Traffic')
ax.set_ylabel('Total UL Bytes')
st.pyplot(fig)

# Mean Engagement
fig, ax = plt.subplots()
ax.hist(user_engagement_table['mean_engagement'], bins=20)
ax.set_xlabel('Mean Engagement')
ax.set_ylabel('Count')
st.pyplot(fig)

# import streamlit as st
# import pandas as pd

# # Get the total usage for each application
# total_usage = new_df[['Social Media (Bytes)', 'YouTube (Bytes)', 'Netflix (Bytes)', 'Google (Bytes)', 'Email (Bytes)', 'Gaming (Bytes)', 'Other (Bytes)']].sum()

# # Get the top 3 most used applications
# top_3_most_used = total_usage.nlargest(3)

# # Create a bar chart for the top 3 most used applications
# st.title('Top 3 Most Used Applications')

# # Create a container for the chart
# with st.container():
#     # Create the bar chart
#     chart_data = pd.DataFrame(top_3_most_used)
#     chart_data = chart_data.reset_index()
#     chart_data.columns = ['Application', 'Total Bytes']
#     st.bar_chart(chart_data, x='Application', y='Total Bytes')

# # Add additional information
# st.write('This chart shows the top 3 most used applications based on total bytes consumed.')
