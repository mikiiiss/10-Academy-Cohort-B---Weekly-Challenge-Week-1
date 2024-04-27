import streamlit as st
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

### using the following engagement metrics: sessions frequency , the duration of the session , the sessions total traffic (download and upload (bytes))

new_df['session frequency'] = new_df['Avg RTT DL (ms)'] + new_df['Avg RTT UL (ms)']
new_df['session frequency']
new_df['session total traffic'] = new_df['Total DL (Bytes)'] + new_df['Total UL (Bytes)']
new_df['session total traffic']
# Assuming user_engagement_table is your DataFrame
new_df['mean_engagement'] = new_df[['session total traffic', 'session frequency', 'Dur. (ms)']].mean(axis=1)




user_engagement_columns = new_df[['MSISDN/Number', 'session frequency', 'Dur. (ms)', 'Total UL (Bytes)','session total traffic' , 'mean_engagement']]
user_engagement_table = user_engagement_columns.copy()
user_engagement_table

top_10_engagement = user_engagement_table.nlargest(10, 'mean_engagement')
top_10_engagement



# Assuming top_10_engagement contains the top 10 customers based on mean_engagement
plt.figure(figsize=(10, 6))
plt.scatter(top_10_engagement['MSISDN/Number'], top_10_engagement['mean_engagement'], color='skyblue')
plt.xlabel('MSISDN/Number')
plt.ylabel('Mean Engagement')
plt.title('Top 10 Customers by Mean Engagement')
plt.show()

from sklearn.preprocessing import MinMaxScaler
engagement_metrics = user_engagement_table[['session total traffic', 'session frequency', 'Dur. (ms)']]

# Normalize the engagement metrics
scaler = MinMaxScaler()
normalized_engagement = scaler.fit_transform(engagement_metrics)

# Run k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
user_engagement_table['engagement_group'] = kmeans.fit_predict(normalized_engagement)
user_engagement_table

user_engagement_table['engagement_group'].value_counts()
# Merge cluster assignments with the original DataFrame
clustered_data = user_engagement_table.copy()  # Create a copy of the DataFrame
clustered_data['cluster'] = kmeans.labels_  # Add the cluster assignments to the DataFrame

# Group the data by the cluster assignments
cluster_groups = clustered_data.groupby('cluster')

# Compute the minimum, maximum, average, and total non-normalized metrics for each cluster
cluster_summary = cluster_groups.agg({
    'session total traffic': ['min', 'max', 'mean', 'sum'],
    'session frequency': ['min', 'max', 'mean', 'sum'],
    'Dur. (ms)': ['min', 'max', 'mean', 'sum']
    
})
cluster_summary



# Visualize the computed metrics for each cluster using bar charts
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

metrics = ['session total traffic', 'session frequency', 'Dur. (ms)']
titles = ['Session Total Traffic', 'Session Frequency', 'Duration (ms)']

for i, metric in enumerate(metrics):
    ax = axes[i]
    cluster_summary[metric].plot(kind='bar', ax=ax)
    ax.set_title(titles[i])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

# Compute the minimum, maximum, average, and total non-normalized metrics for each cluster
cluster_summary = cluster_groups.agg({
    'session total traffic': ['min', 'max', 'mean', 'sum'],
    'session frequency': ['min', 'max', 'mean', 'sum'],
    'Dur. (ms)': ['min', 'max', 'mean', 'sum']
})

# Display the computed metrics for each cluster
print(cluster_summary)



# Visualize the computed metrics for each cluster using bar charts
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))

metrics = ['session total traffic', 'session frequency', 'Dur. (ms)']
titles = ['Session Total Traffic', 'Session Frequency', 'Duration (ms)']
aggregations = ['min', 'max', 'mean', 'sum']

for i, metric in enumerate(metrics):
    for j, aggregation in enumerate(aggregations):
        ax = axes[i, j]
        cluster_summary[metric][aggregation].plot(kind='bar', ax=ax)
        ax.set_title(f'{titles[i]} - {aggregation}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

# Calculate the total traffic for each application
new_df['Social Media (Bytes)'] = new_df['Social Media DL (Bytes)'] + new_df['Social Media UL (Bytes)']
new_df['YouTube (Bytes)'] = new_df['Youtube DL (Bytes)'] + new_df['Youtube UL (Bytes)']
new_df['Netflix (Bytes)'] = new_df['Netflix DL (Bytes)'] + new_df['Netflix UL (Bytes)']
new_df['Google (Bytes)'] = new_df['Google DL (Bytes)'] + new_df['Google UL (Bytes)']
new_df['Email (Bytes)'] = new_df['Email DL (Bytes)'] + new_df['Email UL (Bytes)']
new_df['Gaming (Bytes)'] = new_df['Gaming DL (Bytes)'] + new_df['Gaming UL (Bytes)']
new_df['Other (Bytes)'] = new_df['Other DL (Bytes)'] + new_df['Other UL (Bytes)']

# Create a new table with the specified columns
app_traffic_summary = new_df[['MSISDN/Number', 
                              'Social Media (Bytes)', 
                              'YouTube (Bytes)', 
                              'Netflix (Bytes)', 
                              'Google (Bytes)', 
                              'Email (Bytes)', 
                              'Gaming (Bytes)', 
                              'Other (Bytes)']]
app_traffic_summary
top_10_social_media_users = new_df.nlargest(10, 'Social Media (Bytes)')
top_10_youtube_users = new_df.nlargest(10, 'YouTube (Bytes)')
top_10_netflix_users = new_df.nlargest(10, 'Netflix (Bytes)')
top_10_google_users = new_df.nlargest(10, 'Google (Bytes)')
top_10_email_users = new_df.nlargest(10, 'Email (Bytes)')
top_10_gaming_users = new_df.nlargest(10, 'Gaming (Bytes)')
top_10_other_users = new_df.nlargest(10, 'Other (Bytes)')

# Get the total usage for each application
total_usage = new_df[['Social Media (Bytes)', 'YouTube (Bytes)', 'Netflix (Bytes)', 'Google (Bytes)', 'Email (Bytes)', 'Gaming (Bytes)', 'Other (Bytes)']].sum()

# Get the top 3 most used applications
top_3_most_used = total_usage.nlargest(3)
print(top_3_most_used)

import matplotlib.pyplot as plt

# Get the total usage for each application
total_usage = new_df[['Social Media (Bytes)', 'YouTube (Bytes)', 'Netflix (Bytes)', 'Google (Bytes)', 'Email (Bytes)', 'Gaming (Bytes)', 'Other (Bytes)']].sum()

# Get the top 3 most used applications
top_3_most_used = total_usage.nlargest(3)

# Create a bar chart for the top 3 most used applications
plt.figure(figsize=(10, 6))
top_3_most_used.plot(kind='bar', color='green')
plt.title('Top 3 Most Used Applications')
plt.xlabel('Application')
plt.ylabel('Total Bytes')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the engagement metrics
engagement_metrics_normalized = scaler.fit_transform(engagement_metrics)

engagement_metrics_normalized

import matplotlib.pyplot as plt

# Initialize an empty list to store the values of the within-cluster sum of squares (WCSS)
wcss = []

# Specify a range of k values to test
k_values = range(1, 11)  # You can adjust the range based on your specific requirements

# Calculate the WCSS for different values of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(engagement_metrics_normalized)
    wcss.append(kmeans.inertia_)

# Plot the WCSS for different values of k
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Assuming X is your data

# Calculate WCSS for different values of k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(engagement_metrics_normalized)
    wcss.append(kmeans.inertia_)

# Calculate slopes
slopes = []
for i in range(1, len(wcss)):
    slope = (wcss[i] - wcss[i-1]) / (i - (i-1))
    slopes.append(slope)

# Plot the slopes
plt.plot(range(2, 11), slopes)
plt.title('Slope of WCSS vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Slope')
plt.show()


# Select relevant engagement metrics for clustering
engagement_metrics = new_df[['session total traffic', 'session frequency', 'Dur. (ms)']]

# Standardize the engagement metrics
scaler = StandardScaler()
engagement_metrics_standardized = scaler.fit_transform(engagement_metrics)

# Initialize a range of k values
k_values = range(1, 11)  # Assuming a range from 1 to 10 for example

# Fit the k-means algorithm for each k and calculate the sum of squared distances
ssd = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(engagement_metrics_standardized)
    ssd.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, ssd, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.show()


X = new_df['session total traffic']
Y = new_df['session frequency']

# Visualize the clusters in a 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('User Engagement Clusters')
plt.xlabel('Social Media (Bytes)')
plt.ylabel('YouTube (Bytes)')
plt.colorbar(label='Cluster')
plt.show()


X = new_df['session total traffic']
Y = new_df['Dur. (ms)']

# Visualize the clusters in a 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('User Engagement Clusters')
plt.xlabel('Social Media (Bytes)')
plt.ylabel('YouTube (Bytes)')
plt.colorbar(label='Cluster')
plt.show()
X = new_df['session frequency']
Y = new_df['Dur. (ms)']

# Visualize the clusters in a 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('User Engagement Clusters')
plt.xlabel('Social Media (Bytes)')
plt.ylabel('YouTube (Bytes)')
plt.colorbar(label='Cluster')
plt.show()
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'session total traffic', 'session frequency', and 'Dur. (ms)' are used as features for clustering
X = new_df['session total traffic']
Y = new_df['session frequency']
Z = new_df['Dur. (ms)']

# Visualize the clusters in a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X, Y, Z, c=cluster_labels, cmap='viridis', s=50, alpha=0.5)

ax.set_title('User Engagement Clusters')
ax.set_xlabel('Session Total Traffic')
ax.set_ylabel('Session Frequency')
ax.set_zlabel('Duration (ms)')
plt.colorbar(scatter, label='Cluster')

plt.show()




