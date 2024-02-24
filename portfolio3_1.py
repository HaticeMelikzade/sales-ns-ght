#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import os 


# In[3]:


df_f = pd.read_csv(r'C:\Users\Melikzade\Downloads\Features_dataset.csv')
df_st = pd.read_csv(r'C:\Users\Melikzade\Downloads\stores_dataset.csv')
df_s = pd.read_csv(r'C:\Users\Melikzade\Downloads\sales_dataset.csv')



# In[24]:


# Merge the Features and Sales datasets
FeatSaleDf = pd.merge(df_f, df_s, on=['Store', 'Date', 'IsHoliday'], how='left')
df_merged = pd.merge(FeatSaleDf, df_st, on = ['Store'], how = 'left')

# Select relevant columns for the 3D plot
selected_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']
df_3d = df_merged[selected_columns]

# Drop rows with missing values
df_3d = df_3d.dropna()

# Standardize the data
scaler = StandardScaler()
df_3d_scaled = pd.DataFrame(scaler.fit_transform(df_3d), columns=df_3d.columns)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_3d_scaled['Temperature'], df_3d_scaled['Fuel_Price'], df_3d_scaled['CPI'], c=df_3d_scaled['Weekly_Sales'], cmap='viridis', s=20)

# Set labels
ax.set_xlabel('Temperature')
ax.set_ylabel('Fuel_Price')
ax.set_zlabel('CPI')
ax.set_title('3D Scatter Plot of Features with Weekly Sales')

# Show the plot
plt.show()


# In[5]:


df_merged.columns


# In[29]:


df_merged['Date'] = pd.to_datetime(df_merged['Date'], format='%d/%m/%Y')
df_2011 = df_merged[df_merged['Date'].dt.year == 2011]

fig, axes = plt.subplots(nrows=5, figsize=(16, 16), sharex=True)

color_mapping = {'A': 'purple', 'B': 'dodgerblue', 'C': 'orange'} 

for typ in df_2011['Type'].unique().tolist():
    filtered_data = df_2011[df_2011['Type'] == typ]
    for ax, column, ylabel in zip(axes, ['Temperature', 'Fuel_Price', 'Unemployment', 'Weekly_Sales', 'CPI'],
                                  ['Temperature', 'Fuel Price', 'Unemployment', 'Weekly Sales', 'Consumer Price Index']):
        ax.scatter(filtered_data['Date'], filtered_data[column],
                   color=color_mapping[typ], alpha=0.6, label=f'Type {typ}')
        ax.set_ylabel(ylabel, fontsize=16)

for ax in axes:
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
axes[-1].set_xlabel('Date', fontsize=16)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(df_2011['Type'].unique()), fontsize=14)

fig.subplots_adjust(top=0.95)  # Adjust top to make room for title
fig.suptitle('Timeseries for a single year (2011)', fontsize=20)

plt.show()


# In[36]:


grouped_data = df_merged.groupby(['Type', 'Date']).mean()

fig, axes = plt.subplots(nrows=5, figsize=(16, 16), sharex=True)

for typ in df_merged['Type'].unique():
    axes[0].plot(grouped_data.loc[typ].index, grouped_data.loc[typ]['Temperature'], label=f'Type {typ}')
    axes[1].plot(grouped_data.loc[typ].index, grouped_data.loc[typ]['Fuel_Price'], label=f'Type {typ}')
    axes[2].plot(grouped_data.loc[typ].index, grouped_data.loc[typ]['Unemployment'], label=f'Type {typ}')
    axes[3].plot(grouped_data.loc[typ].index, grouped_data.loc[typ]['Weekly_Sales'], label=f'Type {typ}')
    axes[4].plot(grouped_data.loc[typ].index, grouped_data.loc[typ]['CPI'], label=f'Type {typ}')

ylabels = ['Temperature', 'Fuel Price', 'Unemployment', 'Weekly Sales', 'CPI']
for ax, label in zip(axes, ylabels):
    ax.set_ylabel(label, fontsize=16)
    ax.grid(True, alpha=0.5)

axes[-1].set_xlabel('Date', fontsize=16)

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=16)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(df_merged['Type'].unique()), fontsize=14)

fig.subplots_adjust(top=0.95)
fig.suptitle('Average over all stores and departments', fontsize=20)
plt.show()


# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt

sales_summary = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].sum().unstack()

sales_summary.fillna(0, inplace=True)

plt.figure(figsize=(20, 10))
sns.heatmap(sales_summary, annot=False)

plt.figure(figsize=(20, 10))
sns.heatmap(sales_summary, annot=False, cmap='plasma')  # Remove the space after 'plasma'

plt.title('Total Sales for Each Department Across All Stores')
plt.xlabel('Department', fontsize=14)
plt.ylabel('Store', fontsize=14)
plt.show()


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate total sales by store and department
totalSales_byStore = df_merged.groupby('Store')['Weekly_Sales'].sum()
totalSales_byDept = df_merged.groupby('Dept')['Weekly_Sales'].sum()

# Rank stores and departments based on total sales
rankedStores = totalSales_byStore.sort_values(ascending=False)
rankedDepts = totalSales_byDept.sort_values(ascending=False)

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 20))

# Plot total sales by store
sns.barplot(x=rankedStores.index, y=rankedStores.values, ax=axes[0], palette='viridis')
axes[0].set_title('Total Sales per Store')
axes[0].set_xlabel('Store Number')
axes[0].set_ylabel('Total Sales ($)')
axes[0].tick_params(axis='x', rotation=45)

# Plot total sales by department
sns.barplot(x=rankedDepts.index, y=rankedDepts.values, ax=axes[1], palette='viridis')
axes[1].set_title('Total Sales per Department')
axes[1].set_xlabel('Department Number')
axes[1].set_ylabel('Total Sales ($)')
axes[1].tick_params(axis='x', rotation=90)

# Set y-axis limit
axes[0].set_ylim([0, 5e8])
axes[1].set_ylim([0, 5e8])

plt.show()


# In[75]:


print('Highest Performing Store = Store #', rankedStores.index[0])
print('Highest Performing Department = Department # ', int(rankedDepts.index[0]))


# In[77]:


# Get the holidays version of the dataset
holidaysDF = df_merged[df_merged['IsHoliday']==True]

fig, ax =plt.subplots()

# Plotting each markdown series as a line plot on the same axes with different colors
holidaysDF.plot.line(x="Date", y="MarkDown1", ax=ax, color='red', label='MarkDown1')
holidaysDF.plot.line(x="Date", y="MarkDown2", ax=ax, color='blue', label='MarkDown2')
holidaysDF.plot.line(x="Date", y="MarkDown3", ax=ax, color='green', label='MarkDown3')
holidaysDF.plot.line(x="Date", y="MarkDown4", ax=ax, color='purple', label='MarkDown4')
holidaysDF.plot.line(x="Date", y="MarkDown5", ax=ax, color='orange', label='MarkDown5')

ax.set_title('Markdown Over The Holidays')
ax.set_xlabel('Date')
ax.set_ylabel('Markdown Value')

# Add a legend Below the plot
ax.legend(loc='upper center', bbox_to_anchor= (0.5, -0.15), ncol=3)

#adjust layout to make room for the legend
plt.subplots_adjust(bottom=0.2)

plt.show()


# In[87]:


# List of Markdown columns
markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Filter rows where all Markdown columns are greater than 0
holidaysMarkdownDF = holidaysDF[holidaysDF[markdown_columns].gt(0).any(axis=1)].copy()

# Extract rows where all Markdown columns are equal to 0
holidaysNoMarkDownDF = holidaysDF[holidaysDF[markdown_columns].isna().all(axis=1)].copy()

# Extract the Year from the date for both dataframes
holidaysMarkdownDF['Year'] = holidaysMarkdownDF['Date'].dt.year
holidaysNoMarkDownDF['Year'] = holidaysNoMarkDownDF['Date'].dt.year

# Aggregate Weekly_Sales by year for both dataframes
sales_markdown = holidaysMarkdownDF.groupby('Year')['Weekly_Sales'].mean()
sales_no_markdown = holidaysNoMarkDownDF.groupby('Year')['Weekly_Sales'].mean()

# Combine the data into a single dataframe for plotting
combined_sales = pd.DataFrame({'With Markdown': sales_markdown, 'Without Markdown': sales_no_markdown})


# Plotting
ax = combined_sales.plot(kind='bar', figsize=(12, 6), rot=0)
plt.title('Yearly Sales Comparison during Holidays')
plt.xlabel('Year')
plt.ylabel('Mean Weekly Sales')
plt.xtricks(rotation=0)
plt.show()


# In[91]:


weekly_mean_sales = df_merged.groupby('Date')['Weekly_Sales'].mean()
weekly_mean_markdown = df_merged.groupby('Date')[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean()


# Replace NaN with 0 in both sales and markdown
weekly_mean_sales.replace(np.nan, 0, inplace=True)
weekly_mean_markdown.replace(np.nan, 0, inplace=True)

# Create a figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot for sales
sns.lineplot(x=weekly_mean_sales.index, y=weekly_mean_sales.values, label='Weekly Sales', color='teal', linewidth=2, ax=ax1)

# Colors for each markdown
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Create a second y-axis for markdown
ax2 = ax1.twinx()

# Plot for each markdown
for i, col in enumerate(weekly_mean_markdown.columns):
    ax2.plot(weekly_mean_markdown.index, weekly_mean_markdown[col], label=f'Markdown{i+1}', color=colors[i])

# Set labels for both y-axes
ax1.set_ylabel('Weekly Sales', color='teal')
ax2.set_ylabel('Markdowns', color='black')

# Draw vertical lines for the start date of each markdown
for col in weekly_mean_markdown.columns:
    start_date = weekly_mean_markdown[col].first_valid_index()
    plt.axvline(x=start_date, color='black', linestyle='--')

    # Add annotation for the start date
    plt.annotate(f'Start of {col}', 
                 xy=(start_date, 0), 
                 xycoords='data', 
                 xytext=(-90, 30), 
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 color='black')

# Set title and labels
plt.title('Total Sales and Markdowns Over Time')
plt.xlabel('Date')
plt.xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2013-12-01'))

# Show legend for both y-axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()


# In[100]:


# Extract year and month
df_merged['Year'] = df_merged['Date'].dt.year
df_merged['Month'] = df_merged['Date'].dt.month_name()

# Group by year and month, and calculate the mean of sales and markdowns
grouped = df_merged.groupby(['Year', 'Month'])
monthly_data = grouped[['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean().reset_index()

# Add a column for average markdown
monthly_data['AvgMarkDown'] = monthly_data[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean(axis=1)

# Define the correct order for the months
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'Month' to a categorical type with the defined order
monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=months_order, ordered=True)

# Sort the data
monthly_data.sort_values(by=['Year', 'Month'], inplace=True)

# Create a figure with two subplots, one above the other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))

# Plotting the sales data on the first subplot using line plot
sns.lineplot(data=monthly_data, x='Month', y='Weekly_Sales', hue='Year', palette='inferno', marker='o', ax=ax1)
ax1.set_title('Mean Monthly Sales for Each Year')
ax1.set_xlabel('Month')
ax1.set_ylabel('Weekly Sales')
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', rotation=45)

# Plotting the markdown data on the second subplot using line plot
sns.lineplot(data=monthly_data, x='Month', y='AvgMarkDown', hue='Year', palette='viridis', marker='o', ax=ax2)
ax2.set_title('Average Monthly Markdowns for Each Year')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Markdown')
ax2.legend(loc='upper left')
ax2.tick_params(axis='x', rotation=45)

# Adjust the layout
plt.tight_layout()
plt.show()


# In[115]:


# Add a column for average markdown
df_merged['AvgMarkDown'] = df_merged[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean(axis=1)

# Define the correct order for the months
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'Month' to a categorical type with the defined order
df_merged['Month'] = pd.Categorical(df_merged['Month'], categories=months_order, ordered=True)

# Group the data by 'Year' and 'Month'
grouped = df_merged.groupby(['Year', 'Month'])

# Calculate the mean sales and markdowns for each group
mean_sales = grouped['Weekly_Sales'].mean().reset_index()
mean_markdowns = grouped['AvgMarkDown'].mean().reset_index()

# Plot the average monthly sales and markdowns for each year
plt.figure(figsize=(12, 6))

# Create a barplot for sales
plt.subplot(1, 2, 1)
sns.barplot(x='Month', y='Weekly_Sales', hue='Year', data=mean_sales, palette=palette)
plt.title('Average Monthly Sales for Each Year')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Create a barplot for markdowns
plt.subplot(1, 2, 2)
sns.barplot(x='Month', y='AvgMarkDown', hue='Year', data=mean_markdowns, palette=palette)
plt.title('Average Monthly Markdowns for Each Year')
plt.xlabel('Month')
plt.ylabel('Average Markdown')
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Adjust the layout
plt.tight_layout()
plt.show()


# In[116]:


# Find top 10 and bottom 10 stores based on weekly sales
top_stores = df_merged.groupby('Store')['Weekly_Sales'].sum().nlargest(10).index
bottom_stores = df_merged.groupby('Store')['Weekly_Sales'].sum().nsmallest(10).index

# Filter the data for top and bottom stores
top_stores_data = df_merged[df_merged['Store'].isin(top_stores)]
bottom_stores_data = df_merged[df_merged['Store'].isin(bottom_stores)]

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plotting top 10 stores
sns.barplot(x='Store', y='Weekly_Sales', data=top_stores_data, ax=axes[0], palette='viridis')
axes[0].set_title('Top 10 Stores Based on Weekly Sales')
axes[0].set_ylabel('Total Sales ($)')
axes[0].set_xlabel('Store Number')
axes[0].tick_params(axis='x', rotation=0)

# Plotting bottom 10 stores
sns.barplot(x='Store', y='Weekly_Sales', data=bottom_stores_data, ax=axes[1], palette='viridis')
axes[1].set_title('Bottom 10 Stores Based on Weekly Sales')
axes[1].set_ylabel('Total Sales ($)')
axes[1].set_xlabel('Store Number')
axes[1].tick_params(axis='x', rotation=0)

# Adjust layout
plt.tight_layout()
plt.show()


# In[117]:


# Select relevant columns
correlation_columns = ['Fuel_Price', 'CPI', 'Temperature', 'Weekly_Sales']

# Create a DataFrame with selected columns
correlation_data = df_merged[correlation_columns]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap: Fuel_Price, CPI, Temperature, Weekly_Sales')
plt.show()


# In[118]:


# Group by year and calculate total sales
total_sales_by_year = df_merged.groupby('Year')['Weekly_Sales'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Weekly_Sales', data=total_sales_by_year, palette='viridis')
plt.title('Total Sales by Year')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.show()


# In[119]:


# Group by year and type, and calculate total sales
total_sales_by_year_type = df_merged.groupby(['Year', 'Type'])['Weekly_Sales'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Year', y='Weekly_Sales', hue='Type', data=total_sales_by_year_type, palette='viridis')
plt.title('Total Sales by Year and Store Type')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.legend(title='Store Type')
plt.show()



# In[120]:


# Filter data for the years 2011 and 2012
df_2011_2012 = df_merged[df_merged['Year'].isin([2011, 2012])]

# Select relevant columns for correlation analysis
correlation_columns = ['Unemployment', 'CPI', 'Fuel_Price', 'Temperature', 'Weekly_Sales']

# Calculate the correlation matrix
correlation_matrix = df_2011_2012[correlation_columns].corr()

# Create a heatmap for visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix (2011-2012)')
plt.show()


# In[129]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming your DataFrame is named 'df_merged' with columns 'Date' and 'Weekly_Sales'
df_merged['Date'] = df_merged.index
df_merged['Date'] = pd.to_datetime(df_merged['Date'])
df_merged.set_index('Date', inplace=True)

# Interpolate missing values
df_merged['Weekly_Sales'].interpolate(inplace=True)

# Create a time series DataFrame
ts_df = df_merged.dropna()

# Plot the autocorrelation function (ACF) for Weekly_Sales
plt.figure(figsize=(12, 6))
plot_acf(ts_df['Weekly_Sales'], lags=25)
plt.title("Autocorrelation Plot for Weekly_Sales")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.show()

# Plot the partial autocorrelation function (ACF) for Weekly_Sales
plt.figure(figsize=(12, 6))
plot_pacf(ts_df['Weekly_Sales'], lags=25)
plt.title("Partial Autocorrelation Plot for Weekly_Sales")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




