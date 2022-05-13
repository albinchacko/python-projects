# This code uses datascientist.csv file and generates full pledged data visualization #


# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import wordcloud as wc
import sweetviz


# Data Import
uc_data = pd.read_csv('archive/DataScientist.csv')


# Data Overview

print(uc_data.head())
print(uc_data.tail())
print(uc_data.info())
print(uc_data.describe())
print(uc_data.shape)
print(uc_data.columns)

# This function call generates a full pledged data overview dashboard
my_report  = sweetviz.analyze([uc_data, 'Dataset Overview']) #/ Takes dataset as the argument
my_report.show_html(filepath='unclean_overview.html', open_browser=True, layout='widescreen', scale=None)

# Data Cleaning

# Dropping unnecessary columns
drop_cols = ['index', 'Unnamed: 0']
uc_data.drop(drop_cols, axis=1, inplace=True)

# Replacing irrelevant values with relevant values
uc_data = uc_data.replace('-1', np.nan)
uc_data = uc_data.replace(-1, np.nan)
uc_data = uc_data.replace(-1.0, np.nan)
uc_data['Easy Apply'] = uc_data['Easy Apply'].fillna(False).astype('bool')
uc_data['Revenue'] = uc_data['Revenue'].replace('Unknown / Non-Applicable', np.nan)

# Extracting company name from Rating values
uc_data['Company Name'],_=uc_data['Company Name'].str.split('\n', 1).str

# Removing unnecessary words from Salary estimate column
uc_data['Salary Estimate'],_=uc_data['Salary Estimate'].str.split('(', 1).str

# Extracting state/country from location column
uc_data['Location'], uc_data['State/Country'] = uc_data['Location'].str.split(',').str

# Extracting department from Job title
for val in range(uc_data.shape[0]):
    if ',' in uc_data.loc[val, 'Job Title']:
        uc_data.loc[val, 'Job Title'], _ = uc_data.loc[val, 'Job Title'].split(',', 1)

    # Converting abrevations
    uc_data.loc[val, 'Job Title'] = uc_data.loc[val, 'Job Title'].replace('Sr.', 'Senior')
    uc_data.loc[val, 'Job Title'] = uc_data.loc[val, 'Job Title'].replace('Jr.', 'Junior')



# Data Organization

# Extracting Starting and Highest Salaries from Salary Estimate column
uc_data['Starting Salary'], uc_data['Highest Salary'] = uc_data['Salary Estimate'].str.split('-').str
uc_data['Starting Salary'] = uc_data['Starting Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0)
uc_data['Highest Salary'] = uc_data['Highest Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0)

# Index of salary value stored in an array
sal_loc = []
for i in range(uc_data.shape[0]):
    sal_value = uc_data.loc[i,"Salary Estimate"]
    if "Per Hour" in sal_value:
        sal_loc.append(i)

# Converting Starting salary values 
for val in range(uc_data.shape[0]):    
    if val in sal_loc:
        uc_data.loc[val, 'Starting Salary'] = int(uc_data.loc[val, 'Starting Salary']) * 40 * 52 # Assuming 40 hours/week and 52 weeks/year
    else:
        uc_data.loc[val, 'Starting Salary'] = int(uc_data.loc[val, 'Starting Salary']) * 1000

# Converting Highest salary values
for val in range(uc_data.shape[0]):
    if val in sal_loc:
        uc_data.loc[val, 'Highest Salary'], _ = uc_data.loc[val, 'Highest Salary'].split('Per')
        uc_data.loc[val, 'Highest Salary'] = uc_data.loc[val, 'Highest Salary'].strip()        
        uc_data.loc[val, 'Highest Salary'] = int(uc_data.loc[val, 'Highest Salary']) * 40 * 52 # Assuming 40 hours/week and 52 weeks/year
    else:
        uc_data.loc[val, 'Highest Salary'] = int(uc_data.loc[val, 'Highest Salary']) * 1000

# Dropping the Salary estimate column
uc_data.drop(['Salary Estimate'], axis=1, inplace=True)
uc_data['Mean Salary'] = uc_data[['Starting Salary', 'Highest Salary']].mean(axis = 1)


# Extracting average revenue from the Revenue column
for val in range(uc_data.shape[0]):
    uc_data.loc[val, 'Revenue'] = str(uc_data.loc[val, 'Revenue']).strip('(USD)').replace('$', '')

    if (('million' in uc_data.loc[val, 'Revenue']) and ('billion' not in uc_data.loc[val, 'Revenue'])):
        if ('Less than' in uc_data.loc[val, 'Revenue']):
            min_rev = 0.0
            max_rev = 1000000.0
            uc_data.loc[val, 'Average Revenue'] = (max_rev + min_rev) / 2 # Calculates mean of the salary values
        else:
            min_rev, max_rev = (uc_data.loc[val, 'Revenue'].replace('million','')).split('to')
            min_rev = float(min_rev) * 1000000
            max_rev = float(max_rev) * 1000000
            uc_data.loc[val, 'Average Revenue'] = (max_rev + min_rev) / 2 # Calculates mean of the salary values
        
    elif (('billion' in uc_data.loc[val, 'Revenue']) and ('million' not in uc_data.loc[val, 'Revenue'])):
        if ('+' in uc_data.loc[val, 'Revenue']):
            min_rev = 10000000000.0
            max_rev = 11000000000.0
            uc_data.loc[val, 'Average Revenue'] = (max_rev + min_rev) / 2 # Calculates mean of the salary values
        else:
            min_rev, max_rev = (uc_data.loc[val, 'Revenue'].replace('billion','')).split('to')
            min_rev = float(min_rev) * 1000000000
            max_rev = float(max_rev) * 1000000000
            uc_data.loc[val, 'Average Revenue'] = (max_rev + min_rev) / 2 # Calculates mean of the salary values      

    elif (('billion' in uc_data.loc[val, 'Revenue']) and ('million' in uc_data.loc[val, 'Revenue'])):
        min_rev, max_rev = (uc_data.loc[val, 'Revenue'].replace('million','').replace('billion','')).split('to')
        min_rev = float(min_rev) * 1000000
        max_rev = float(max_rev) * 1000000000
        uc_data.loc[val, 'Average Revenue'] = (max_rev + min_rev) / 2 # Calculates mean of the salary values

# Dropping the Revenue column
uc_data.drop(['Revenue'], axis=1, inplace=True)

# The cleaned data is stored in a new variable
c_data = uc_data

# Data Overview after data cleaning
my_report  = sweetviz.analyze([c_data, 'Dataset_Overview'])
my_report.show_html(filepath='clean_overview.html', open_browser=True, layout='widescreen', scale=None)

# Data Visualization

# 1. Salary Visualization

# --> Distribution of Salary

# Salary distribution of Data Science job roles
sns.set(style='white', palette='muted', color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True) # Plots subplots of two plots
sns.despine(left=True) # Removes edges
sns.histplot(c_data['Starting Salary'], color='b', ax=axes[0]) # Draws histogram in one side of plot
sns.histplot(c_data['Highest Salary'], color='r', ax=axes[1])  # Draws histogram in another side of plot
plt.tight_layout()

# Distribution of salary compared to the company's ratings
fig = px.scatter(c_data, x=c_data['Rating'], y=c_data['Mean Salary']) # Scatter plot function
fig.update_layout(title='Distribution of Mean salary and Company ratings')
fig.show()

# --> Salary Comparison  

# a.) Top 15 Data Science job roles with good average salary

# Top 15 Job roles in the dataset according to the Salary data
sal_df = c_data.groupby('Job Title')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary')) # Bar chart function
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 Job roles in the dataset according to the Salary data', barmode='group')
fig.show()

# b.) Salary comparison of Data Science core job roles

# Dataframes for only core Data Science job roles
da_data = c_data[c_data['Job Title'].str.contains('Data Analyst', regex=True) == True].reset_index()
ds_data = c_data[c_data['Job Title'].str.contains('Data Scientist', regex=True) == True].reset_index()
de_data = c_data[c_data['Job Title'].str.contains('Data Engineer', regex=True) == True].reset_index()
ba_data = c_data[c_data['Job Title'].str.contains('Business Analyst', regex=True) == True].reset_index()
ml_data = c_data[c_data['Job Title'].str.contains('Machine Learning Engineer|Machine Learning|ML Engineer', regex=True) == True].reset_index()
# Mean value of salary from core Data Science job roles
da_mean = da_data[['Mean Salary']].nlargest(len(da_data['Mean Salary']),"Mean Salary").mean()
ds_mean = ds_data[['Mean Salary']].nlargest(len(ds_data['Mean Salary']),"Mean Salary").mean()
de_mean = de_data[['Mean Salary']].nlargest(len(de_data['Mean Salary']),"Mean Salary").mean()
ba_mean = ba_data[['Mean Salary']].nlargest(len(ba_data['Mean Salary']),"Mean Salary").mean()
ml_mean = ml_data[['Mean Salary']].nlargest(len(ml_data['Mean Salary']),"Mean Salary").mean()
# Dictionary of respective mean values
mean_df = {'Data Analyst': da_mean[0], 'Data Scientist': ds_mean[0], 'Data Engineer': de_mean[0], 'Business Analyst': ba_mean[0], 'ML Engineer': ml_mean[0]}

# Dictionary converted into dataframe
mean_df = pd.DataFrame.from_dict([mean_df])

# Plot of mean salary of Data Science job roles
plt.figure(figsize=(10, 7))
chart = sns.barplot(    
    data=mean_df,    
    palette='Set1'
)
chart=chart.set_xticklabels(
    chart.get_xticklabels(),
    fontweight='light',
)
plt.title('Mean Salary of Data Science jobs', fontsize=18)
plt.xlabel('Job Role', fontsize=15)
plt.ylabel('Mean Salary', fontsize=15)

# c) Top 15 well paying Companies

# Top 15 well paying Companies in the dataset according to the Salary data
sal_df = c_data.groupby('Company Name')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary'))
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 well paying Companies', barmode='group')
fig.show()

# d) Top 15 well paid working Sectors

# Top 15 well paid working sectors in the dataset according to the Salary data
sal_df = c_data.groupby('Sector')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary'))
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 well paid sectors', barmode='relative')
fig.show()

# e) Top 15 well paid cities

# Top 15 well paid cities in the dataset according to the Salary data
sal_df = c_data.groupby('Location')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary'))
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 well paid cities', barmode='relative')
fig.show()

# f) Top 15 well paying Companies (currently hiring)

# Top 15 well paying Companies (currently hiring) in the dataset according to the Salary data
sal_df = c_data[c_data['Easy Apply']==True].groupby('Company Name')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary'))
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 well paying Companies (currently hiring)', barmode='group')
fig.show()

# g) Top 15 well paying Industries

# Top 15 well paying Industries in the dataset according to the Salary data
sal_df = c_data.groupby('Industry')[['Starting Salary','Highest Salary']].mean().nlargest(15,['Starting Salary','Highest Salary']).round(2).head(15)
fig = go.Figure()
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Starting Salary'], name='Starting Salary'))
fig.add_trace(go.Bar(x=sal_df.index, y=sal_df['Highest Salary'], name='Highest Salary'))
fig.update_layout(title='Top 15 well paid Industries', barmode='relative')
fig.show()

# 2. Job Title Visualization  

# a) 15 Most popular job titles

# Plot of 15 Most popular job titles
plt.figure(figsize=(10, 7))
title_df = c_data['Job Title'].value_counts()[0:15]
chart = sns.barplot(x=title_df.index, y=title_df, palette = 'spring')
chart = chart.set_xticklabels(
    chart.get_xticklabels(),    
    horizontalalignment='right',
    rotation=60,
)
plt.xlabel('Job Title',fontsize=15)
plt.ylabel('Number of Jobs',fontsize=15)
plt.title('15 Most popular job titles',fontsize=18)

# b) Most popular Data science core job roles

# Dictionary of respective counts
length_df = {'Data Analyst': len(da_data), 'Data Scientist': len(ds_data), 'Data Engineer': len(de_data), 'Business Analyst': len(ba_data), 'ML Engineer': len(ml_data)}

# Dictionary converted into dataframe
length_df = pd.DataFrame.from_dict([length_df])

# Plot of popular Data science core job roles
plt.figure(figsize=(10, 7))
chart = sns.barplot(    
    data=length_df,    
    palette='Set1'
)
plt.title('Popular Data science core job roles', fontsize=18)
plt.xlabel('Job Role', fontsize=15)
plt.ylabel('Number of Jobs', fontsize=15)

# c) Word cloud of Job titles in the dataset

# Plot of word cloud of different Job titles in the dataset
plt.figure(figsize=(12, 12))
word_plot = wc.WordCloud(background_color='white', width=450, height= 300)
text = c_data['Job Title']
word_plot.generate(str(' '.join(text)))
plt.imshow(word_plot)
plt.axis("off")
plt.show()

# 3. Revenue Visualization

# a) Distribution of revenue according to company founding year

# Distribution of revenue according to company founding year
fig = px.scatter(c_data, x=c_data['Founded'], y=c_data['Average Revenue'])
fig.update_layout(title='Distribution of Average revenue by company founding year')
fig.show()

# b) Top 15 Highest Revenue sectors

# Plot of Top 15 Highest Revenue sectors
rev_df = c_data.groupby('Sector')[['Average Revenue']].mean().nlargest(15,['Average Revenue']).round(2).head(15)
fig = px.bar(rev_df, x=rev_df.index, y=rev_df['Average Revenue'], color='Average Revenue')
fig.update_layout(title='Top 15 Highest Revenue sectors', barmode='group')
fig.show()

# c) Top 15 Highest Revenue industries

# Plot of Top 15 Highest Revenue industries
rev_df = c_data.groupby('Industry')[['Average Revenue']].mean().nlargest(15,['Average Revenue']).round(2).head(15)
fig = px.bar(rev_df, x=rev_df.index, y=rev_df['Average Revenue'], color='Average Revenue')
fig.update_layout(title='Top 15 Highest Revenue industries', barmode='relative')
fig.show()

# d) Revenue distribution by type of ownership

# Plot of revenue by type of ownership
rev_df = c_data.groupby('Type of ownership')[['Average Revenue']].mean()
fig = px.pie(rev_df, values=rev_df['Average Revenue'], names=rev_df.index, color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title = 'Revenue distribution by type of ownership')
fig.show()

# 4. Company wise Visualization

# a) Top 15 companies with data science jobs

# Plot of top 15 companies with data science jobs
plt.figure(figsize=(10, 7))
cmp_df = c_data['Company Name'].value_counts().nlargest(15)
chart = sns.barplot(x=cmp_df.index, y=cmp_df.values, palette = 'summer')
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.xlabel('Company Name',fontsize=15)
plt.ylabel('Number of Jobs',fontsize=15)
plt.title('Top 15 companies with data science jobs',fontsize=18)

# b) Top competitor companies of Top 10 Companies

# A dataframe is created for the top 10 companies
cmp_df = c_data['Company Name'].value_counts().nlargest(10)
cmp_name = str()
# String with the names of top companies formed
for i in range(len(cmp_df.index.values)):
    cmp_name = cmp_name + '|' + cmp_df.index.values[i]
cmp_name = cmp_name.lstrip('|') 
# Top companies checked and company dataframe created
top_comp = c_data[c_data['Company Name'].str.contains(cmp_name, regex=True)].reset_index()

cmp_list = [] # To store list of competitors
for i in range(len(top_comp['Competitors'])):    
    for j in range(len(str(top_comp['Competitors'][i]).split(','))):        
        if(str(top_comp['Competitors'][i]) != 'nan'):
            cmp_list.append(str(top_comp['Competitors'][i]).split(',')[j].strip(' '))

cmp_list = pd.DataFrame(cmp_list) # Dataframe with competitor value counts formed
cmp_list = cmp_list.rename(columns={0 : 'Competitor'}) 

# Plot of the top competitors
plt.figure(figsize=(10, 7))
c_df = cmp_list['Competitor'].value_counts()
chart = sns.barplot(x=c_df.index, y=c_df.values, palette = 'autumn')
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.xlabel('Company Name',fontsize=15)
plt.ylabel('Number of Jobs',fontsize=15)
plt.title('Top competitor companies of Top 10 Companies',fontsize=18)

# c) Word cloud of Top competitors

# Plot of competitor word cloud
plt.figure(figsize=(12, 12))
word_plot = wc.WordCloud(background_color='lightyellow', width=450, height= 300)
text = cmp_list['Competitor']
word_plot.generate(str(' '.join(text)))
plt.imshow(word_plot)
plt.axis("off")
plt.show()

# d) Employee numbers in Top 15 Companies

# Plot of employee numbers
plt.figure(figsize=(10, 7))
size_df = top_comp['Size'].value_counts()
chart = sns.barplot(x=size_df.index, y=size_df.values, palette='Set2')
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.xlabel('Size range',fontsize=15)
plt.ylabel('Number of Jobs',fontsize=15)
plt.title('Employee numbers in Top 15 Companies',fontsize=18)

# e) Revenue distribution of Top 15 companies

# Pie Plot of revenue
top_df = top_comp.groupby('Company Name')[['Average Revenue']].mean()
fig = px.pie(top_df, values=top_df['Average Revenue'], names=top_df.index)
fig.update_layout(title = 'Revenue distribution of Top 15 companies')
fig.show()

# f) Word cloud of Top 15 company's headquarters

# Plot of headquarters
plt.figure(figsize=(12, 12))
word_plot = wc.WordCloud(background_color = 'lightpink', width=450, height= 300)
text = top_comp['Headquarters']
word_plot.generate(str(' '.join(text)))
plt.imshow(word_plot)
plt.axis("off")
plt.show()


# 5. Location Visualization

# a) Top 20 Locations where Data science jobs are available

# Plot of the locations
plt.figure(figsize=(10, 7))
loc_df = c_data['Location'].value_counts().nlargest(20)
chart = sns.barplot(x=loc_df.index, y=loc_df.values, palette = 'winter')
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.xlabel('Location',fontsize=15)
plt.ylabel('Number of Jobs',fontsize=15)
plt.title('Top 20 Locations for Data science jobs',fontsize=18)

# b) States in USA having Data Science jobs

# Map of states plotted
loc_data = c_data[c_data['State/Country'].str.contains('United Kingdom') == False]
fig = px.choropleth(
    locationmode='USA-states',
    locations=loc_data['State/Country'].str.strip().values,        
    labels=loc_data['State/Country'].str.strip().values,
    color=loc_data['State/Country'].str.strip().values,    
    scope='usa',    
)
fig.show()

# c) Heatmap of States in USA having Data Science jobs

# Heatmap plotted
fig = go.Figure(data=go.Choropleth(
    locations=loc_data['State/Country'].value_counts().index.str.strip(),
    z = loc_data['State/Country'].value_counts().values,
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "No. of Jobs",
))
fig.update_layout(
    title_text = 'Heatmap of USA States with No. of Data Science jobs',
    geo_scope = 'usa',
)
fig.show()

# 6. Data science requirements visualization

# a) Data science programming languages requirements

# Assumed 'Python', 'R', 'Java', 'C++', 'Linux', 'Javascript' as common languages
plang = ['Python', 'R', 'Java', 'C++', 'Linux', 'Javascript']
lang_freq = dict()
# Frequency of occurence in each cell
for lang in plang:
    count = 0
    for desc in c_data['Job Description']:
        if lang in desc:
            count = count + 1
        lang_freq[lang] = count
# Dataframe for languages count
plang_df = pd.DataFrame.from_dict(lang_freq.items())
plang_df.columns = ['Language Name', 'Count']

# Plot of bar and pie charts
sns.set(style='white', palette='muted', color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
chart = sns.barplot(x=plang_df['Language Name'], y=plang_df['Count'], color='lightgreen', ax=axes[0])
explode = [0.01 for x in range(len(plang_df))]
axes[1].pie(plang_df['Count'], explode = explode, labels=plang_df['Language Name'], autopct='%1.1f%%', startangle=0)
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.title('Data science Programming languages requirements',fontsize=20)
plt.tight_layout()

# b) Data science qualification requirements

# Assumed 'PhD', 'MD', 'Pharm.D', 'Doctorate', 'Postdoc', 'Masters', 'MSc', 'Bachelors', 'BSc' as some common qualifications
edu = ['PhD', 'MD', 'Pharm.D', 'Doctorate', 'Postdoc', 'Masters', 'MSc', 'Bachelors', 'BSc']
edu_freq = dict()
# Frequency of occurence in each cell
for ed in edu:
    count = 0
    for desc in c_data['Job Description']:
        if ed in desc:
            count = count + 1
        edu_freq[ed] = count
# Dataframe for qualification counts
edu_df = pd.DataFrame.from_dict(edu_freq.items())
edu_df.columns = ['Qualification Name', 'Count']

# Plot of bar and pie charts
sns.set(style='white', palette='muted', color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
chart = sns.barplot(x=edu_df['Qualification Name'], y=edu_df['Count'], color='yellow', ax=axes[0])
explode = [0.025 for x in range(len(edu_df))]
axes[1].pie(edu_df['Count'], explode = explode, labels=edu_df['Qualification Name'], autopct='%1.1f%%', startangle=0)
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.title('Data science Qualification requirements',fontsize=20)
plt.tight_layout()

# c) Data science Subject requirements

# Assumed 'Mathematics', 'Computer Science', 'Statistics', 'Biology', 'Chemistry', 'Biochemistry' as common subjects
sub = ['Mathematics', 'Computer Science', 'Statistics', 'Biology', 'Chemistry', 'Biochemistry']
sub_freq = dict()
# Frequency of occurence in each cell
for sb in sub:
    count = 0
    for desc in c_data['Job Description']:
        if sb in desc:
            count = count + 1
        sub_freq[sb] = count
# Dataframe for subject counts
sub_df = pd.DataFrame.from_dict(sub_freq.items())
sub_df.columns = ['Subject Name', 'Count']

# Plot of bar and pie charts
sns.set(style='white', palette='muted', color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
chart = sns.barplot(x=sub_df['Subject Name'], y=sub_df['Count'], color='violet', ax=axes[0])
explode = [0.025 for x in range(len(sub_df))]
axes[1].pie(sub_df['Count'], explode = explode, labels=sub_df['Subject Name'], autopct='%1.1f%%', startangle=0)
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.title('Data science Subject requirements',fontsize=20)
plt.show()

# d) Data science tools requirements

# Assuming 'SQL', 'NoSQL', 'Tableau', 'Excel', 'Hadoop', 'Spark', 'SAS', 'Stata', 'Hive', 'Scala', 'AWS', 'GCP', 'Azure', 'Google Cloud' as common tools
tools = ['SQL', 'NoSQL', 'Tableau', 'Excel', 'Hadoop', 'Spark', 'SAS', 'Stata', 'Hive', 'Scala', 'AWS', 'GCP', 'Azure', 'Google Cloud']
tools_freq = dict()
# Frequency of occurence in each cell
for tool in tools:
    count = 0
    for desc in c_data['Job Description']:
        if tool in desc:
            count = count + 1
        tools_freq[tool] = count
# Dataframe for tools count
tools_df = pd.DataFrame.from_dict(tools_freq.items())
tools_df.columns = ['Tool Name', 'Count']

# Plot of bar and pie charts
sns.set(style='white', palette='muted', color_codes=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
chart = sns.barplot(x=tools_df['Tool Name'], y=tools_df['Count'], color='indigo', ax=axes[0])
explode = [0.025 for x in range(len(tools_df))]
axes[1].pie(tools_df['Count'], explode = explode, labels=tools_df['Tool Name'], autopct='%1.1f%%', startangle=0)
chart = chart.set_xticklabels(
    chart.get_xticklabels(),
    horizontalalignment='right',
    rotation=60,
)
plt.title('Data science Tools requirements',fontsize=20)
plt.tight_layout()



# % References used:

# 1. YASHVI PATEL. Data Analyst jobs visualization. Available: https://www.kaggle.com/code/yashvi/data-analyst-jobs-visualization#Size-of-Employees-Vs-No-of-Companies

# 2. TAHA07. Data Scientists Jobs. Available: https://www.kaggle.com/code/taha07/data-scientists-jobs-analysis-visualization

# 3. SAMRUDDHI MHATRE. Analysis of Data Scientist Jobs. Available: https://www.kaggle.com/code/samruddhim/analysis-of-data-scientist-jobs#2.-Statistics

# 4. ROHIT SAHOO. Data Scientist Job Analysis. Available: https://www.kaggle.com/code/rohitsahoo/data-scientist-jobs-analysis#Data-Visualization
