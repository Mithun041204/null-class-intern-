#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[23]:


data=pd.read_csv(r"C:\Users\SUNIL\Downloads\heart-disease.csv")
data


# In[25]:


print(data.head())


# In[26]:


data.tail()


# In[27]:


data.head()


# In[28]:


data.describe()


# In[29]:


data.info()


# In[32]:


import numpy as np
data.isnull().sum()


# In[35]:


get_ipython().system('pip install matplotlib')


# In[36]:


import matplotlib.pyplot as plt


# In[39]:


get_ipython().system('pip install seaborn')


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


get_ipython().system('pip install plotly')


# In[62]:


import matplotlib.pyplot as plt
import numpy as np

xpoints=np.array([0,6])
ypoints=np.array([0,250])


# In[69]:



import matplotlib.pyplot as plt


xpoints=np.array([1,2,6,8])
ypoints=np.array([3,8,1,10])


# In[84]:


plt.plot(xpoints,ypoints)
plt.show()


# In[85]:


y=np.array([35,25,25,15])


# In[90]:


plt.pie(y)
plt.show()


# In[93]:


import pandas as pd
apps_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")
reviews_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\User Reviews.csv")


# In[94]:


apps_df.head()


# In[95]:


reviews_df.head()


# In[100]:


apps_df=apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)


# In[113]:


apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)
apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)
apps_df.dtypes


# In[116]:


merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')
merged_df.head()


# In[117]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in  size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)
apps_df


# In[120]:


apps_df['Log_Installs']=np.log1p(apps_df['Installs'])
apps_df['Reviews']=apps_df['Reviews'].astype(int)
apps_df['Log_Reviews']=np.log1p(apps_df['Reviews'])
apps_df.dtypes


# In[123]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)


# In[125]:


apps_df['Revenue']=apps_df['Price']*apps_df['Installs']


# In[126]:


apps_df


# !pip install nltk

# In[6]:


get_ipython().system('pip install nltk')


# In[15]:


nltk.download


# In[17]:


import nltk


# In[2]:


get_ipython().system('pip install textBlob')


# In[9]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
sia=SentimentIntensityAnalyzer()


# In[11]:


review="This app is amazing! I love the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[12]:


review="This app very bad! I hate the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[13]:


review="This app is okay."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[17]:


import pandas as pd
reviews_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\User Reviews.csv")
reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
reviews_df.head()


# In[3]:


import pandas as pd
apps_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")
apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')
apps_df['Year']=apps_df['Last Updated'].dt.year
apps_df.head()


# In[1]:


import plotly.express as px
fig=px.bar(x=["A","B","C"],y=[1,3,2],title="Sample Bar Chart")
fig.show()


# In[11]:


get_ipython().system('pip install plotly')


# In[ ]:


fig.write_html("interactive_plot.html")


# In[42]:


pip install plotly --upgrade


# In[19]:


import os
html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
    


# In[21]:


plot_containers=""


# In[ ]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
def save_plot_as_html(fig,filename,insight):
    global plot_containers
    filepath=os.path.join(html_files_path,filename)
    html_content=pio.to_html(fig,full_html=False,include_plotlyjs='inline')
    plot_containers += f"""
    <div class = "plot-container" id="{filename}" onclick ="openPlot('{filename}')">
        <div class="plot">{html_content}</div> 
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath,full_html=False,include_plotlyjs='inline')


# In[27]:


plot_width=400
plot_height=300
plot_bg_colour='black'
text_colour='white'
title_font={'size':16}
axis_font={'size':12}


# In[ ]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
apps_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")
category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

pio.write_html(fig1,file="Category Graph 1.html",auto_open=True)


# In[ ]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
apps_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    labels={'x':'Category','y':'Count'},
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10, r=10, t=30, b=10)
)

pio.write_html(fig1,file="Type Graph 2.html",auto_open=True)


# In[ ]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
apps_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

pio.write_html(fig3,file="Rating Graph 3.html",auto_open=True)


# In[10]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
reviews_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\User Reviews.csv")
sentiment_counts=reviews_df['Sentiment'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

pio.write_html(fig4,file="Sentiment Graph 4.html",auto_open=True)


# In[9]:


print(reviews_df.columns)


# In[21]:


print(apps_df['Installs'].dtype)
print(apps_df['Installs'].head(10))


# In[23]:


print(apps_df['Installs'].head(20))


# In[24]:


print(apps_df['Installs'].isnull().sum())


# In[25]:



apps_df['Installs'] = apps_df['Installs'].fillna('0').astype(str)

apps_df['Installs'] = apps_df['Installs'].str.replace(r'\D', '', regex=True)

apps_df['Installs'] = apps_df['Installs'].replace('', '0').astype(int)


# In[26]:


print(apps_df['Installs'].dtype)
print(apps_df['Installs'].head(10))


# In[19]:


import pandas as pd
import plotly.express as px

# Load the dataset
apps_df = pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")

# Step-by-step cleaning of the 'Installs' column
apps_df['Installs'] = apps_df['Installs'].fillna('0').astype(str)          # Replace NaN with '0' and convert to string
apps_df['Installs'] = apps_df['Installs'].str.replace(r'\D', '', regex=True)  # Remove non-digit characters
apps_df['Installs'] = apps_df['Installs'].replace('', '0').astype(int)     # Replace empty strings with '0' and convert to int

# Group by 'Category' and get the top 10 categories with the highest installs
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)

# Plot the bar chart
fig5 = px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)

fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save the chart as an HTML file
fig5.write_html("Category Graph 5.html", auto_open=True)


# In[21]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
reviews_df=pd.read_csv(r"C:\Users\SUNIL\Downloads\User Reviews.csv")

# Load the data
try:
    apps_df = pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")

except FileNotFoundError:
    print("Error: File not found at the specified path.")
    exit()

# Debugging: Print column names and a sample of the dataset
print("Columns in the dataset:", reviews_df.columns)
print("Sample data:\n", reviews_df.head())

# Ensure 'Genres' column is available
if 'Genres' in apps_df.columns:
    # Split genres and count occurrences
    genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)

    # Plot the bar chart
    fig7 = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        labels={'x': 'Genre', 'y': 'Count'},
        title='Top Genres',
        color_discrete_sequence=px.colors.sequential.OrRd,
        width=400,
        height=300
    )

    # Update layout for black background
    fig7.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size': 16},
        xaxis=dict(title_font={'size': 12}, tickfont={'color': 'white'}),
        yaxis=dict(title_font={'size': 12}, tickfont={'color': 'white'}),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Save the chart as an HTML file
    pio.write_html(fig7, file="Sentiment Graph 7.html", auto_open=True)

else:
    print("The 'Genres' column is missing from the dataset.")


# In[ ]:


import pandas as pd
import plotly.express as px

apps_df = pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")

apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')

apps_df = apps_df.dropna(subset=['Last Updated'])

fig8 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',  
    color='Type',
    title='Impact of Last Update on Rating',  
    width=800,
    height=400
)


fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

fig8.write_html("LastUpdatedScatter.html", auto_open=True)


# In[25]:


import pandas as pd
import plotly.express as px

apps_df = pd.read_csv(r"C:\Users\SUNIL\Downloads\Play Store Data.csv")

apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')

apps_df = apps_df.dropna(subset=['Last Updated'])


fig9 = px.box(
    apps_df,
    x='Type',
    y='Rating',  
    color='Type',
    title='Rating for paid vs Free Apps', 
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=800,
    height=400
)

fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

fig9.write_html("Paid free garph 10.html", auto_open=True)


# In[26]:


#intern task 1
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\SUNIL\Downloads\User Reviews.csv")


print(data.head())
print(data.columns)  


def assign_rating(sentiment):
    if sentiment == 'Positive':
        return 5
    elif sentiment == 'Neutral':
        return 3
    elif sentiment == 'Negative':
        return 1
    else:
        return None


data['rating'] = data['Sentiment'].apply(assign_rating)


print("Data after assigning ratings:")
print(data.head())


app_review_counts = data['App'].value_counts()
print("App review counts:")
print(app_review_counts)


apps_with_reviews = app_review_counts[app_review_counts > 100].index
data = data[data['App'].isin(apps_with_reviews)]


print("Data after filtering apps with more than 100 reviews:")
print(data.head())


def rating_group(rating):
    if rating <= 2:
        return '1-2 stars'
    elif rating <= 4:
        return '3-4 stars'
    else:
        return '4-5 stars'

data['rating_group'] = data['rating'].apply(rating_group)


print("Data after grouping ratings:")
print(data.head())


top_apps = data['App'].value_counts().nlargest(5).index
data = data[data['App'].isin(top_apps)]


print("Data after filtering top apps:")
print(data.head())


sentiment_counts = data.groupby(['App', 'rating_group', 'Sentiment']).size().unstack(fill_value=0)


print("Sentiment counts:")
print(sentiment_counts)


if not sentiment_counts.empty:
    sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Sentiment Distribution of User Reviews by Rating Groups')
    plt.xlabel('App')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No data to plot.")


# In[2]:


#intern task2
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz

file_path = r"C:\Users\SUNIL\Downloads\Play Store Data.csv"
data = pd.read_csv(file_path)

data['Installs'] = data['Installs'].str.replace(' ', '').str.replace('+', '').str.replace(',', '')
data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce')


data['Country'] = 'United States'  

filtered_data = data[~data['Category'].str.startswith(('A', 'C', 'G', 'S'))]
top_categories = filtered_data.groupby('Category')['Installs'].sum().nlargest(5).index
filtered_data = filtered_data[filtered_data['Category'].isin(top_categories)]
filtered_data = filtered_data[filtered_data['Installs'] > 1_000_000]

ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).time()

start_time = datetime.strptime("18:00", "%H:%M").time()
end_time = datetime.strptime("20:00", "%H:%M").time()

if start_time <= current_time <= end_time:
    fig = px.choropleth(
        filtered_data,
        locations='Country',  
        locationmode='country names',
        color='Installs',
        hover_name='Category',
        title='Global Installs by App Categories',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.show()
else:
    print("The graph can only be displayed between 6 PM and 8 PM IST.")


# In[4]:


#intern task 3
import pandas as pd
import plotly.graph_objects as go
import datetime
import pytz

file_path = r"C:\Users\SUNIL\Downloads\Play Store Data.csv"
df = pd.read_csv(file_path)

def clean_data(df):
    df['Installs'] = (
        df['Installs']
        .str.replace('[+,]', '', regex=True)
        .str.extract('(\d+)')
        .dropna()
        .astype(int)
    )
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    df.dropna(subset=['Last Updated'], inplace=True)
    df['Month'] = df['Last Updated'].dt.to_period('M')
    return df

def filter_data(df):
    filtered_df = df[
        (df['Content Rating'] == 'Teen') &
        (df['App'].str.startswith('E')) &
        (df['Installs'] > 10000)
    ]
    return filtered_df

def calculate_monthly_trend(df):
    monthly_installs = df.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
    monthly_installs['MoM Growth'] = monthly_installs.groupby('Category')['Installs'].pct_change() * 100
    return monthly_installs

def create_time_series_chart():
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.datetime.now(ist)
    start_time = current_time.replace(hour=18, minute=0, second=0, microsecond=0)
    end_time = current_time.replace(hour=21, minute=0, second=0, microsecond=0)

    if start_time <= current_time <= end_time:
        cleaned_df = clean_data(df)
        filtered_df = filter_data(cleaned_df)
        trend_data = calculate_monthly_trend(filtered_df)

        fig = go.Figure()
        categories = trend_data['Category'].unique()

        for category in categories:
            category_data = trend_data[trend_data['Category'] == category]
            fig.add_trace(
                go.Scatter(
                    x=category_data['Month'].astype(str),
                    y=category_data['Installs'],
                    mode='lines',
                    name=category,
                )
            )

            significant_growth = category_data[category_data['MoM Growth'] > 20]
            fig.add_trace(
                go.Scatter(
                    x=significant_growth['Month'].astype(str),
                    y=significant_growth['Installs'],
                    fill='tozeroy',
                    name=f"{category} - Significant Growth",
                    mode='none',
                    fillcolor='rgba(0, 200, 100, 0.2)'
                )
            )

        fig.update_layout(
            title="Monthly Trend of Total Installs by App Category",
            xaxis_title="Month",
            yaxis_title="Total Installs",
            legend_title="Category",
            template="plotly_white"
        )
        fig.show()
    else:
        print("The graph is only available between 6 PM and 9 PM IST.")

create_time_series_chart()


# In[ ]:




