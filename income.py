import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from prophet import Prophet

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text for word cloud
def preprocess_text_for_wordcloud(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.add('nan')
    stop_words.add('anonymous')
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    text = ' '.join(tokens)
    return text

# Load data with caching to improve performance
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Clean_Title'] = df['Title'].astype(str).apply(preprocess_text_for_wordcloud)
    df.dropna(subset=['Clean_Title'], inplace=True)
    return df

df = load_data("C:/Users/Admin/Desktop/SPAN COMMUNICATIONS/INCOME TAX/INCOME_TAX.xlsx")

# Set title and description
st.title("LISTENING OVERVIEW")
st.write("_Get a deeper understanding of your incoming mentions across all channels_", unsafe_allow_html=True)
st.divider()

# Define brand and duration options
options_brand = ("Income Tax Department of India",)
options_duration = ("2 years",)

# Create a layout with two columns
col1, spacer, col2 = st.columns([1, 1, 1])

# Add selectboxes in each column
with col1:
    selected_brand = st.selectbox("**Brand**", options_brand)

with col2:
    selected_duration = st.selectbox("**Duration**", options_duration)

st.divider()

# Sidebar for different insights
st.sidebar.title("LISTENING INSIGHTS")

overview_icon = "📊"
influencer_icon = "👩‍💻"
campaign_icon = "📅"
pr_icon = "📰"
predictive_icon = "📈"

overview_button = st.sidebar.button(f"{overview_icon} Overview", key="overview", help="Overview of Mentions")
influencer_button = st.sidebar.button(f"{influencer_icon} Influencer", key="influencer", help="Influencer Analysis")
campaign_button = st.sidebar.button(f"{campaign_icon} Campaign Analysis", key="campaign", help="Campaign Analysis")
pr_button = st.sidebar.button(f"{pr_icon} PR Dashboard", key="PR", help="PR Dashboard")
predictive_button = st.sidebar.button(f"{predictive_icon} Predictive Analysis", key='Predictive', help="Predictive Analysis")

# Overview calculations
mentions = df['Date'].count()
mention_day = df['Year_Day'].value_counts().mean()
mention_hour = df['Hour_Date'].value_counts().mean()

# Display overview content based on button clicks
if overview_button:
    st.title(f"OVERVIEW")
    st.divider()
    st.write("Total Mentions:", mentions)
    st.write("Average Mentions / Day:", mention_day)
    st.write("Average Mentions / Hour:", mention_hour)

    # Sentiment analysis pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    sentiments = sentiment_counts.index
    counts = sentiment_counts.values
    percentages = 100 * counts / counts.sum()
    users = df['User Name'].nunique()
    colors = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'orange'
    }

    fig = go.Figure(data=[go.Pie(labels=sentiments, values=counts, pull=[0, 0, 0.2], hole=0.3,
                                 marker=dict(colors=[colors[s] for s in sentiments]))])
    fig.update_layout(
        title="Sentiment Distribution",
        title_font=dict(size=16, family='Arial'),
        showlegend=True,
        legend=dict(
            title="Sentiment",
            x=1, y=0.5,
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="Black",
            borderwidth=1
        )
    )
    st.plotly_chart(fig)
    st.write("Total unique users:", users)

    # Sentiment trends line chart
    sentiment_counts = df.groupby(['Year_Month', 'Sentiment']).size().reset_index(name='Count')
    fig = px.line(sentiment_counts, x='Year_Month', y='Count', color='Sentiment',
                  title='Sentiment Trends over Month',
                  labels={'Year_Month': 'Year_Month', 'Count': 'Count', 'Sentiment': 'Sentiment'},
                  markers=True)
    fig.update_traces(line_shape='spline', marker=dict(size=6, symbol='circle-open'))
    colors = {'Neutral': 'yellow', 'Negative': 'red', 'Positive': 'blue'}
    fig.for_each_trace(lambda trace: trace.update(line=dict(color=colors[trace.name])))
    fig.update_layout(xaxis_title='Year_Month', yaxis_title='Count')
    st.plotly_chart(fig)

    # Monthly trends line chart
    monthly_sums = df.groupby('Year_Month').agg({'Impression': 'sum', 'Reach': 'sum', 'Total Engagement': 'sum'}).reset_index()
    traces = []
    metrics = ['Impression', 'Reach', 'Total Engagement']
    colors = ['blue', 'red', 'green']
    for metric, color in zip(metrics, colors):
        trace = go.Scatter(
            x=monthly_sums['Year_Month'],
            y=monthly_sums[metric],
            mode='lines+markers',
            name=metric,
            line=dict(shape='spline', color=color),
            marker=dict(size=6, symbol='circle-open')
        )
        traces.append(trace)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Monthly Trends of Impressions, Reach, and Total Engagement',
        xaxis_title='Year_Month',
        yaxis_title='Sum',
        hovermode='x unified'
    )
    st.plotly_chart(fig)

    # Platform distribution bar graph with percentages
    platform_counts = df['Platform'].value_counts()
    platforms = platform_counts.index
    counts = platform_counts.values
    percentages = 100 * counts / counts.sum()

    fig = go.Figure(data=[go.Bar(x=platforms, y=counts, text=[f'{p:.2f}%' for p in percentages], 
                                 textposition='auto', marker_color=px.colors.qualitative.Plotly)])
    fig.update_layout(
        title_text="Channel Wise Distribution",
        title_font_size=20,
        xaxis_title="Platforms",
        yaxis_title="Counts",
        showlegend=False
    )
    st.plotly_chart(fig)

    # Platform sentiment counts table
    platform_sentiment_counts = df.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0).reset_index()
    st.table(platform_sentiment_counts)

    # Download button for platform sentiment counts as CSV
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(platform_sentiment_counts)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='platform_sentiment_counts.csv',
        mime='text/csv',
    )
    # Language distribution bar graph
    language_counts = df['Language'].value_counts()
    languages = language_counts.index
    counts = language_counts.values
    percentages = 100 * counts / counts.sum()

    fig = go.Figure(data=[go.Bar(x=languages, y=counts, text=[f'{p:.2f}%' for p in percentages], 
                                 textposition='auto', marker_color=px.colors.qualitative.Plotly)])
    fig.update_layout(
        title_text="Language Distribution",
        title_font_size=20,
        xaxis_title="Languages",
        yaxis_title="Counts",
        showlegend=False
    )
    st.plotly_chart(fig)    
    # Generate word cloud from 'Clean_Title' column
    st.title("Word Cloud")
    titles_text = ' '.join(df['Clean_Title'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Influencer analysis calculations and display
if influencer_button:
    st.title(f"INFLUENCER")
    st.divider()
    total_influencers = df[df['Followers'] > 1000]['User Name'].nunique()
    st.write("Total Influencers :", total_influencers)

# Campaign analysis calculations and display
if campaign_button:
    st.title("CAMPAIGN ANALYSIS")
    hour_counts = df['Hour'].value_counts().reset_index()
    hour_counts.columns = ['Hour', 'Count']
    hour_counts = hour_counts.sort_values(by='Hour')
    fig = px.line(hour_counts, x='Hour', y='Count', title='Best Time to Engage', labels={'Hour': 'Hour of the Day', 'Count': 'Sum of Count'})
    fig.update_traces(mode='lines+markers', line_shape='spline', marker=dict(size=8, symbol='circle'))
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    st.subheader("Media Type Distribution")
    media_counts = df['Media Type'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=media_counts.index, values=media_counts.values, hole=0.5, rotation=45)])
    st.plotly_chart(fig)

    # Word cloud for each sentiment
    st.subheader("Word Cloud for each Sentiment")
    sentiments = df['Sentiment'].unique()
    for sentiment in sentiments:
        st.subheader(f"{sentiment.capitalize()} Sentiment")
        sentiment_text = ' '.join(df[df['Sentiment'] == sentiment]['Clean_Title'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# PR dashboard calculations and display
if pr_button:
    st.title("PR DASHBOARD")
    st.divider()
    st.subheader('Top Author Name')
    # Table for top 50 author name value counts
    author_counts = df['User Name'].value_counts().reset_index()
    author_counts.columns = ['User Name', 'Count']

    # Show only top 50 authors
    top_authors = author_counts.head(50)

    # Display the table
    st.table(top_authors)

    # Download button for top 50 authors as CSV
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(top_authors)

    st.download_button(
        label="Download Top 50 Authors as CSV",
        data=csv,
        file_name='top_50_authors.csv',
        mime='text/csv',
    )
    st.subheader('Top Domain')
    # Table for top 50 author name value counts
    author_counts = df['Domain'].value_counts().reset_index()
    author_counts.columns = ['Domain', 'Count']

    # Show only top 50 authors
    top_authors = author_counts.head(50)

    # Display the table
    st.table(top_authors)

    # Download button for top 50 authors as CSV
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(top_authors)

    st.download_button(
        label="Download Top Domains as CSV",
        data=csv,
        file_name='top_domain.csv',
        mime='text/csv',
    )

# Predictive analysis section
if predictive_button:
    st.title("PREDICTIVE ANALYSIS")
    st.divider()

    # Convert Date to the desired format and calculate sentiment scores
    df["Date2"] = df["Date"].dt.strftime("%Y-%m-%d %H")
    df2 = pd.DataFrame(df.groupby("Date2")["Sentiment"].value_counts())
    df2 = df2.reset_index()
    df2 = df2.pivot(index="Date2", columns="Sentiment", values="count").fillna(0)
    df2["Score"] = df2["Negative"] - df2["Positive"]
    df2["cumScore"] = df2["Score"].cumsum()
    df3 = df2.reset_index()[['Date2', 'cumScore']].rename(columns={"Date2": "ds", "cumScore": "y"})

    # Plot cumulative sentiment scores
    fig, ax = plt.subplots()
    df3.plot(x='ds', y='y', ax=ax)
    plt.title('Cumulative Sentiment Score over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sentiment Score')
    st.pyplot(fig)

    # Initialize and fit the Prophet model
    m = Prophet(
        interval_width=0.99,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.fit(df3)

    # Create future dataframe for predictions
    future = m.make_future_dataframe(periods=300, freq='H')
    forecast = m.predict(future)

    # Plot the forecast
    fig1 = m.plot(forecast)
    plt.title('Sentiment Trend Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    st.pyplot(fig1)
