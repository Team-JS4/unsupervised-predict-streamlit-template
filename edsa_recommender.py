"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
        from PIL import Image
        train = pd.read_csv("resources/data/ratings.csv")
        movies = pd.read_csv("resources/data/movies.csv")
        if st.checkbox('Show team members'):                    
                        image = Image.open('resources/imgs/IMG-20220707-WA0072.jpg')
                        st.image(image, caption='Senior Data Scientist',width=500)
                        image = Image.open('resources/imgs/5026eb40-a23e-4c2f-a9a4-c2a5a87bc522.jpg')
                        st.image(image, caption='Big Data Engineer',width=500)
                        image = Image.open('resources/imgs/28dce6c3-e238-4318-9ee6-eefb87d1dbb6.jpg')
                        st.image(image, caption='Business Intelligence Analyst',width=500)
        if st.checkbox('Overview'):
                    st.info("General Overview")
                    st.markdown("### Data Description")
                    st.markdown("This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!")
                    st.markdown("In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.")
                    st.markdown("With this context, EDSA us challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.")
                    st.markdown("Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being personalised recommendations - generating platform affinity for the streaming services which best facilitates their audience's viewing.")
                    st.markdown("#### Raw data")
                    st.info("Ratings")
                    st.write(train[['userId', 'movieId', 'rating', 'timestamp']])
                    st.info("Movies")
                    st.write(movies[['movieId', 'title', 'genres']])
                    
        if st.checkbox('EDA'):
                    st.info("Perfomance metrics")
                    import plotly.express as px
                    from datetime import datetime
                    import plotly.graph_objects as go
                    import re
                    import warnings
                    warnings.filterwarnings("ignore")
                    import scipy as sp # <-- The sister of Numpy, used in our code for numerical efficientcy. 
                    # Entity featurization and similarity computation
                    from sklearn.metrics.pairwise import cosine_similarity 
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    # Libraries used during sorting procedures.
                    import operator # <-- Convienient item retrieval during iteration 
                    import heapq # <-- Efficient sorting of large lists
                    # Imported for our sanity
                    import warnings
                    
                    import cufflinks as cf
                    from sklearn import preprocessing
                    from sklearn.decomposition import PCA
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor
                    # data processing
                    
                    
                    import altair as alt
                    from sklearn import preprocessing
                    from sklearn.datasets import make_blobs
                    from sklearn.preprocessing import StandardScaler
                    import cufflinks as cf
                    from sklearn import preprocessing
                    from sklearn.decomposition import PCA
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor
                    # plotting
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import ListedColormap
                    import seaborn as sns
                    
                    sns.set(style='whitegrid', palette='muted',
                                    rc={'figure.figsize': (15,10)})
                    sns.set_style('dark')
                    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
                    init_notebook_mode(connected=True)
                    trains = pd.read_csv("resources/data/train.csv")
                    tags = pd.read_csv("resources/data/tags.csv")
                    
                    imdb_data = pd.read_csv("resources/data/imdb_data.csv")
                    
                    movies = pd.read_csv("resources/data/movies.csv")
                    genres_value_counts = movies['genres'].str.split('|', expand=True).stack().value_counts()
                   
                    # Extracting movie release years into one column

                    genres_series = genres_value_counts
                    genres_series_df = pd.DataFrame(genres_series)
                    genres_series_list = list(genres_series)
                    genres_index = genres_series_df.index
                    genres_index_list = list(genres_index)
                    genres_count = pd.Series(genres_series_list)
                    genres_ = pd.Series(genres_index_list)
                    genres_frame = { 'genres': genres_index_list, 'genres_count': genres_series_list }
                    genres_frame_df = pd.DataFrame(genres_frame)

                    # Extracting movie release years into one column
                    movies['movie_year'] = movies.title.str.extract('.*\((.*)\).*')
                    movie_year = movies["movie_year"].unique()
                    movie_year_series = movies['movie_year'].value_counts(ascending=False).head(10)
                    movie_year_series_df = pd.DataFrame(movie_year_series)
                    movie_year_series_list = list(movie_year_series)
                    movie_year_index = movie_year_series_df.index
                    movie_year_index_list = list(movie_year_index)
                    movie_year_count = pd.Series(movie_year_series_list)
                    movie_year_ = pd.Series(movie_year_index_list)
                    movie_year_frame = { 'movies': movie_year_index_list, 'movie_count': movie_year_series_list }
                    movie_year_frame_df = pd.DataFrame(movie_year_frame)

                    director = imdb_data["director"].unique()
                    director_series = imdb_data['director'].value_counts(ascending=False).head(20)
                    director_series_df = pd.DataFrame(director_series)
                    director_series_list = list(director_series)
                    director_index = director_series_df.index
                    director_index_list = list(director_index)
                    director_count = pd.Series(director_series_list)
                    directors = pd.Series(director_index_list)
                    director_frame = { 'directors': director_index_list, 'director_count': director_series_list }
                    director_frame_df = pd.DataFrame(director_frame)

                    rating_series = trains['rating'].value_counts(ascending=False)
                    rating_series_df = pd.DataFrame(rating_series)
                    rating_series_list = list(rating_series)
                    rating_index = rating_series_df.index
                    rating_index_list = list(rating_index)
                    rating_count = pd.Series(rating_series_list)
                    rating = pd.Series(director_index_list)
                    rating_frame = { 'ratings': rating_index_list, 'rating_count': rating_series_list }
                    rating_frame_df = pd.DataFrame(rating_frame)

                    imdb_data['title_cast']=imdb_data['title_cast'].str.split('|') #spliting the title cast into a list
                    imdb_data['plot_keywords']=imdb_data['plot_keywords'].str.split('|') #spliting the Key words into a list
                    title_cast = imdb_data['title_cast'].explode()
                    title_cast_series = title_cast.value_counts(ascending=False).head(20)
                    title_cast_series_df = pd.DataFrame(title_cast_series)
                    title_cast_series_list = list(title_cast_series)
                    title_cast_index = title_cast_series_df.index
                    title_cast_index_list = list(title_cast_index)
                    title_cast_count = pd.Series(title_cast_series_list)
                    title_cast_ = pd.Series(title_cast_index_list)
                    title_cast_frame = { 'title_cast': title_cast_index_list, 'title_cast_count': title_cast_series_list }
                    title_cast_frame_df = pd.DataFrame(title_cast_frame)

                    #Merging train data and movies data
                    mov_rating = pd.merge(movies,trains, on = "movieId")
                    #Merging above dataframe with imdb data
                    mov_rating = pd.merge(mov_rating,imdb_data, on ="movieId")
                    mov_rating.head()
                    best_director = pd.DataFrame(mov_rating.groupby('director')['rating'].mean().sort_values(ascending=False))
                    best_director['No_of_ratings'] = mov_rating.groupby('director')['rating'].count()
                    best_director_df = best_director.sort_values(by=['No_of_ratings', 'rating'], ascending=False).head(10)
                    best_director_list = list(best_director_df.index)
                    number_of_ratings_list = list(best_director_df['No_of_ratings'])

                    def UNIX_to_Readable(df):
                        return pd.to_datetime(datetime.fromtimestamp(df).strftime('%Y-%m-%d %H:%M:%S'))


                    # Converting Unix date-format to readable format
                    tags.timestamp = tags.timestamp.apply(UNIX_to_Readable)
                    tags.head()

                    monthly = tags.resample('m', on='timestamp').size()
                    am = pd.DataFrame(monthly)
                    monthly_series = am[0]
                    monthly_series_df = pd.DataFrame(monthly_series)
                    monthly_series_list = list(monthly_series)
                    monthly_index = monthly_series_df.index
                    monthly_index_list = list(monthly_index)

                    monthly_count = pd.Series(monthly_series_list)
                    monthly_ = pd.Series(monthly_index_list)
                    monthly_frame = { 'monthly': monthly_index_list, 'monthly count': monthly_series_list }
                    monthly_frame_df = pd.DataFrame(monthly_frame)
                    

                 

                    
                    if st.checkbox("Show  Monthly ratings Bar Graph"):
                        st.subheader("Monthly ratings Bar Graph")                      
                        data_set = {
                            'Months': monthly_index_list,
                            'Ratings': monthly_series_list
                        }

                        df = pd.DataFrame(data_set)

                        line = alt.Chart(df).mark_line().encode(
                            x = 'Months',
                            y = 'Ratings'
                        )
                        st.altair_chart(line)

                        if st.checkbox('Show Monthly ratings Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* From 2006 to the begining of 2018 the ratings are below 20 thousand.")
                                        st.markdown("* Then there's a huge spike increase in ratings, during 2018 month(s).")
                                       
                                
                             

                    best_director_dict = {}
                    for i in range(len(best_director_list)):
                        best_director_dict[best_director_list[i]] = number_of_ratings_list[i]
                    if st.checkbox("Show Best Director Bar Graph"):
                      
                        best_director_list2 = st.multiselect("Which directors would you like to see?", best_director_list)
                        best_director_emp_list = []
                        if len(best_director_list2) == 0:
                                st.subheader("Best Director Bar Graph")
                                
                                source = pd.DataFrame({
                                    'Director rating counts': number_of_ratings_list,
                                    'Directors': best_director_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y="Director rating counts:Q",
                                    x='Directors:O',
                                   
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)

                                if st.checkbox('Show Best Director Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
                                
                        else:
                                for i in best_director_list2:
                                        best_director_emp_list.append(best_director_dict[i])
                                
                                st.subheader("Best Director Bar Graph")
                                
                                source = pd.DataFrame({
                                'Director rating counts': best_director_emp_list, 'Directors': best_director_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                      y="Director rating counts:Q",
                                    x='Directors:O',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                    title_cast_dict = {}
                    for i in range(len(title_cast_index_list)):
                        title_cast_dict[title_cast_index_list[i]] = title_cast_series_list[i]
                    if st.checkbox("Show Title cast Bar Graph"):
                      
                        title_cast_list2 = st.multiselect("Which cast would you like to see?", title_cast_index_list)
                        title_cast_emp_list = []
                        if len(title_cast_list2) == 0:
                                st.subheader("Title cast Bar Graph")
                                
                                source = pd.DataFrame({
                                    'Title cast count': title_cast_series_list,
                                    'Title cast': title_cast_index_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y="Title cast count:Q",
                                    x='Title cast:O',
                                   
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)

                                if st.checkbox('Show Top Title cast Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
                                

                                
                        else:
                                for i in title_cast_list2:
                                        title_cast_emp_list.append(title_cast_dict[i])
                                
                                st.subheader("Title cast Bar Graph")
                                
                                source = pd.DataFrame({
                                'Title cast count': title_cast_emp_list, 'Title cast': title_cast_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                      y="Title cast count:Q",
                                    x='Title cast:O',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                    rating_dict = {}
                    for i in range(len(rating_index_list)):
                        rating_dict[rating_index_list[i]] = rating_series_list[i]
                    if st.checkbox("Show Ratings Bar Graph"):
                        rating_list2 = st.multiselect("Which rating would you like to see?", rating_index_list)
                        rating_emp_list = []
                        if len(rating_list2) == 0:
                                st.subheader("Ratings Bar Graph")
                                
                                source = pd.DataFrame({
                                    'Number of ratings': rating_series_list,
                                    'Ratings': rating_index_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y="Number of ratings:Q",
                                    x='Ratings:O',
                                   
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('Show Top Ratings Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
                                
                        else:
                                for i in rating_list2:
                                        rating_emp_list.append(rating_dict[i])
                                
                                st.subheader("Ratings Bar Graph")
                                
                                source = pd.DataFrame({
                                'Number of ratings': rating_emp_list, 'Ratings': rating_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                      y="Number of ratings:Q",
                                    x='Ratings:O',
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                    director_dict = {}
                    for i in range(len(director_index_list)):
                        director_dict[director_index_list[i]] = director_series_list[i]
                    if st.checkbox("Show Directors Bar Graph"):
                        director_list2 = st.multiselect("Which director would you like to see?", director_index_list)
                        director_emp_list = []
                        if len(director_list2) == 0:
                                st.subheader("Directors Bar Graph")
                                source = pd.DataFrame({
                                    'Director movies count': director_series_list,
                                    'Directors': director_index_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y="Director movies count:Q",
                                    x='Directors:O'
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('Show Directors Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in director_list2:
                                        director_emp_list.append(director_dict[i])
                                
                                st.subheader("Directors Bar Graph")
                               
                                source = pd.DataFrame({
                                'Director movies count': director_emp_list, 'Directors': director_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y="Director movies count:Q",
                                    x='Directors:O'
                                )
                                st.altair_chart(bar_chart, use_container_width=True)
                

                    genre_dict = {}
                    for i in range(len(genres_index_list)):
                        genre_dict[genres_index_list[i]] = genres_series_list[i]
                    if st.checkbox("Show Top 20 genre Bar Graph"):
                        genre_list2 = st.multiselect("Which genre would you like to see?", genres_index_list)
                        genre_emp_list = []
                        if len(genre_list2) == 0:
                                st.subheader("Top 20 genre Bar Graph")
                                source = pd.DataFrame({
                                    'Genre count': genres_series_list,
                                    'Genre': genres_index_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Genre count:Q',
                                    x="Genre:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                if st.checkbox('Show Top 20 genre Bar Graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")
                        else:
                                for i in genre_list2:
                                        genre_emp_list.append(genre_dict[i])
                                
                                st.subheader("Top 20 genre Bar Graph")
                                
                                source = pd.DataFrame({
                                'Genre count': genre_emp_list, 'Genre': genre_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y='Genre count:Q',
                                    x="Genre:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                                
                    movie_year_dict = {}
                    for i in range(len(movie_year_index_list)):
                        movie_year_dict[movie_year_index_list[i]] = movie_year_series_list[i]
                    if st.checkbox("Show Top 10 movie years graph"):
                        movie_year_list2 = st.multiselect("Which year would you like to see?", movie_year_index_list)
                        movie_year_emp_list = []
                        if len(movie_year_list2) == 0:
                                st.subheader("Top 10 movie years graph")
                                source = pd.DataFrame({
                                    'Number of movies': movie_year_series_list,
                                    'Year': movie_year_index_list
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Number of movies:Q',
                                    x="Year:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('Show Top Top 10 movie years graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that pro tweets make up the majority with 53.92%")
                                        st.markdown("* Followed by news 23.01%, neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in movie_year_list2:
                                        movie_year_emp_list.append(movie_year_dict[i])
                                
                                st.subheader("Top 10 movie years graph")
                                
                                source = pd.DataFrame({
                                'Number of movies': movie_year_emp_list, 'Year': movie_year_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y='Number of movies:Q',
                                    x="Year:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True) 
                                
        if st.checkbox('Netflix Pitch'):
                    if st.checkbox("Show Netflix Streak graph"):
                        image = Image.open('resources/imgs/7677.jpg')
                        st.image(image, caption='STREAK ENDS!!!!!!!!!',width=500)
                    #Graph1
                    region = ["US and Canada", "Europe, Middle East, and Africa", "Latin America", "Asia Pacific"]
                    revenue = [2979.51, 2137.16, 788.52, 684.61]
                    revenue_x_region_dict = {}                     
                    for i in range(len(region)):
                        revenue_x_region_dict[region[i]] = revenue[i]
                    if st.checkbox("Show Netflix Revenue by Region in 2020 graph"):
                        region_list2 = st.multiselect("Which region would you like to see?", region)
                        region_emp_list = []
                        if len(region_list2) == 0:
                                st.subheader("Netflix Revenue by Region in 2020")
                                source = pd.DataFrame({
                                    'Revenue(in millions USD)': revenue,
                                    'Region': region
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Revenue (in millions USD):Q',
                                    x="Region:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('("Show Netflix Revenue by Region in 2020 graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                                        st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in region_list2:
                                        region_emp_list.append(revenue_x_region_dict[i])
                                
                                st.subheader("Netflix Revenue by Region in 2020 graph")
                                
                                source = pd.DataFrame({
                                'Number of movies': movie_year_emp_list, 'Year': movie_year_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                        y='Number of movies:Q',
                                    x="Year:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True) 
                                            
                    #Graph2
                    region = ["US and Canada", "Europe, Middle East, and Africa", "Latin America", "Asia Pacific"]
                    monthly_revenue = [13.51, 11.05, 9.32, 7.12]
                    monthly_revenue_x_region_dict = {}                     
                    for i in range(len(region)):
                        revenue_x_region_dict[region[i]] = revenue[i]
                    if st.checkbox("Show Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020 graph"):
                        region_list2 = st.multiselect("Which region would you like to see?", region)
                        region_emp_list = []
                        if len(region_list2) == 0:
                                st.subheader("Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020")
                                st.subheader("in USD")
                                source = pd.DataFrame({
                                    'Monthly Revenue': monthly_revenue,
                                    'Region': region
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Monthly Revenue :Q',
                                    x="Region:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('("Show Netflix Revenue by Region in 2020 graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                                        st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in region_list2:
                                        region_emp_list.append(monthly_revenue_x_region_dict[i])
                                
                                st.subheader("Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020")
                                st.subheader("in USD")
                                source = pd.DataFrame({
                                'Monthly Revenue': region_emp_list, 'Region': region_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Monthly Revenue:Q',
                                    x="Region:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                    #Graph3
                    region = ["US and Canada", "Europe, Middle East, and Africa", "Latin America", "Asia Pacific"]
                    share = [73.94, 66.7, 37.58, 25.49]
                    share_x_region_dict = {}                     
                    for i in range(len(region)):
                        share_x_region_dict[region[i]] = revenue[i]
                    if st.checkbox("Show Share of Netflix Subscribers Worldwide by Region as of Q4 2020 graph"):
                        region_list2 = st.multiselect("Which region would you like to see?", region)
                        region_emp_list = []
                        if len(region_list2) == 0:
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                    'Monthly Revenue': monthly_revenue,
                                    'Region': region
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Share :Q',
                                    x="Region:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('("Show Netflix Revenue by Region in 2020 graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                                        st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in region_list2:
                                        region_emp_list.append(share_x_region_dict[i])
                                
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                'Share': region_emp_list, 'Region': region_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Share:Q',
                                    x="Region:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

                    #Graph4
                    region = ["US and Canada", "Europe, Middle East, and Africa", "Latin America", "Asia Pacific"]
                    share = [73.94, 66.7, 37.58, 25.49]
                    share_x_region_dict = {}                     
                    for i in range(len(region)):
                        share_x_region_dict[region[i]] = revenue[i]
                    if st.checkbox("Show Share of Netflix Subscribers Worldwide by Region as of Q4 2020 graph"):
                        region_list2 = st.multiselect("Which region would you like to see?", region)
                        region_emp_list = []
                        if len(region_list2) == 0:
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                    'Monthly Revenue': monthly_revenue,
                                    'Region': region
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Share :Q',
                                    x="Region:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                                
                                if st.checkbox('("Show Netflix Revenue by Region in 2020 graph interperetation'):
                                        st.markdown("**Looking at the above bar graph:**")
                                        st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                                        st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                                
                        else:
                                for i in region_list2:
                                        region_emp_list.append(share_x_region_dict[i])
                                
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                'Share': region_emp_list, 'Region': region_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Share:Q',
                                    x="Region:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True) 
                                
                    if st.checkbox("Show Countries with the most Netflix Subscribers"):
                        country = ["United States", "Brazil", "United Kingdom", "France", "Germany", "Mexico", "Canada", "Australia", "Argentina", "Japan"]
                        subscribers = [65.3, 17.9, 16.7, 8.6, 8.2, 8.1, 7.9, 6.9, 5.4, 4.2]
                        subscibers_x_country_dict = {}
                        for i in range(len(region)):
                            subscibers_x_country_dict[country[i]] = subscribers[i]
                        country_list2 = st.multiselect("Which hashtag would you like to see?", country)
                        subscribers_emp_list = []
                        if len(country_list2) == 0:
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                    'Subscribers': subscribers,
                                    'Country': country
                                 })
                                

                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Subscribers:Q',
                                    y="Country:O"
                                )#.properties(height=700)
                                st.altair_chart(bar_chart, use_container_width=True)
                        else:
                                for i in country_list2:
                                        subscribers_emp_list.append(subscibers_x_country_dict[i])
                                
                                st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                                st.subheader("in millions")
                                source = pd.DataFrame({
                                'Subscribers': subscribers_emp_list, 'Country': country_list2 })
  
                                bar_chart = alt.Chart(source).mark_bar().encode(
                                    x='Subscribers:Q',
                                    y="Country:O"
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

if __name__ == '__main__':
    main()
