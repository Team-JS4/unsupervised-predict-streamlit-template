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

from streamlit_option_menu import option_menu
import altair as alt
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

raw = pd.read_csv("resources/data/ratings.csv")

rat = pd.read_csv("resources/data/table.csv")

# App declaration
def main():

	# DO NOT REMOVE the 'Recommender System' option below, however,
	# you are welcome to add more options to enrich your app.
	with st.sidebar:
		selection = option_menu("Main Menu", ["Home", 'Visualisation','Top Rated', 'Development team','About Us','Contact Us'],
		icons=['house', 'pie-chart', 'star-fill','people-fill','info-square-fill','envelope'], menu_icon="cast", default_index=0)
	

	# -------------------------------------------------------------------
	# ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
	# -------------------------------------------------------------------
	#page_selection = st.sidebar.selectbox("Choose Option", page_options)
	if selection == "Home":
		# Header contents
		st.image('logo.png', width=200)
		st.write('# Movie Recommender Engine')
		st.write('### EXPLORE Data Science Academy Unsupervised Predict')
		st.image('resources/imgs/Image_header.png',use_column_width=True)
		# Recommender System algorithm selection
		sys = st.radio("Select an algorithm",
					   ('Content Based Filtering',
						'Collaborative Based Filtering'))

		# User-based preferences
		st.write('### Enter Your Three Favorite Movies')
		movie_1 = st.selectbox('First Option',title_list[100:200])
		movie_2 = st.selectbox('Second Option',title_list[1000:2000])
		movie_3 = st.selectbox('Third Option',title_list[2000:3000])
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
	elif selection == "Visualisation":
		st.title("Visualisation of the Raw Data")
		st.write("Here is a visual overview of the total number of ratings and top 10 users by number of ratings")

		opt = st.radio('Visualise data aspects:',['Total number of ratings', 'Top ten users by number of ratings'])

		if opt =='Total number of ratings':
			xx = raw['rating'].value_counts()
			col1, mid, col2 = st.columns([20,2,80])
			with col1:
				st.write(xx)
			with col2:
				xx = pd.DataFrame(xx).reset_index()
				xx.columns = ['rating','number of ratings']
				cc = alt.Chart(xx).mark_bar().encode(
				x = f'{xx.columns[0]}:N',
				y = xx.columns[1],
				color ='rating:N')
				st.altair_chart(cc, use_container_width=True)

		if opt =='Top ten users by number of ratings':
			xy = raw['userId'].value_counts().sort_values(ascending=False).head(10)
			col1, mid, col2 = st.columns([20,2,80])
			with col1:
				st.write(xy)
			with col2:
				xz = pd.DataFrame(xy).reset_index()
				xz.columns = ['user','count']
		
		
				c = alt.Chart(xz).mark_bar().encode(
			x = alt.X(f'{xz.columns[0]}:N',sort='-y'),
			y = xz.columns[1],
			color ='user:N'
			)
				st.altair_chart(c, use_container_width=True)
		#st.bar_chart(xy)
	# You may want to add more sections here for aspects such as an EDA,
	# or to provide your business pitch.

	elif selection == "Top Rated":
		st.markdown("<h2 style=color:#3FBEBF;>Top Rated Movies By Genres</h2>", unsafe_allow_html=True)
		col1, mid, col2 = st.columns([20, 2, 80])
		with col1:
			act = st.checkbox("Action")
			war = st.checkbox("War")
			rom = st.checkbox("Romantic")
			com = st.checkbox("Comedy")
			drm = st.checkbox("Drama")
			adv = st.checkbox("Adventure")
			sf = st.checkbox("Sci-Fi")
			thr = st.checkbox("Thriller")
			ani = st.checkbox("Animation")

			ls = "True"
			btn = st.button("Explore")

			if btn:
				if act:
					ls = ls + " & rat['genres'].str.contains(\"Action\")"

				if war:
					ls = ls + " & rat['genres'].str.contains(\"War\")"

				if rom:
					ls = ls + " & rat['genres'].str.contains(\"Romance\")"

				if com:
					ls = ls + " & rat['genres'].str.contains(\"Comedy\")"

				if drm:
					ls = ls + " & rat['genres'].str.contains(\"Drama\")"

				if adv:
					ls = ls + " & rat['genres'].str.contains(\"Adventure\")"

				if sf:
					ls = ls + " & rat['genres'].str.contains(\"Sci-Fi\")"

				if thr:
					ls = ls + " & rat['genres'].str.contains(\"Thriller\")"

				if ani:
					ls = ls + " & rat['genres'].str.contains(\"Animation\")"
				with col2:

					exec("st.write(rat[" + ls + "].sort_values(by=['rating'], ascending=False)[['title','genres']])")





	elif selection == "Development team":
		st.markdown("<h2 style=color:#3FBEBF;>Meet Our Team</h2>", unsafe_allow_html=True)
		st.title("")
		col1, mid, col2 = st.columns([80, 10, 80])
		with col2:
			st.subheader("Fabian Brijlal - Director")
			st.write(
				"Fabian has worked as a Project Manager, Product Manager, Systems and Production developer. When he is not coding he enjoys watching sport on television.")
		with col1:
			st.image('fabian.png', width=200)
		col1, mid, col2 = st.columns([80, 10, 80])
		with col1:
			st.subheader("Angela Morris - Dep Director")
			st.write(
				"Angela has worked as a Product Manger and has worked with many startups. On her spare time she enjoys reading novels.")
		with col2:
			st.image('angela.jpg', width=200)
		col1, mid, col2 = st.columns([80, 10, 80])
		with col2:
			st.subheader("Malesela M. Mohlake - Senior Data Analyst")
			st.write(
				"Mr Mohlake has worked as a lead data analyst for multiple F.A.N.G. companies. When he's not working he enjoys spending time with his family and friends.")
		with col1:
			st.image('mucus.jpg', width=200)
		col1, mid, col2 = st.columns([80, 10, 80])
		with col1:
			st.subheader("Lungelo Ndlovu - Senior Data Analyst")
			st.write(
				"Lungelo has worked as researcher for MIT in statistics. Lungelo enjoys turtoring young aspiring data sciensist and data analyst alike.")
		with col2:
			st.image('lungelo.jpg', width=200)
		col1, mid, col2 = st.columns([80, 10, 80])
		with col2:
			st.subheader("Lindokuhle Mtshali - Server Engineer")
			st.write(
				"Lindokuhle has worked as a back-end itern for multiple companies. On his spare time he enjoys tv shows.")
		with col1:
			st.image('me.jpeg', width=200)


	elif selection == 'About Us':
		st.markdown("<h2 style=color:#3FBEBF;>About Us</h2>", unsafe_allow_html=True)
		st.title("")

		col1, mid, col2 = st.columns([80, 10, 80])
		with col1:
			st.image('logo.png')
		with col2:
			st.markdown("<h3 style=color:#3FBEBF;>Data Lens Corp</h3>", unsafe_allow_html=True)
			st.write("")
			st.write(
				"Data Lens Corp was foundend in 2015 by Fabian Brijlal. The aim of the company is to help companies and organisations make sense of their data, this helps them make informed decisions based on data.")
		st.title("")
		st.markdown("<h3 style=color:#3FBEBF;>Our Vision</h3>", unsafe_allow_html=True)
		st.write("")
		st.write(
			"We have been working with small busineses to help them transist to a digital world so as to effectively collect data on their customers. In our Attempt to bridge the digital gap between the low income class and the high income class, we have donated over 15 computer labs to under priveledged schools.")



	elif selection == 'Contact Us':
		col1, mid, col2 = st.columns([80, 80, 80])
		with col1:
			st.write("")
		with mid:
			st.markdown("<h3 style=color:#3FBEBF;>Contact Us</h3>", unsafe_allow_html=True)
		col1, mid, col2 = st.columns([80, 10, 80])
		with col1:
			st.markdown("<h4 style=color:#3FBEBF;>Durban Offices</h4>", unsafe_allow_html=True)

			st.write(":phone: (+27) 23 456 7890")
			st.write(":email: durbandatalens@dlc.ac.za")
			st.write(":round_pushpin: Durban Central, West Street block c, building 200")

		with col2:
			st.markdown("<h4 style=color:#3FBEBF;>Cape Town Offices</h4>", unsafe_allow_html=True)

			st.write(":phone: (+27) 23 456 7890")
			st.write(":email: cptnandatalens@dlc.ac.za")
			st.write(":round_pushpin: Cape Town City, Kloof Street, building 3 on block F")



if __name__ == '__main__':
	main()
