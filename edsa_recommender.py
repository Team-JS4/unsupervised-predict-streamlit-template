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
#from streamlit_lottie import st_lottie
# Data handling dependencies

from streamlit_option_menu import option_menu

import json
import requests

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#def load_lottieurl(filepath: str):
    #r = requests.get(url)
    #if r.status_code != 200:
        #return None
    #return r.json()




import pandas as pd
from pandas import Timestamp
import numpy as np
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

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based1 import collab_model
from recommenders.content_based import content_model

# Data Loading


title_list = load_movie_titles('resources/data/movies.csv')

best_director_list = ['Quentin Tarantino', 'Michael Crichton', 'J.R.R. Tolkien', 'Lilly Wachowski', 'Stephen King', 'Ethan Coen', 'James Cameron', 'Luc Besson'
                        , 'Jonathan Nolan', 'Thomas Harris']
number_of_ratings_list = [109919, 65157, 62963, 60988, 59903, 51185, 51178, 44015, 42645, 36425]
title_cast_index_list = ['Samuel L. Jackson', 'Steve Buscemi', 'Keith David', 'Willem Dafoe', 'Robert De Niro', 'Brian Cox', 'Christopher Walken', 'Gérard Depardieu'
                         , 'Bruce Willis', 'Danny Glover', 'Morgan Freeman', 'Peter Stormare', 'Alec Baldwin', 'Nicolas Cage', 'Stanley Tucci', 'Julianne Moore'
                         , 'Richard Jenkins', 'Susan Sarandon', 'Stellan Skarsgård', 'John Goodman', 'Woody Harrelson', 'Tom Wilkinson', 'Antonio Banderas'
                         ,'Christopher McDonald', 'Val Kilmer', 'Jeff Bennett', 'Johnny Depp', 'Ed Harris', 'Donald Sutherland', 'John Leguizamo', 'Forest Whitaker'
                         , 'Harvey Keitel', 'John Cusack', 'Ray Liotta', 'Paul Giamatti', 'Luis Guzmán', 'Stephen Tobolowsky', 'George W. Bush', 'David Strathairn'
                         , 'Danny Trejo', 'Jim Broadbent', 'John Malkovich', 'Richard Riehle', 'Ewan McGregor', 'Kathy Bates', 'Robert Downey Jr.', 'Ving Rhames'
                         , 'Patricia Clarkson', 'Jim Cummings', 'William H. Macy', 'Billy Bob Thornton', 'Michael Gambon', 'Eric Roberts', 'Ben Kingsley'
                         , 'Robin Williams', 'William Hurt', 'Ron Perlman', 'Frank Welker', 'John Hurt', 'John Turturro']
title_cast_series_list = [83, 68, 61, 59, 58, 57, 57, 57, 56, 56, 56, 55, 55, 55, 54, 54, 54, 54, 53, 53, 53, 53, 52, 52, 52, 51, 50, 50, 49, 49, 49, 49, 49, 49, 48
                          , 48, 48, 47, 47, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46, 46, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45]
genres_index_list = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Documentary', 'Crime', '(no genres listed)', 'Adventure', 'Sci-Fi', 'Children'
                     , 'Animation', 'Mystery', 'Fantasy', 'War', 'Western', 'Musical', 'Film-Noir', 'IMAX']
genres_series_list = [25606, 16870, 8654, 7719, 7348, 5989, 5605, 5319, 5062, 4145, 3595, 2935, 2929, 2925, 2731, 1874, 1399, 1054, 353, 195]
movie_year_series_list = [2513, 2488, 2406, 2374, 2173, 2034, 1978, 1838, 1724, 1691, 1632, 1498, 1446, 1255, 1172, 1028, 1024, 994, 971, 929]
movie_year_index_list = ['2015', '2016', '2014', '2017', '2013', '2018', '2012', '2011', '2009', '2010', '2008', '2007', '2006', '2005', '2004', '2003', '2002', '2019', '2001', '2000']
director_series_list = [28, 26, 26, 24, 19, 17, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 12, 11, 11]
director_index_list = ['See full summary', 'Woody Allen', 'Luc Besson', 'Stephen King', 'William Shakespeare', 'Ki-duk Kim', 'Lars von Trier', 'Tyler Perry'
                       , 'Alex Gibney', 'Robert Rodriguez', 'Takeshi Kitano', 'Peter Farrelly', 'David Mamet', 'Kevin Smith', 'Olivier Assayas', 'Sang-soo Hong'
                       , 'Clive Barker', 'Charles Band', 'John Sayles', 'Akira Toriyama']
rating_series_list = [2652977, 1959759, 1445230, 1270642, 880516, 656821, 505578, 311213, 159731, 157571]
rating_index_list = [4.0, 3.0, 5.0, 3.5, 4.5, 2.0, 2.5, 1.0, 1.5, 0.5]

rmse = [1.0054, 1.009, 1.0091, 0.9981, 0.9949, 1.007, 1.0103, 1.0036]
algorithm = ["KNN Basic", "KNN Means", "KNN ZScore", "SVD", "SVDpp", "NMF", "SlopeOne", "CoClustering"] 

mae = [0.8024, 0.8049, 0.8047, 0.7951, 0.7924, 0.8038, 0.8061, 0.8014]
algorithms = ["KNN Basic", "KNN Means", "KNN ZScore", "SVD", "SVDpp", "NMF", "SlopeOne", "CoClustering"]

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
               
        selected = option_menu(
            menu_title = None,
            options = ["Company", "Team", "EDA", "Netflix & SABC partnership", "Overview", "Models", "Terms & conditions"],
            icons = ["cash-coin", "person", "graph-up-arrow", "book", "cloud", "gear", "hammer",  "wrench-adjustable"],#https://icons.getbootstrap.com/
            orientation = "horizontal"
        )   
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
        from PIL import Image
        train = pd.read_csv("resources/data/ratings.csv")
        movies = pd.read_csv("resources/data/movies.csv")
        hello_lottie = load_lottiefile("resources/imgs/20942-hello-boy.json")

        if selected == "Company":
                
            st.title("Acquire Data")
            image = Image.open('resources/imgs/ttttt.png')
            st.image(image, caption=None,width=800)
            if st.checkbox('Show Acquire data history, goals & clients'):
                
                st.markdown("# WHAT WE DO")
                st.markdown("Acquire Data Corporation is a South African Enterprise IT Service provider. Acquire Data was established in 2022. We are a Software company that provides top data analytics products.")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("# VISIT US")
                st.markdown("## PRETORIA OFFICE")
                st.markdown("### PH4467 Phomolong")
                st.markdown("### Atteridgeville")
                st.markdown("### 0008")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("# CALL US")
                st.markdown("### 079 919 4716")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("### © Copyright 2022 | All Rights Reserved  | Powered By Mamilodi Agency")
                
        
       
        if selected == "Terms & conditions":
                
            st.title("Streamlit Recommender terms & conditions")
            image = Image.open('resources/imgs/terms-conditions-340x250.png')
            st.image(image, caption=None,width=700)
           
            if st.checkbox('Show Streamlit Recommender terms & conditions'):
                
                st.markdown("* Kindly follow the instructions")
                st.markdown("* Choose from the options given to you")
                st.markdown("* Do not alter or hack the code")
                st.markdown("* 8 gig RAM recommended")
                            
            if st.checkbox('Show Streamlit  terms & conditions'):                   
                image = Image.open('resources/imgs/0_7mUI9yTv9TUXCco3.png')
                st.image(image, caption='streamlit',width=700)
                st.markdown("## Terms of use")
                st.markdown("### Hello and thanks for using Streamlit!")
                st.markdown("Please read below for the terms that cover the usage of the Streamlit website, the Streamlit community forum, and Components submitted by the Streamlit community.")
                st.markdown("The Streamlit library is open sourced under the Apache 2.0 license. You can find instructions to download the Streamlit software **in the documentation** and find its software license on the streamlit **github page**")
                st.markdown("The Streamlit Cloud platform has its own terms of service. You can sign up for the platform here and find the **software terms for Streamlit Cloud here**.")
                st.markdown("### Terms of use for the Streamlit website and support services")
                st.markdown('Please read these Terms of Use (the “Terms”) carefully because they are a legal agreement that governs the website of Streamlit, Inc. (“Streamlit”) located at http://streamlit.io (the “Site”), the community support forum offered through the Site and the User Components (collectively, the “Support Services”).')
                st.markdown("**By using the Support Services, which may include accessing and using User Components in connection with Streamlit, you agree to be bound by these Terms and the Privacy Policy referenced in Section 1 below. If you don’t agree to be bound by these Terms and the Privacy Policy, do not use the Support Services.**")
                st.markdown("### 1. Privacy policy")
                st.markdown("Please refer to our Privacy Policy at **http://streamlit.io/privacy-policy** for information on how we collect and disclose information from our users, what data we store, and how we use that data. You agree that your use of Streamlit and the Support Services is subject to our Privacy Policy.")
                st.markdown("### 2. Changes to terms")
                st.markdown("We may update these Terms at any time, in our sole discretion. If we do so, we’ll let you know either by posting the updated Terms on the Site or through other communications. It’s important that you review the Terms whenever we update them or you use the Site. If you continue to use the Site after we have posted updated Terms, you are agreeing to be bound by the updated Terms. If you don’t agree to be bound by the updated Terms, then you may not use the Support Services anymore.")
                st.markdown("### 3. License to Use the support services")
                st.markdown("Provided that you are 18 years or older and capable of forming a binding contract with Streamlit and are not barred from using the Support Services under applicable law, and conditioned upon your compliance with these Terms, Streamlit grants you a limited, non-exclusive, non-transferable, non-sublicensable license to use the Support Services. Use of the community support forum offered through the Site requires that you create an account on the Site (“Account”). It’s important that you provide us with accurate, complete and up-to-date information for your Account. You agree to update such information to keep it accurate, complete and up-to-date. If you don’t, we might have to suspend or terminate your Account. You agree that you won’t disclose your Account password to anyone and you’ll notify us immediately of any unauthorized use of your Account. You’re responsible for all activities that occur under your Account, whether or not you know about them.")
                st.markdown("### 4. Feedback")
                st.markdown("We welcome feedback, comments and suggestions for improvements to the Support Services and the Software (“Feedback”). You can submit Feedback by posting on our community forum at https://discuss.streamlit.io, submitting feedback on GitHub at https://github.com/streamlit, or by emailing us at product@streamlit.io. You grant to us a non-exclusive, transferable, worldwide, perpetual, irrevocable, fully-paid, royalty-free license, with the right to sublicense, under any intellectual property rights that you own or control to use, copy, modify, create derivative works based upon and otherwise use the Feedback for any purpose.")
                st.markdown("### 5. User content and usage data")
                st.markdown("#### 5.1. User content")
                st.markdown("“User Content” means any text, graphics, images, software, works of authorship, and information or other materials that you provide to be made available through the Support Services, but excludes Submitted Components. For example, a screenshot of an app you made that you posted onto the Streamlit forum. Streamlit does not claim any ownership rights in any User Content and nothing in these Terms will restrict any rights that you may have to use and exploit your User Content. By making any User Content available through the Support Services you grant to us a non-exclusive, transferable, worldwide, royalty-free license, with the right to sublicense, to use, copy, modify, create derivative works based upon, distribute, publicly display, and publicly perform your User Content in connection with operating and providing the Support Services. Submitted Components are governed by the **Components Addendum**.")
                st.markdown("#### 5.2. Your responsibility for user content ")
                st.markdown("You are solely responsible for all your User Content. You represent and warrant that you own all your User Content or you have all rights that are necessary to grant us the license rights in your User Content under these Terms, and that neither your User Content, nor your use and provision of your User Content to be made available through the Support Services, nor any use of your User Content by Streamlit on or through the Support Services will infringe, misappropriate or violate a third party’s intellectual property rights, or rights of publicity or privacy, or result in the violation of any applicable law or regulation.")
                st.markdown("#### 5.3. Removal of user content")
                st.markdown("You can remove your User Content by specifically deleting it. However, in certain instances, some of your User Content (such as posts or comments you make on the community support forum) may not be completely removed and copies of your User Content may continue to exist on the Support Services. We are not responsible or liable for the removal or deletion of (or the failure to remove or delete) any of your User Content.")
                st.markdown("### 6. User components")
                st.markdown("1. The Streamlit software (the “Software”) enables you to engage and use third-party code, components, widgets and tools (collectively, “User Components”) when using the Software. These User Components allow you to write your own JavaScript and HTML components that can be rendered in Streamlit apps, receive data from apps, and send data to Streamlit Python scripts. You may create User Components for your own use, or choose to share them with the Streamlit Community. These terms apply to User Components that have been shared within the Streamlit Community via the Support Services.")
                st.markdown("2. You acknowledge and agree that regardless of how User Components may be offered to you, Streamlit is only acting as an intermediary between you and the third party (which may include other users of the Software) offering such User Components. Streamlit does not, in any way, endorse any User Components and shall not be in any way responsible or liable with respect to any such User Components. Your relationship with the User Components and any terms governing your use of User Components is subject to a separate agreement between you and the provider of a User Components (“Third Party Agreements”). Streamlit is not party to, or responsible, in any manner, for your or the provider of the User Component’s compliance with Third Party Agreements.")
                st.markdown("3. If you choose to share your User Component with Streamlit or to upload any User Component to the Support Services (a “Submitted Component”) you agree to the terms and conditions of the Submitted Component Addendum, the terms of which are hereby incorporated by reference.")
                st.markdown("4. Streamlit and the providers of each User Components reserve the right to discontinue the use or suspend the availability of any User Component, for any reason and with no obligation to provide any explanation or notice. Such discontinuation may result in the inability to use certain features and actions of a User Component along with the Software.")
                st.markdown("### 7. General prohibitions and our enforcement rights")
                st.markdown("In connection with your use of the Support Services, you agree not to do any of the following:")
                st.markdown("1. Post, upload, publish, submit or transmit any User Content or other materials that: (i) infringe, misappropriate or violate a third party’s patent, copyright, trademark, trade secret, moral rights or other intellectual property rights, or rights of publicity or privacy; (ii) violate, or encourage any conduct that would violate, any applicable law or regulation or would give rise to civil liability; (iii) are fraudulent, false, misleading or deceptive; (iv) are defamatory, obscene, pornographic, vulgar or offensive; (v) promote discrimination, bigotry, racism, hatred, harassment or harm against any individual or group; (vi) are violent or threatening or promote violence or actions that are threatening to any person or entity; or (vii) promote illegal or harmful activities or substances;")
                st.markdown("2. Access, tamper with, or use non-public areas of the Support Services, Streamlit’s computer systems, or the technical delivery systems of Streamlit’s providers;")
                st.markdown("3. Attempt to probe, scan or test the vulnerability of the Support Services, any system or network of Streamlit or its providers, or breach any security or authentication measures of Streamlit or its providers;")
                st.markdown("4. Avoid, bypass, remove, deactivate, impair, descramble or otherwise circumvent any technological measure implemented by Streamlit or any of Streamlit’s providers or any other third party (including another user) to protect the Support Services, or the system or network of Streamlit or its providers;")
                st.markdown("5. Attempt to access or search the Support Services or download materials from the Support Services through the use of any engine, software, tool, agent, device or mechanism (including spiders, robots, crawlers, data mining tools or the like) other than the software and/or search agents provided by Streamlit or other generally available third-party web browsers;")
                st.markdown("6. Interfere with, or attempt to interfere with, the access of any user, host or network, including, but not limited to, sending a virus, overloading, flooding, spamming, or mail-bombing the Support Services or the system or network of any Streamlit provider;")
                st.markdown("7. Use the Support Services or any of its component features for competitive analysis or benchmarking, or to otherwise develop, sell, license, commercialize or contribute to the development or commercialization of any product or service that could, directly or indirectly, compete with the Support Services or any other product or service of Streamlit;")
                st.markdown("8. Impersonate or misrepresent your affiliation with any person or entity;")
                st.markdown("9. Violate any applicable law or regulation; or")
                st.markdown("10. Encourage or enable any other individual to do any of the foregoing.")
                st.markdown("Although we’re not obligated to monitor access to or use of the Support Services or to review or edit any User Content, we have the right to do so for the purpose of operating or protecting the Support Services, to ensure compliance with these Terms and to comply with applicable law or other legal requirements. We reserve the right, but are not obligated, to remove or disable access to any User Content, at any time and without notice, including, but not limited to, if we, at our sole discretion, consider any Content to be objectionable or in violation of these Terms. We have the right to investigate violations of these Terms or conduct that affects the Support Services. We may also consult and cooperate with law enforcement authorities to prosecute users who violate the law.")
                st.markdown("### 8. DMCA/copyright policy")
                st.markdown("We respect copyright law and the intellectual property rights of others, and we expect our users to do the same. It is our policy to terminate in appropriate circumstances Account holders who repeatedly infringe or are believed to be repeatedly infringing the rights of copyright holders. Please see Streamlit’s Copyright and IP Policy at https://streamlit.io/copyright-policy, for further information.")
                st.markdown("### 9. Links to third party websites or resources")
                st.markdown("The Support Services may contain links to third-party websites or resources. We provide these links only as a convenience and are not responsible for the content, products or services on or available from those websites or resources or links displayed on such websites. You acknowledge sole responsibility for and assume all risk arising from your use of any third-party websites or resources.")
                st.markdown("### 10. Termination")
                st.markdown("We may terminate or suspend your access to and use of the Support Services if you violate these Terms, if necessary for us to conduct maintenance on or implement updates to the Support Services, or if we discontinue the existing version of the Support Services, at any time and without notice to you. You may cancel your Account at any time by sending an email to us at legal@streamlit.io. Upon any termination, discontinuation or cancellation of the Support Services or your Account, the following Sections will survive: [6.1-6.3, 6.4] (last sentence), and [11-14].")
                st.markdown("### 11. Warranty disclaimers")
                st.markdown("TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, NEITHER STREAMLIT NOR ANY OTHER PARTY INVOLVED IN CREATING, PRODUCING, OR DELIVERING THE SUPPORT SERVICES WILL BE LIABLE FOR DAMAGES OF ANY KIND, INCLUDING BUT NOT LIMITED TO ANY INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, OR DAMAGES FOR LOST PROFITS, LOST REVENUES, LOST SAVINGS, LOST BUSINESS OPPORTUNITY, LOSS OF DATA OR GOODWILL, SERVICE INTERRUPTION, COMPUTER DAMAGE OR SYSTEM FAILURE OR THE COST OF SUBSTITUTE PRODUCTS OR SERVICES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH THESE TERMS OR FROM THE USE OF OR INABILITY TO USE THE SUPPORT SERVICES, WHETHER BASED ON WARRANTY, CONTRACT, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER LEGAL THEORY, AND WHETHER OR NOT STREAMLIT OR ANY OTHER PARTY HAS BEEN INFORMED OF THE POSSIBILITY OF SUCH DAMAGE, EVEN IF A LIMITED REMEDY SET FORTH HEREIN IS FOUND TO HAVE FAILED OF ITS ESSENTIAL PURPOSE. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OR LIMITATION OF LIABILITY FOR CONSEQUENTIAL OR INCIDENTAL DAMAGES, SO THE ABOVE LIMITATION MAY NOT APPLY TO YOU. THE PARTIES HAVE AGREED THAT THESE LIMITATIONS WILL SURVIVE AND APPLY EVEN IF ANY LIMITED REMEDY SPECIFIED IN THESE TERMS IS FOUND TO HAVE FAILED OF ITS ESSENTIAL PURPOSE.")
                st.markdown("### 12. Disclaimers with respect to user components")
                st.markdown("WITHOUT LIMITING SECTION 11, THE USER COMPONENTS MADE AVAILABLE BY STREAMLIT ARE PROVIDED ON AN “AS-IS”, “WITH ALL FAULTS” AND “AS AVAILABLE” BASIS WITHOUT WARRANTIES OF ANY KIND. WITHOUT LIMITING SECTION 11, WE HEREBY DISCLAIM ANY AND ALL REPRESENTATIONS AND WARRANTIES OF ANY KIND, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FUNCTIONALITY, TITLE, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT, WHETHER EXPRESS, IMPLIED OR STATUTORY.")
                st.markdown("### 13. Limitation of liability")
                st.markdown("TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, NEITHER STREAMLIT NOR ANY OTHER PARTY INVOLVED IN CREATING, PRODUCING, OR DELIVERING THE SUPPORT SERVICES WILL BE LIABLE FOR DAMAGES OF ANY KIND, INCLUDING BUT NOT LIMITED TO ANY INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, OR DAMAGES FOR LOST PROFITS, LOST REVENUES, LOST SAVINGS, LOST BUSINESS OPPORTUNITY, LOSS OF DATA OR GOODWILL, SERVICE INTERRUPTION, COMPUTER DAMAGE OR SYSTEM FAILURE OR THE COST OF SUBSTITUTE PRODUCTS OR SERVICES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH THESE TERMS OR FROM THE USE OF OR INABILITY TO USE THE SUPPORT SERVICES, WHETHER BASED ON WARRANTY, CONTRACT, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER LEGAL THEORY, AND WHETHER OR NOT STREAMLIT OR ANY OTHER PARTY HAS BEEN INFORMED OF THE POSSIBILITY OF SUCH DAMAGE, EVEN IF A LIMITED REMEDY SET FORTH HEREIN IS FOUND TO HAVE FAILED OF ITS ESSENTIAL PURPOSE. WITHOUT LIMITING THE FOREGOING, STREAMLIT HAS NO RESPONSIBILITY OR LIABILITY FOR USER COMPONENTS OR FOR ANY ACTS OR OMISSIONS BY THIRD PARTIES, INCLUDING WITHOUT LIMITATION, THE OPERABILITY OR INTEROPERABILITY OF SUCH USER COMPONENTS WITH THE SOFTWARE, OR THE SECURITY, ACCURACY, RELIABILITY, DATA PROTECTION AND PROCESSING PRACTICES OF THE PROVIDERS OF ANY USER COMPONENTS. BY ACCESSING OR USING USER COMPONENTS IN CONNECTION WITH THE SOFTWARE, YOU ACKNOWLEDGE THAT YOUR ACCESS AND USE OF USER COMPONENTS ARE AT YOUR SOLE DISCRETION AND RISK. YOU ARE SOLELY RESPONSIBLE FOR VERIFYING THE OPERATION AND PRACTICES OF THE PROVIDER OF USER COMPONENTS AND ITS RESPECTIVE THIRD PARTY AGREEMENT. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OR LIMITATION OF LIABILITY FOR CONSEQUENTIAL OR INCIDENTAL DAMAGES, SO THE ABOVE LIMITATION MAY NOT APPLY TO YOU. THE PARTIES HAVE AGREED THAT THESE LIMITATIONS WILL SURVIVE AND APPLY EVEN IF ANY LIMITED REMEDY SPECIFIED IN THESE TERMS IS FOUND TO HAVE FAILED OF ITS ESSENTIAL PURPOSE.")
                st.markdown("### 14. General terms")
                st.markdown("#### 14.1. Entire agreement")
                st.markdown("These Terms, including the Privacy Policy, constitute the entire and exclusive understanding and agreement between Streamlit and you regarding the Support Services, and these Terms supersede and replace any and all prior oral or written understandings or agreements between Streamlit and you regarding the same. If any provision of these Terms is held invalid or unenforceable by a court of competent jurisdiction, that provision will be enforced to the maximum extent permissible and the other provisions of these Terms will remain in full force and effect. **You may not assign or transfer these Terms, by operation of law or otherwise, without Streamlit’s prior written consent. Any attempt by you to assign or transfer these Terms, without such consent, will be void. Streamlit may freely assign or transfer these Terms without restriction**. Subject to the foregoing, these Terms will bind and inure to the benefit of the parties, their successors and permitted assigns.")
                st.markdown("#### 14.2. Notices")
                st.markdown("Any notices or other communications provided by Streamlit under these Terms, including those regarding modifications to these Terms, will be given: (i) via email; or (ii) by posting to the Support Services. For notices made by e-mail, the date of receipt will be deemed the date on which such notice is transmitted.")
                st.markdown("#### 14.3. Waiver of rights")
                st.markdown("Streamlit’s failure to enforce any right or provision of these Terms will not be considered a waiver of such right or provision. The waiver of any such right or provision will be effective only if in writing and signed by a duly authorized representative of Streamlit. Except as expressly set forth in these Terms, the exercise by either party of any of its remedies under these Terms will be without prejudice to its other remedies under these Terms or otherwise.")
                st.markdown("### 15. Contact information")
                st.markdown("If you have any questions about these Terms or the Support Services, please contact Streamlit at **legal@streamlit.io**")
                st.markdown("### Addendum: Components")
                st.markdown("Hello and thanks for making a custom component for Streamlit! Your contributions greatly extend what we can all do with Streamlit, and we really appreciate your contribution and general support of Streamlit.")
                st.markdown("Please read this addendum for the terms that govern your submission, or otherwise making available, of User Components created by you to us.")
                st.markdown("By submitting User Components to Streamlit through the Support Services, you agree to be bound by the terms of this addendum. If you don’t agree to be bound by the terms of this addendum, please do not submit or send any User Components to us.")
                st.markdown("If you have any questions about these Terms or the Support Services, please contact Streamlit at **legal@streamlit.io**")
                st.markdown("### A. Streamlit terms of use")
                st.markdown("This addendum hereby incorporates by reference the Streamlit Terms of Use listed above.")
                st.markdown("### B. Submitted components")
                st.markdown("1. A “Submitted Component” is defined in the Streamlit Terms of Use, and means any User Component that is uploaded to the Support Services or otherwise made available to Streamlit.")
                st.markdown("2. As between you and Streamlit, you will be the owner of all intellectual property rights in and to User Components created by you. Without limiting the foregoing, with respect to all Submitted Components you submit to Streamlit, you hereby grant, and agree to grant, to Streamlit a non-exclusive, transferable, worldwide, perpetual, irrevocable, fully-paid, royalty-free license, with the right to sublicense, to use, copy, modify, distribute, publicly display, publicly perform and otherwise use such Submitted Component in connection with the provision of Streamlit’s products and services. You hereby agree to perform such further acts and execute such further documents as requested by Streamlit to carry out the intent of this provision.")
                st.markdown("3. By providing any Submitted Component to Streamlit or uploading any Submitted Component to the Support Services, you represent and warrant to us that you have sufficient rights in any such Submitted Component to grant Streamlit the licenses described hereunder. You further represent and warrant your submission of that Submitted Component, and your granting of rights in such Submitted Component hereunder will infringe, misappropriate or violate a third party’s intellectual property rights, or result in the violation of any applicable contract, law or regulation.")
                st.markdown("4. In some cases, Submitted Components may be considered for incorporation by us into the Software. You understand and agree that we are not obligated to use, distribute or continue to distribute any Submitted Component submitted by you and we reserve the right, but not the obligation, to restrict or remove Submitted Components for any reason.")
                st.markdown("5. Both you and Streamlit reserves the right to discontinue the use or suspend the availability of any User Component, for any reason and with no obligation to provide any explanation or notice. Such discontinuation may result in the inability to use certain features and actions of a User Component along with the Software.")
                           
                            
        if selected == "Team":
                
            st.title("TEAM JS4")
            
            image = Image.open('resources/imgs/Team-PNG-Pic.png')
            st.image(image, caption=None,width=600)
            
            if st.checkbox('Show team members'):                   
                image = Image.open('resources/imgs/IMG-20220707-WA0072.jpg')
                st.image(image, caption='CEO',width=500)
                image = Image.open('resources/imgs/5026eb40-a23e-4c2f-a9a4-c2a5a87bc522.jpg')
                st.image(image, caption='Big Data Engineer',width=500)
                image = Image.open('resources/imgs/28dce6c3-e238-4318-9ee6-eefb87d1dbb6.jpg')
                st.image(image, caption='Business Intelligence Analyst',width=500)
                image = Image.open('resources/imgs/72b224de-7528-4ec2-9908-b5f47fe3ca84.jpg')
                st.image(image, caption='Solutions Architect',width=500)
                image = Image.open('resources/imgs/e7a321d6-7698-4762-a37c-ce7fa65470c9.jpg')
                st.image(image, caption='Data Analyst',width=500)
                image = Image.open('resources/imgs/IMG_20220719_150430.jpg')
                st.image(image, caption='Machine Learning Engineer',width=400)

        if selected == "Overview":
            st.title("OVERVIEW")
            
            image = Image.open('resources/imgs/Team-Work-Transparent-PNG-1.png')
            st.image(image, caption=None,width=600)
            if st.checkbox('Overview'):
                st.info("General Overview")
                st.markdown("### Data Description")
                st.markdown("This dataset consists of several million 5-star ratings obtained from users of Netflix service. The Netflix dataset has long been used by industry , and now we get it!")
                st.markdown("In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.")
                st.markdown("With this context, **Acquire Data** is challenging us to construct a recommendation algorithm based on content and collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.")
                st.markdown("Providing an accurate and robust solution to this challenge has immense economic potential for **Netflix**, with users of the system being personalised recommendations - generating platform affinity for the streaming services which best facilitates their audience's viewing.")
                st.markdown("#### Raw data")
                st.info("Ratings")
                st.write(train[['userId', 'movieId', 'rating', 'timestamp']])
                st.info("Movies")
                st.write(movies[['movieId', 'title', 'genres']])
        if selected == "EDA":
            st.title("EXPLORATORY DATA ANALYSIS")
            image = Image.open('resources/imgs/Blue_bar_graph.png')
            st.image(image, caption=None,width=600)
            if st.checkbox('Show EDA'):
                
              
                
                st.info("These are the plots displaying distrinutions of the datasets")
    
               
                                    
                            
                            

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
                            st.markdown("* Quentin Tarantino is the best director in terms of ratings, followed by Michael Chrichton, J.R.R Tolkien, Lilly Wachowski Stephen King.")
                            st.markdown("* All the remaining directors have ratings below 60 000.")
                            
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
                            st.markdown("The mode ranking in the chart above is 4.0. It appears that users tend to provide whole-number ratings (e.g. 4.0) as opposed to fractional ratings (e.g. 3.5). The mean rating of approx 3.5 shows that users typically lean towards more favourable ratings and are not excessively harsh in their criticism. There are few ratings below 2.0., it could be that users tend to only rate those movies that they enjoy. Many users might terminate an unenjoyable movie mid-stream without bothering to give it a rating.")
                                    
                            
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
                                st.markdown("* Seefull summary has directed the most movies, followed by Luc Besson and Woody Allen in second place in second place.")
                                st.markdown("* Stephen King is 3rd, all the other directors have less than 20 movies.")

                            
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
                            st.markdown("Drama is the most common genre, meaning that most movies are drama films, then Comedy followed by Thriller and more")
                            st.markdown("IMAX is the least common genre since it is the most expensive so few people watch it, it is only available at the cinema and not streaming platforms such as Netflix.")
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
                if st.checkbox("Show Top 20 movie years graph"):
                    movie_year_list2 = st.multiselect("Which year would you like to see?", movie_year_index_list)
                    movie_year_emp_list = []
                    if len(movie_year_list2) == 0:
                        st.subheader("Top 20 movie years graph")
                        source = pd.DataFrame({
                            'Number of movies': movie_year_series_list,
                            'Year': movie_year_index_list
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Number of movies:Q',
                            x="Year:O"
                        ).properties(width=500,height=500)
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        if st.checkbox('Show Top Top 20 movie years graph interperetation'):
                            st.markdown("**Looking at the above bar graph:**")
                            st.markdown("* We see that 2015 has the most movies released, followed by the year 2016")
                            st.markdown("* From 2000 up to 2015 there's a gradual increase in number of movies, then from 2016 to 2019, there's a drastic drop in number of movies.")

                                    
                    else:
                            for i in movie_year_list2:
                                movie_year_emp_list.append(movie_year_dict[i])
                            
                            st.subheader("Top 20 movie years graph")
                            
                            source = pd.DataFrame({
                            'Number of movies': movie_year_emp_list, 'Year': movie_year_list2 })

                            bar_chart = alt.Chart(source).mark_bar().encode(
                                    y='Number of movies:Q',
                                x="Year:O"
                            )
                            st.altair_chart(bar_chart, use_container_width=True) 
        if selected == "Netflix & SABC partnership":
            st.title("NETFLIX & SABC partnership")
            
            image = Image.open('resources/imgs/Hand-Shake-PNG-File.png')
            st.image(image, caption=None,width=800)
            st.markdown("The SABC has gone into a partnership with streaming giant Netflix 14 Billion dollar project.")
            st.markdown("NETFLIX would exchange data with the SABC, with the goal of bringing a richer, tailored experience for the SouthAfrican market.")
            
            
            
           
                
                
                
            if st.checkbox("Show SABC Brief History"):
                    image = Image.open('resources/imgs/South_African_Broadcasting_Corporation_logo.svg.png')
                    st.image(image, caption=None,width=800)
                    st.markdown('Established in August 1936, The South African Broadcasting Corporation (SOC) Limited (“The SABC”) is a public broadcaster (Schedule 2: “Major Public Entity” in terms of the Public Finance Management Act No. 1 of 1999, “PFMA”), with a mandate to inform, educate and entertain the public of South Africa. Founded on the statute of The Broadcasting Act (The Act) the SABC’s obligations are captured in the Independent Communications Authority of South Africa (“ICASA”) Regulations and license conditions, providing five television channels and 18 Radio stations.')
                    
            
                
                
                
            if st.checkbox("Show Netflix Brief History"):
                image = Image.open('resources/imgs/18-182387_logo-netflix-3d-hdvector-illustrator-lockheed-martin-logo.png')
                st.image(image, caption=None,width=800)
                st.subheader("Netflix American company")
                st.markdown("Netflix, Inc. is an American subscription streaming service and production company. Launched on August 29, 1997, it offers a film and television series library through distribution deals as well as its own productions, known as Netflix Originals.")
                st.markdown("As of March 31, 2022, Netflix had over 221.6 million subscribers worldwide, including 74.6 million in the United States and Canada, 74.0 million in Europe, the Middle East and Africa, 39.9 million in Latin America and 32.7 million in Asia-Pacific. It is available worldwide aside from Mainland China, Syria, North Korea, and Russia. Netflix has played a prominent role in independent film distribution, and it is a member of the Motion Picture Association (MPA).")
                st.markdown("Netflix can be accessed via web browsers or via application software installed on smart TVs, set-top boxes connected to televisions, tablet computers, smartphones, digital media players, Blu-ray players, video game consoles and virtual reality headsets on the list of Netflix-compatible devices. It is available in 4K resolution. In the United States, the company provides DVD and Blu-ray rentals delivered individually via the United States Postal Service from regional warehouses.")
                st.markdown("Netflix was founded on the aforementioned date by Reed Hastings and Marc Randolph in Scotts Valley, California. Netflix initially both sold and rented DVDs by mail, but the sales were eliminated within a year to focus on the DVD rental business. In 2007, Netflix introduced streaming media and video on demand. The company expanded to Canada in 2010, followed by Latin America and the Caribbean. Netflix entered the content-production industry in 2013, debuting its first series House of Cards. In January 2016, it expanded to an additional 130 countries and then operated in 190 countries.")
                st.markdown("The company is ranked 115th on the Fortune 500 and 219th on the Forbes Global 2000. It is the second largest entertainment/media company by market capitalization as of February, 2022. In 2021, Netflix was ranked as the eighth-most trusted brand globally by Morning Consult. During the 2010s, Netflix was the top-performing stock in the S&P 500 stock market index, with a total return of 3,693%.")
                st.markdown("Netflix is headquartered in Los Gatos, California, in Santa Clara County, with the two CEOs, Hastings and Ted Sarandos, split between Los Gatos and Los Angeles, respectively. It also operates international offices in Asia, Europe and Latin America including in Canada, France, Brazil, Netherlands, India, Italy, Japan, South Korea and the United Kingdom. The company has production hubs in Los Angeles, Albuquerque, London, Madrid, Vancouver and Toronto. Compared to other distributors, Netflix pays more for TV shows up front, but keeps more 'upside' (i.e. future revenue opportunities from possible syndication, merchandising, etc.) on big hits.")

                
                if st.checkbox("Show Netflix Streak graph"):
                    image = Image.open('resources/imgs/7677.jpeg')
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
                            y='Revenue(in millions USD):Q',
                            x="Region:O"
                        )#.properties(height=700)
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        if st.checkbox('Show Netflix Revenue by Region in 2020 graph interperetation'):
                            st.markdown("**Looking at the above bar graph:**")
                            st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                            st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                            
                    else:
                        for i in region_list2:
                            region_emp_list.append(revenue_x_region_dict[i])
                        
                        st.subheader("Netflix Revenue by Region in 2020 graph")
                        
                        source = pd.DataFrame({
                        'Revenue(in millions USD)': region_emp_list, 'Region': region_list2})

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Revenue(in millions USD):Q',
                            x="Region:O"
                        )
                        st.altair_chart(bar_chart, use_container_width=True) 
                                        
                #Graph2
                region = ["US and Canada", "Europe, Middle East, and Africa", "Latin America", "Asia Pacific"]
                monthly_revenue = [13.51, 11.05, 9.32, 7.12]
                monthly_revenue_x_region_dict = {}
                region_emp_list2 = []
                for i in range(len(region)):
                    monthly_revenue_x_region_dict[region[i]] = monthly_revenue[i]
                if st.checkbox("Show Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020 graph"):
                    region_list2 = st.multiselect("Which regions would you like to see?", region)
                    
                    #st.write(monthly_revenue_x_region_dict["Asia Pacific"])
                    if len(region_list2) == 0:
                        st.subheader("Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020")
                        st.subheader("in USD")
                        source = pd.DataFrame({
                            'Monthly Revenue': monthly_revenue,
                            'Region': region
                            })
                        

                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Monthly Revenue:Q',
                            x="Region:O"
                        )#.properties(height=700)
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        if st.checkbox('Show Netflix Revenues by Region in 2020 graph interperetation'):
                            st.markdown("**Looking at the above bar graph:**")
                            st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                            st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                            
                    else:
                        for i in region_list2:
                            region_emp_list2.append(monthly_revenue_x_region_dict[i])
                        
                        st.subheader("Netflix Average Monthly Revenue per Paying Subscriber, Q40 2020")
                        st.subheader("in USD")
                        source = pd.DataFrame({
                        'Monthly Revenue': region_emp_list2, 'Region': region_list2 })

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
                    share_x_region_dict[region[i]] = share[i]
                if st.checkbox("Show Share of Netflix Subscribers Worldwide by Region as of Q4 2020 graph"):
                    region_list3 = st.multiselect("Whichs region would you like to see?", region)
                    region_emp_list3 = []
                    if len(region_list3) == 0:
                            
                        st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                        st.subheader("in millions")
                        source = pd.DataFrame({
                            'Share': share,'Region': region
                            })
                    
                        bar_chart = alt.Chart(source).mark_bar().encode(
                            y='Share:Q',
                            x="Region:O"
                        )#.properties(height=700)
                        st.altair_chart(bar_chart, use_container_width=True)
                        
                        if st.checkbox('Show Netflixs Revenue by Region in 2020 graph interperetation'):
                            st.markdown("**Looking at the above bar graph:**")
                            st.markdown("* We see that region 'US and Canada' is bringing in the most revenue")
                            st.markdown("* Followed by regions 'Europe, Middle East, and Africa', neutral 14.87% and lastly 8.19%")

                            
                    else:
                        for i in region_list3:
                            region_emp_list3.append(share_x_region_dict[i])
                        
                        st.subheader("Share of Netflix Subscribers Worldwide by Region as of Q4 2020")
                        st.subheader("in millions")
                        source = pd.DataFrame({
                        'Share': region_emp_list3, 'Region': region_list3 })

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
                               
        
        if selected == "Models":
            st.title("Models")
            
            image = Image.open('resources/imgs/96-967460_ai-machine-learning-icon-300x212.png')
            st.image(image, caption=None,width=500)
            
            
            
            if st.checkbox('Brief description of recommenders & models'):
                st.subheader("Content Based Recommender System")
                image = Image.open('resources/imgs/my4.png')
                st.image(image, caption='Recommender System',width=500)
                st.markdown("A Content-Based Recommender works by the data that we take from the user, either explicitly (rating) or implicitly (clicking on a link). By the data we create a user profile, which is then used to suggest to the user, as the user provides more input or take more actions on the recommendation, the engine becomes more accurate.")

                st.subheader("Collaborative Based Recommender System")
                image = Image.open('resources/imgs/Screenshot-2893.png')
                st.image(image, caption='Recommender System',width=500)
                st.markdown("In Collaborative Filtering, we tend to find similar users and recommend what similar users like. In this type of recommendation system, we don’t use the features of the item to recommend it, rather we classify the users into the clusters of similar types, and recommend each user according to the preference of its cluster.")
                
                st.subheader("SVD")
                image = Image.open('resources/imgs/1_dnvjYsiEhj-NzFf6ZeCETg.jpeg')
                st.image(image, caption='Recommender System',width=500)
                st.markdown("The Singular Value Decomposition (SVD) of a matrix is a factorization of that matrix into three matrices. It has some interesting algebraic properties and conveys important geometrical and theoretical insights about linear transformations. It also has some important applications in data science. In this article, I will try to explain the mathematical intuition behind SVD and its geometrical meaning. ")
            
            if st.checkbox("Show  Comparison of Algorithm on RMSEE graph"):
                st.subheader("Comparison of Algorithm on RMSE") 
                                     
                image = Image.open('resources/imgs/a3bc481d-9fcd-4161-a0ec-5618f65224db.jpg')
                st.image(image, caption=None,width=500)

                if st.checkbox('Show Comparison of Algorithm on RMSE graph'):
                    st.markdown("**Looking at the above graph:**")
                    st.markdown("* SVDpp is the best performing model in terms of RMSE, in second place is SVD model.")
                    st.markdown("* The remaining models perform worse in terms of RMSE, worst of which is SlopeOne.") 

            if st.checkbox("Show  Comparison of Algorithm on MAE graph"):
                image = Image.open('resources/imgs/8f91a0cb-f6e4-4f8a-bbc6-e3294b00c5e8.jpg')
                st.image(image, caption=None,width=500)
                

                if st.checkbox('Show Comparison of Algorithm on MAE graph'):
                    st.markdown("**Looking at the above graph:**")
                    st.markdown("* SVDpp is the best performing model in terms of MAE, in second place is SVD model.")
                    st.markdown("* The remaining models perform worse in terms of MAE, worst of which is SlopeOne.")             
if __name__ == '__main__':
    main()
