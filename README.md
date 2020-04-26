# Reddit-Flare-Prediction

This repository contains 3 Jupyter notebooks and 1 python web application deployed using Flask.
Jupyter notebook contains code for following operations:
1. Data Collection - 
This notebook contains code for web scraping data from "reddit.com/r/India" . The web scraping library used is BeautifulSoup and data is stored in .csv format. The data which is scraped from each post includes Title of the post, Author of the post, Web-domain of the post and Flair(Topic).

2. Exploratory Data Analysis -
This notebook contains code for data analysis of the scraped data using various kinds of plots such as barplot, histogram, violinplot, boxplot, etc. The plots are made by using pandas, numpy, matplotlib and seaborn library.

3. Flare Prediction - 
This notebook contains code for some pre-processing and trying various models on the given data such as Naive-Bayes, Random-Forest, Support Vector Machines and XgBoost. It also contains a neural network implemented by keras using tensorflow backend.The prediction accuracy is coming out to be around 83%. Libraries used are numpy, pandas, sklearn and nltk

Web app developed using Flask
The web app is developed in Flask framework.The serialized Model and Vectorizer are loaded to predict from the Link provided to us.
There are two choices in the web app to predict the topic of the article provided by the link:
1. Directly write the link
2. Upload a file containing links



