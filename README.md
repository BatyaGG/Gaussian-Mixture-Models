# Gaussian-Mixture-Models
Python implementation of Gaussian Mixture Models(GMM) and Gaussian Mixture Regression(GMR) algorithms with 
examples and data files. GMM is a soft clustering algorithm which considers data as finite gaussian distributions
with unknown parameters. Current approach uses Expectation-Maximization(EM) algorithm to find gaussian states parameters.
EM is an iterative algorithm which converges to true gaussian parameters and stopped by log-likelihood threshold or
iteration number limit. To initialize gaussian parameters k-means clustering algorithm is used. After GMM is fitted,
the model is used to fit GMR to retrieve output data by specified inputs.

All math and concepts are referred mainly from the book:

Calinon, S. (2009)
Robot Programming by Demonstration: A Probabilistic Approach
EPFL Press ISBN 978-2-940222-31-5, CRC Press ISBN 978-1-4398-0867-2.
http://calinon.ch/paper6001.htm

Additional sources:

Tsishchanka, K.
Elementary Statistics: Chapter 9
http://www.tkiryl.com/Elementary%20Statistics/Chapter_9.pdf

Mathworks documentation
Clustering Using Gaussian Mixture Models
http://www.mathworks.com/help/stats/clustering-using-gaussian-mixture-models.html

Princeton University Library
Data and Statistical Services: Introduction to Regression
http://dss.princeton.edu/online_help/analysis/regression_intro.htm

# Installation
Clone or download the project 

Install following packages:
numpy <1.11.3>,
matplotlib <1.5.3>,
scipy <0.16.1>

Other versions of the packages were not tested, but higher versions are welcome.
Report me to b.saduanov@gmail.com if you have any problems.

# Usage
To use GMM_GMR algorithms copy all python files to your folder. In your main script
file import GMM_GMR class by writing "from GMM_GMR import GMM_GMR". To create instance
of GMM_GMR just call its constructor with two parameters, for example "gmm = GMM_GMR(4, 100)". Firts parameter is a number
of clusters to be used and second is number of data points to be created in regression.
Algorithm does clustering and regression by one method "fit(numpy.array: data)". Data should be of numpy.array(type = float) type,
and has dimension of NxM, where N is a number of features and M is a number of recordings or data points. Data matrix
should have first row as time series variable. Regression considers first row as independent input variable and other rows as
dependent (on first row) output variables.
