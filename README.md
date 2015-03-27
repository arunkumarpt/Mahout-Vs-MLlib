# Mahout-Vs-MLlib

# What is Machine Learning?

ref :- https://share.coursera.org/wiki/index.php/ML:Introduction

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.


Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.
E = the experience of playing many games of checkers
T = the task of playing checkers.
P = the probability that the program will win the next game.

#Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
    - Regression
    - Classification

#Unsupervised Learning
Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.
    - Clustering

#Linear Regression with One Variable
    - Model Representation
    - The Hypothesis Function
    - Cost Function
    - Gradient Descent
    - Gradient Descent for Linear Regression

#Machine Learning with Mahout

#What is Apache Mahout?
    - A Mahout is an elephant trainer/driver/keeper, hence…
    - Hadoop brings ---
    - Library of machine learning algorithms
#20 News Gruop Classification






#Machine Learning with Spark




# Valuble Links

https://mahout.apache.org/users/clustering/visualizing-sample-clusters.html

https://samarthbhargav.wordpress.com/2014/04/22/logistic-regression-in-apache-spark/

http://www.cise.ufl.edu/class/cis6930fa11lad/cis6930fa11_Spark.pdf

#Quick Start
https://mahout.apache.org/users/basics/quickstart.html
https://mahout.apache.org/users/recommender/userbased-5-minutes.html

<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-core</artifactId>
    <version>0.9</version>
</dependency>


SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".


http://faustineinsun.blogspot.com/2014/01/to-do-run-mahout-build-in-examples-on.html

http://faustineinsun.blogspot.com/2014/01/to-do-run-mahout-build-in-examples-on.html



http://stackoverflow.com/questions/23511459/what-is-the-difference-between-apache-mahout-and-apache-sparks-mllib

The main difference will came from underlying frameworks. In case of Mahout it is Hadoop MapReduce and in case of MLib it is Spark. To be more specific - from the difference in per job overhead 
If Your ML algorithm mapped to the single MR job - main difference will be only startup overhead, which is dozens of seconds for Hadoop MR, and let say 1 second for Spark. So in case of model training it is not that important.
Things will be different if Your algorithm is mapped to many jobs. In this case we will have the same difference on overhead per iteration and it can be game changer. 
Lets assume that we need 100 iterations, each needed 5 seconds of cluster CPU.

On Spark: it will take 100*5 + 100*1 seconds = 600 seconds.
On Hadoop: MR (Mahout) it will take 100*5+100*30 = 3500 seconds.
In the same time Hadoop MR is much more mature framework then Spark and if you have a lot of data, and stability is paramount - I would consider Mahout as serious alternative.


#Mahout on Spark

https://mahout.apache.org/users/sparkbindings/play-with-shell.html



