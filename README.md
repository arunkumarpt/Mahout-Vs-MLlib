# Mahout-Vs-MLlib

# What is Machine Learning?


Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.


Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: Assume that we are creating a cricket game for 1 over to bat with a target of 10 runs.
E = the experience of playing many games of cricket
T = the task of playing cricket.
P = the probability that you will win the next game.


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

#Commonflow 


#Reference

https://share.coursera.org/wiki/index.php/ML:Introduction


#Machine Learning with Mahout 



#What is Apache Mahout?
    - A Mahout is an elephant trainer/driver/keeper, hence…
    - Hadoop brings ---
    - Library of machine learning algorithms
    -Mahout is a Java library – Implementing Machine Learning techniques
    
What Mahout Does
Mahout supports four main data science use cases:

Collaborative filtering – mines user behavior and makes product recommendations (e.g. Amazon recommendations)
Clustering – takes items in a particular class (such as web pages or newspaper articles) and organizes them into naturally occurring groups, such that items belonging to the same group are similar to each other
Classification – learns from existing categorizations and then assigns unclassified items to the best category
Frequent itemset mining – analyzes items in a group (e.g. items in a shopping cart or terms in a query session) and then identifies which items typically appear together


http://hortonworks.com/hadoop/mahout/

How Mahout Works
Mahout provides an implementation of various machine learning algorithms, some in local mode and some in distributed mode (for use with Hadoop). Each algorithm in the Mahout library can be invoked using the Mahout command line.
    









#Machine Learning with Spark
MLlib is Apache Spark's scalable machine learning library.

Ease of Use
Usable in Java, Scala and Python.

MLlib fits into Spark's APIs and interoperates with NumPy in Python (starting in Spark 0.9). You can use any Hadoop data source (e.g. HDFS, HBase, or local files), making it easy to plug into Hadoop workflows.

Performance
High-quality algorithms, 100x faster than MapReduce.

Spark excels at iterative computation, enabling MLlib to run fast. At the same time, we care about algorithmic performance: MLlib contains high-quality algorithms that leverage iteration, and can yield better results than the one-pass approximations sometimes used on MapReduce.


Algorithms
MLlib 1.1 contains the following algorithms:

linear SVM and logistic regression
classification and regression tree
k-means clustering
recommendation via alternating least squares
singular value decomposition
linear regression with L1- and L2-regularization
multinomial naive Bayes
basic statistics
feature transformations








# Valuble Links


https://mahout.apache.org/users/clustering/visualizing-sample-clusters.html

https://samarthbhargav.wordpress.com/2014/04/22/logistic-regression-in-apache-spark/

http://www.cise.ufl.edu/class/cis6930fa11lad/cis6930fa11_Spark.pdf

#Quick Start
https://mahout.apache.org/users/basics/quickstart.html
https://mahout.apache.org/users/recommender/userbased-5-minutes.html




http://faustineinsun.blogspot.com/2014/01/to-do-run-mahout-build-in-examples-on.html





http://stackoverflow.com/questions/23511459/what-is-the-difference-between-apache-mahout-and-apache-sparks-mllib

The main difference will came from underlying frameworks. In case of Mahout it is Hadoop MapReduce and in case of MLib it is Spark. To be more specific - from the difference in per job overhead 
If Your ML algorithm mapped to the single MR job - main difference will be only startup overhead, which is dozens of seconds for Hadoop MR, and let say 1 second for Spark. So in case of model training it is not that important.
Things will be different if Your algorithm is mapped to many jobs. In this case we will have the same difference on overhead per iteration and it can be game changer. 
Lets assume that we need 100 iterations, each needed 5 seconds of cluster CPU.

On Spark: it will take 100*5 + 100*1 seconds = 600 seconds.
On Hadoop: MR (Mahout) it will take 100*5+100*30 = 3500 seconds.
In the same time Hadoop MR is much more mature framework then Spark and if you have a lot of data, and stability is paramount - I would consider Mahout as serious alternative.






##Demo
#Mahout Movie Recommendation 
```
hadoop fs -rmr temp/
bin/mahout recommenditembased --input input/ratings1 --usersFile input/user0 --numRecommendations 20 --output output7/ --similarityClassname SIMILARITY_PEARSON_CORRELATION
```
Referene
http://girlincomputerscience.blogspot.com/2010/11/apache-mahout.html

#Spark Movie Recommendation 
```
bin/spark-submit --driver-memory 2g --class MovieLensALS1  MovieRecommendation.jar movielens/ myratings.txt
```
Referene
https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.html

#Mahout on Spark

```
export MAHOUT_HOME=/Users/arun/mahout
export SPARK_HOME=/Users/arun/Downloads/spark
export MASTER=spark://arun:7077

sbin/start-all.sh

http://localhost:8080/ 
```

Reference
https://mahout.apache.org/users/sparkbindings/play-with-shell.html



