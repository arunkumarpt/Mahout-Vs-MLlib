# Mahout-Vs-MLlib

# What is Machine Learning?


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



