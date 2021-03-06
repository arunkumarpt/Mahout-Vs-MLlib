# Mahout-Vs-MLlib
## Topics

    What is Machine Learning?
    Supervised Learning
    Unsupervised Learning
    Linear Regression with One Variable
    What is Apache Mahout?
    How Mahout Works - Local Demo
    Machine Learning with Spark
    Mahout Vs Spark Mllib
    Demo : Mahout Movie Recommender System
    Demo : Mllib Movie Recommender System
   

## What is Machine Learning?


Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.


Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: Assume that we are creating a cricket game for 1 over to bat with a target of 10 runs.

    - E = the experience of playing many games of cricket
    - T = the task of playing cricket.
    - P = the probability that you will win the next game.


##Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

    - Regression
    - Classification

##Unsupervised Learning
Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

    - Clustering

##Linear Regression with One Variable

    - Model Representation
    - The Hypothesis Function
    - Cost Function
    - Correlation

##Commonflow 

```
Dataset -> split - Traing - Train Model - Test Model
                 - Test
```

#####Reference

https://share.coursera.org/wiki/index.php/ML:Introduction


#Machine Learning with Mahout 

##What is Apache Mahout?

    - Mahout is a Java library – Implementing Machine Learning techniques
    - A Mahout is an elephant trainer/driver/keeper, hence…
    - Hadoop brings
    - Library of machine learning algorithms
    
    
## What Mahout Does

Mahout supports four main data science use cases:

Collaborative filtering – mines user behavior and makes product recommendations (e.g. Amazon recommendations)

Clustering – takes items in a particular class (such as web pages or newspaper articles) and organizes them into naturally occurring groups, such that items belonging to the same group are similar to each other

Classification – learns from existing categorizations and then assigns unclassified items to the best category

Frequent itemset mining – analyzes items in a group (e.g. items in a shopping cart or terms in a query session) and then identifies which items typically appear together



##How Mahout Works

Mahout provides an implementation of various machine learning algorithms, some in local mode and some in distributed mode (for use with Hadoop). Each algorithm in the Mahout library can be invoked using the Mahout command line.
    
#####Reference

http://hortonworks.com/hadoop/mahout/

## Mahout Local Mode example
### Data
```
1,10,1.0
1,11,2.0
1,12,5.0
1,13,5.0
1,14,5.0
1,15,4.0
1,16,5.0
1,17,1.0
1,18,5.0
2,10,1.0
2,11,2.0
2,15,5.0
2,16,4.5
2,17,1.0
2,18,5.0
3,11,2.5
3,12,4.5
3,13,4.0
3,14,3.0
3,15,3.5
3,16,4.5
3,17,4.0
3,18,5.0
4,10,5.0
4,11,5.0
4,12,5.0
4,13,0.0
4,14,2.0
4,15,3.0
4,16,1.0
4,17,4.0
4,18,1.0
```
###Recommender
```
package mahout.ml;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class recommender {

	public static void main(String[] args) throws IOException, TasteException {
		DataModel model = new FileDataModel(new File("dataset.csv"));
		UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
		UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
		
		List<RecommendedItem> recommendations = recommender.recommend(2, 3);
		for (RecommendedItem recommendation : recommendations) {
		  System.out.println(recommendation);
		}
	
	}

}

```
###Output
```
RecommendedItem[item:12, value:4.8328104]
RecommendedItem[item:13, value:4.6656213]
RecommendedItem[item:14, value:4.331242]

```
### Evaluate model

```
package mahout.ml;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


class MyRecommenderBuilder implements RecommenderBuilder{

	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
		return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
	}
	
}


public class EvaluateRecommender {

	
	public static void main(String[] args) throws IOException, TasteException {
		// TODO Auto-generated method stub

		DataModel model = new FileDataModel(new File("dataset.csv"));
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new MyRecommenderBuilder();
		double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
		System.out.println(result);
		 	
	}

}
```
###Output
```
1.2027597427368164

```

#Machine Learning with Spark

##MLlib is Apache Spark's scalable machine learning library.

###Ease of Use
Usable in Java, Scala and Python.

MLlib fits into Spark's APIs. You can use any Hadoop data source (e.g. HDFS, HBase, or local files), making it easy to plug into Hadoop workflows.

###Performance
High-quality algorithms, 100x faster than MapReduce.

Spark excels at iterative computation, enabling MLlib to run fast. At the same time, we care about algorithmic performance: MLlib contains high-quality algorithms that leverage iteration, and can yield better results than the one-pass approximations sometimes used on MapReduce.


###Algorithms
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


#####References

https://mahout.apache.org/users/clustering/visualizing-sample-clusters.html
https://samarthbhargav.wordpress.com/2014/04/22/logistic-regression-in-apache-spark/
http://www.cise.ufl.edu/class/cis6930fa11lad/cis6930fa11_Spark.pdf
https://mahout.apache.org/users/basics/quickstart.html
https://mahout.apache.org/users/recommender/userbased-5-minutes.html
http://faustineinsun.blogspot.com/2014/01/to-do-run-mahout-build-in-examples-on.html


#Mahout Vs Spark

###The main difference will come from underlying frameworks. 
In case of Mahout it is Hadoop MapReduce and in case of MLib it is Spark. To be more specific - from the difference in per job overhead. 

If Your ML algorithm mapped to the single MR job - main difference will be only startup overhead, which is dozens of seconds for Hadoop MR, and let say 1 second for Spark. So in case of model training it is not that important.
Things will be different if Your algorithm is mapped to many jobs. In this case we will have the same difference on overhead per iteration and it can be game changer. 
Lets assume that we need 100 iterations, each needed 5 seconds of cluster CPU.

On Spark: it will take ```100*5 + 100*1``` seconds = 600 seconds.
On Hadoop: MR (Mahout) it will take ```100*5+100*30``` = 3500 seconds.

In the same time Hadoop MR is much more mature framework then Spark and if you have a lot of data, and stability is paramount 
### Comparison

http://www.techwars.io/fight/mahout/mllib/

#####References

http://stackoverflow.com/questions/23511459/what-is-the-difference-between-apache-mahout-and-apache-sparks-mllib

#Demo
##Mahout Movie Recommendation 

##Datasets Samples
###movies.dat 
```
3936::Phantom of the Opera, The (1943)::Drama|Thriller
3937::Runaway (1984)::Sci-Fi|Thriller
3938::Slumber Party Massacre, The (1982)::Horror
3939::Slumber Party Massacre II, The (1987)::Horror
3940::Slumber Party Massacre III, The (1990)::Horror
3941::Sorority House Massacre (1986)::Horror

```
###Ratings
```
5558::1005::1::959391725
5558::1009::2::959391435
5558::1010::1::959391467
5558::1012::1::959391368
5558::1013::2::959391552
5558::1014::1::959391435
5558::1015::1::959391552
5558::1018::2::959391769
```
###Users
```
6014::M::45::1::80634
6015::F::25::9::80013
6016::M::45::1::37209
6017::F::35::7::21117
6018::M::35::1::48906
6019::M::25::0::10024
6020::M::50::16::10023
```
###user0 list
```
Toy Story (1995): 1
Independence Day (a.k.a. ID4) (1996): 5
Dances with Wolves (1990): 1
Star Wars: Episode VI - Return of the Jedi (1983): 5
Mission: Impossible (1996): 5
Ace Ventura: Pet Detective (1994): 5
Die Hard: With a Vengeance (1995): 5
Batman Forever (1995): 5
Pretty Woman (1990): 1
Men in Black (1997): 5
Dumb & Dumber (1994): 5
```
### Commands
```
hadoop fs -rmr temp/
bin/mahout recommenditembased --input input/ratings1 --usersFile input/user0 --numRecommendations 20 --output output7/ --similarityClassname SIMILARITY_PEARSON_CORRELATION
```

###Output
```
1405:5.0  Beavis and Butt-head Do America (1996)
3082:5.0  World Is Not Enough, The (1999)
1779:5.0  Sphere (1998)
2402:5.0  Rambo: First Blood Part II (1985)
466:5.0   Hot Shots! Part Deux (1993)
1597:5.0  Conspiracy Theory (1997)
2195:5.0  Dirty Work (1998)
1377:5.0  Batman Returns (1992)
379:5.0   Timecop (1994)
849:5.0   Escape from L.A.
```

#####Referene

http://girlincomputerscience.blogspot.com/2010/11/apache-mahout.html

##Spark Movie Recommendation 
```
bin/spark-submit --driver-memory 2g --class MovieLensALS1  MovieRecommendation.jar movielens/ myratings.txt
```

###Output
```
1: Anatomy (Anatomie) (2000)
 2: Hunted, The (1995)
 3: Coldblooded (1995)
 4: Police Story 4: Project S (Chao ji ji hua) (1993)
 5: Very Thought of You, The (1998)
 6: Diamonds (1999)
 7: Zachariah (1971)
 8: Spy Hard (1996)
 9: Happy Gilmore (1996)
10: Six-String Samurai (1998)
```
#####Referene

https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.html


#Mahout on Spark
Mahout News : https://mahout.apache.org/

```
export MAHOUT_HOME=/Users/arun/mahout
export SPARK_HOME=/Users/arun/Downloads/spark
export MASTER=spark://arun:7077

bin/mahout spark-shell

http://localhost:8080/ 
```

#####Reference
https://mahout.apache.org/users/sparkbindings/play-with-shell.html



