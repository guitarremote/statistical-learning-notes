
## Questions on statistics and machine learning

* What is the difference between **standard deviation** and **standard error** ?
	* The **Standard Error** of a **statistic** is the standard deviation of its **sampling distribution**. Going by that analogy, the **standard error of the mean**(SEM) is an estimate of how far the **sample mean** is likely to be from the **population mean**. The **standard deviation** of the sample on the other hand is the degree to which individuals within the sample differ from the sample mean.
	* Since population standard deviation is seldom known, the standard error is usually estimated as the sample standard deviation divided by the square root of sample size(N)

* What is the law of large numbers ?
	* Law of large numbers states that as a sample size grows, its mean gets closer to the average of the whole population. There are two laws: weak and strong. The weak law of large numbers refers to convergence in probability, whereas the strong law of large numbers refers to almost sure convergence. Basically there are two different mathematical equations I am not writing here. The main difference is that Almost sure convergence implies convergence in probability, but the converse is not true, hence the names strong and weak.

* What is **68-95-99 rule** ?
	* For normally distributed data the standard deviation has some extra information, namely the **68-95-99.7** rule which tells us the percentage of data lying within 1, 2 or 3 standard deviation from the mean. Note that this rule does not hold good for skewed(left or right) distributions.

* Briefly describe A/B testing ?
	* An A/B  test is an experiment with two groups to establish which one of the two treatments is superior. Often one of the two treatments is the standard existing treatment, which in some cases may be no treatment.
	* The subjects exposed to standard treatment is called the *control group* and the other group is called *treatment group*. For example, testing two prices to determine which yields more net profit.

* What is the difference between a t-test and z-test ?
	* In probability and statistics, the t-distribution is any member of a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown.
	* According to the **central limit theorem**, the sampling distribution of a statistic will follow a normal distribution, as long as the sample size is sufficiently large. Therefore, when we know the standard deviation of the population, we can compute a z-statistic, and use the normal distribution to evaluate probabilities with the sample mean.

	* The t distribution is used when you have small samples. The larger the sample size, the more the t distribution looks like the normal distribution. In fact, for sample sizes larger than 20 (e.g. more degrees of freedom), the distribution is almost exactly like the normal distribution. The probability value corresponding to a t-statistic is looked up in the t-table corresponding to N-1 degrees of freedom and if it is less than the  **significance level**(alpha-level usually 5%), then we can reject the Null Hypothesis.
	* In a z-test, we know the population standard deviation. Instead of t-statistic, we calculate the z-statistic and its corresponding p-value in Normal distribution.

* What are the different types of ML algorithms ?

	* Supervised Learning - Linear regression, Logistic regression etc.
	* Unsupervised Learning - k-means clustering, Hierarchical clustering etc.
	* Reinforcement Learning - Q-learning
* Explain bias-variance tradeoff.
	* The expected test MSE can be broken down for any data point **x** can be broken down into three fundamental quantities: the *variance* of `f_pred(x)`, the square of *bias* of `f_pred(x)` and the *variance* of the error terms **e** (the *irreducible* error) . Our estimate function **f_pred** should be in such a way that it minimizes both the parts of the *reducible* error i.e., the *variance* and square of *bias*.

	* *Variance* refers to the amount by which **f_pred** would change if we estimated it using a different training data set. If a method has high variance then small changes in the training data can result in large changes in `f_pred` . In general, more flexible statistical methods have higher variance. Eg: Decision Trees

	* On the other hand, *bias* refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. For example, linear regression assumes that there is a linear relationship between **Y** and **X1 , X2 , . . . , Xp**. It is unlikely that any real-life problem truly has such a simple linear relationship, and so performing linear regression will undoubtedly result in some bias in the estimate of **f**.

	* As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease.

* How to know if your model has high bias or high variance?
	* We can know this by plotting learning curves. We plot the error on the training set and on the cross-validation set as functions of the number of training examples(randomly selected sets from training) for some set of training set sizes.
	* In the typical *high bias* case, the cross-validation error will initially go down and then plateau as the number of training examples grow. (With high bias, more data doesn’t help beyond a certain point.) The training error will initially go up and then plateau at approximately the level of the cross-validation error (usually a fairly high level of error). So if you have similar cross-validation and training errors for a range of training set sizes, you may have a high-bias model and should look into generating new features or changing the model structure in some other way.
	* In the typical *high variance* case, the training error will increase somewhat with the number of training examples, but usually to a lower level than in the high-bias case. (The classifier is now more flexible and can fit the training data more easily, but will still suffer somewhat from having to adapt to many data points.) The cross-validation error will again start high and decrease with the number of training examples to a lower but still fairly high level. So the crucial diagnostic is that the difference between the cross-validation error and the training set error is high. In this case, you may want to try to obtain more data, or if that isn’t possible, decrease the number of features.

* How to reduce bias or reduce variance in your model?
  * **Get more training examples** :
More training examples will work when you have high variance. More training examples will not fix a high bias, because your underlying model will still not be able to approximate the correct function.
  * **Smaller sets of features**:
Smaller sets of features will work when you have higher variance. Ng says, if you think you have high bias, “for goodness’ sake don’t waste your time by trying to carefully select the best features”.
  * **Try to obtain new features**:
This will work when the model is suffering from high bias.

* Explain Ridge & Lasso.
	* Same as linear regression, but the loss function we try to minimize will have RSS+ lambda times the sum of squares of regression coefficients, where lambda is the tuning parameter
	* LASSO is similar of Ridge, only difference is instead of square of the coefficients, the absolute value multiplied by lambda is added to the loss function
	* Regularization, significantly reduces the variance of the model, without substantial increase in its bias

* What are some important hyperparameters of decision trees :
	* **minsplit** - minimum number of noes needed to attempt a split
	* **cp** - complexity parameter
	* **maxdepth** - depth of a decision tree

* What are the loss functions used for building decision trees ?
	* Information gain on gini index / entropy for classification problems, Residual sum of squares (RSS) for regression

* What are bagged trees ?

	* **Bootstrap aggregation**(Bagging) is a general purpose procedure for reducing the variance of a statistical learning method.
	* **Bootstrapping** is nothing sampling with replacement.
	* Decision Trees have a high variance and low bias
	* In bagged trees, we generate N different bootstrapped datasets and build N different decision trees and take the average of all the N trees
	* Averaging the N trees reduces the variance
	* Bagged trees are basically **ensemble** of decision trees built upon bootstrapped datasets
	* Interpretability is lost because of this though. However, the variable importance measures can be obtained by recording how much the RSS(for regression trees) or the gini index(for classification) has decreased on average across all N trees due to splits over the variable.

* Explain how random forest algorithm works.

	* Random forests provide an improvement over bagged trees by way of a small tweak that **decorrelates** the trees.
	* In bagging, we build a number of decision trees on bootstrapped training samples.
	* In random forests, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors.
	* The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose m ≈ √p


* Does feature standardization improve model performance? Why do we need it?

* What is the problem with having multicollinearity/ correlated features?

* Does coarse classification of continuous variables improve model performance?

* How bagging work for classification problems i.e., when the response variable is *qualitative*?
    * There are quite a few approaches, but the simplest approach is taking the most commonly occurring class(majority vote) in all the bagged trees

* How is OOB error calculated for ensemble models (say trees)?
	* Mathematically we can prove that, on average, each bagged tree makes use of around two-thirds of the observations. The remaining one-third of the observations not used to fit a given bagged tree are referred to as the out-of-bag (OOB) observations. We can predict the response for an observation using each of the trees in which that observation was OOB and take their average (or majority vote). The OOB prediction is used to calculate OOB error.

* How can gbm be parallelized(xgboost) if the second tree is dependent on the first tree?

* How is variable importance calculated in ensemble tree based models?
	* The variable importance measures can be obtained by recording how much the RSS(for regression trees) or the gini index(for classification) has decreased on average across all N trees due to splits over the variable.

* Say I am trying to predict the house prices using gbm model. Will I get correct results if I build the model without log-transforming the price?

* How do class weights work? How is it added in the error function for regression and classification problems?

* What is the difference between an ordinary least squares linear model and generalized linear model(GLM)?

* Does keeping insignificant features impact model performance?

* What is curse of dimensionality?

* Does doing PCA improve model performance?

* What are the important hyperparameters for decision trees, random forests, gbm, xgboost?

* What is the difference between xgboost and gbm?

* What will happen if the response actually doesn't follow a gaussian or binomial distribution? What does the distribution parameter in the gbm function do?

* How does regularization work in tree based models?
    * Currently, there are different tree based models available for supervised learning problems. We have simple Decision Trees, Random Forests and XGBoost. The way regularization is performed is different in all of these.
    * Looking back at linear regression, regularization turns out to be a natural outcome of imposing a prior distribution on the regression parameters. Regularization leads to a modification in the cost function in the following form : Cost = MSE + lambda * f(weights). A typical choice for f will be sum of squares, or some of absolute values in some cases.
    * For a simple decision tree, such a notion of parameters does not exist, simply because the tree is a set of if else conditions and we have a predetermined output in each of the leaves. But, regularization can be implemented to control what the tree structure looks like. For instance, We could simply add a term to the cost function that grows high as the no of leaves grows, or as the depth of the tree increases, thereby restricting the structure, and controlling overfitting to some extent.
    * Similar technique works on Random Forests as well, but since bagging itself is a way to prevent overfitting, added regularization may only lead to underfitting, and can be something that needs to be tested out particular to the problem at hand. An interesting thing to note here is that python's sklearn do not have such implementations, but rather provide individual control over the different levers of the tree structure.
    * XGBoost extensively uses regularization. This is one of those key differences between GBMs and XGBoost. The regularization is implemented in two different forms. First is similar to control the tree structure as described above. Second is in relation to the parameters. XGBoost explicitly determines the best set of weights for a given leaf by minimizing the cost function. This opens up the possibility to add another regularization term as a function of these weights. The total cost function would now be MSE + gamma * Leaf Count + Lambda * f(all weights)

* How does regularization work in classification problems?

* How do you define the number of clusters in a clustering problem?
    * Clustering is an unsupervised learning problem where the aim is to segment the data into several clusters, so that points with similar nature (column values) fall in the same cluster. The objective function here will take some form of distance metric between the different points. The lower the distance, closer the points are and more probable to be in the same cluster. Typically, we would define cluster centers and measure the distance of each point from this cluster center. Sum all of these up to get the total "cost" for the current cluster count. Now, we can repeat the same exercise with different cluster counts and do a plot of cost vs cluster count, which typically would take the shape of an [elbow](https://github.com/DragonWarrior15/statistical-learning-notes/blob/master/images/elbow_plot.png).
    * A better metric to judge the no of clusters can be [Silhouette distance](https://en.wikipedia.org/wiki/Silhouette_(clustering)).

* Explain test-set and validation set approach
    * Test and Validation approach is fundamental to solving any Machine Learning problem. We are given input data X, and some target data y. The aim is to learn a model that can accurately predict the value of y given some X (the approach is true for both regression and classification problems). We can measure accuracy through multiple metrics, but simply determining this on the training data is not the best idea as the model has already seen this data, and is to some extent aware of the actual labels for this data. If such a model is tested on some unseen data set, we may get unexpected results.
    Hence, we incorporate this notion of unseen data into the problem itself, and try to tackle two problems simultaneously. First, to determine what is the best version of the model, and second, a common dataset to compare the results of multiple models. Validation set is a part of the input data that is not used for training and serves the first purpose. Similarly, test data is another part of the input data (not overlapping with the validation data), which is not used for training, and helps compare multiple models with one another.

    **TL;DR** : Split your input data into three parts : Training, Validation and Test. Use training data to train a model, validation data to tune the model hyperparameters, and once you have final versions of different types of models, use the test set to compare them. Usually, the split would be done randomly (say 60:20:20), but in cases like time dependent data, we need to respect the flow of time as well, and in that case, the time order will be test > validation > training. The way this is split is done depends on the problem at hand.

* What is decomposition of variance?

* What are interactions? How do you check for interactions?

* What is Bayes theorem? Difference between prior and posterior probability?
