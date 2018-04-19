## Notes on Statistical learning

#### Overview of Statistical Learning :

#### Linear Regression :

#### Classification : 

* Logistic regression
* Multiple Logistic Regression
* Naive Bayes
* Linear Discriminant Analysis
* Quadratic Discriminant Analysis
* K-Nearest Neighbors

#### Resampling methods :
 
 * Validation set approach
 * Leave-one-out cross validation
 * k-Fold Cross Validation
 * Train,test,validation
 * Bootstrap

#### Tree based models :

* Decision Trees
* Bagged Trees
* Random Forests
* Gradient Boosted Models(Boosting)

#### Unsupervised Learning :

* Principal Component Analysis
* K-means clustering
* Hierarchical clustering

#### Evaluation metrics :

* RMSE
* log-loss/cross-entropy
* AUC
 
#### Regularization :
* Ridge regression 
* LASSO regression
#### General Models Additive :
#### Support Vector Machines :
---
### Overview of Statistical Learning:

#### Types of machine learning algortihms:

* Supervised Learning - Linear regression, Logistic regression etc.
* Unsupervised Learning - k-means clustering, Hierarchical clustering etc. 
* Reinforcement Learning - Q-learning

Generally, suppose that we observe a quantitative response **Y** and **p** different predictors, **X1, X2,...Xp**. We assume that there is some relationship between **Y** and **Xi**, where i =1-p, which can be written in the very general form

**Y=f(X)+e**

**f** is some fixed but unknown function and **e** is a random error term, which independent of **X** and has mean zero.

Say we have a function **f_pred** which is our estimate for **f**.

**Y_pred=f_pred(X)**

In general **f_pred** is not a perfect estimate for **f**, and this inaccuracy will introduce some error. Some part of this error is *reducible* by using appropriate statistical learning techniques. However, even if we come up with a perfect estimate for **f**, our prediction would still have some *irreducible* error in it. This is because **Y** is also a function **e**.

#### Prediction Accuracy-Model Interpretability trade-off

If a model is less flexible or inflexible, such as the linear regression, it is more interpretable. Non-linear methods such as bagging, bossting and SVMs are highly flexible but hard to interpret.

#### Bias-variance tradeoff

The expected test MSE can be broken down for any data point **x** can be broken down into three fundamental quantities: the *variance* of **f_pred(x)**, the square of *bias* of **f_pred(x)** and the *variance* of the error terms **e** (the *irreducible* error) . Our estimate function **f_pred** should be in such a way that it minimizes both the parts of the *reducible* error i.e., the *variance* and sqaure of *bias*.

*Variance* refers to the amount by which **f_pred** would change if we estimated it using a different training data set. If a method has high variance then small changes in the training data can result in large changes in **f_pred** . In general, more flexible statistical methods have higher variance. Eg: Descion Trees

On the other hand, *bias* refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. For example, linear regression assumes that there is a linear relationship between **Y** and **X1 , X2 , . . . , Xp**. It is unlikely that any real-life problem truly has such a simple linear relationship, and so performing linear regression will undoubtedly result in some bias in the estimate of **f**.

As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease.

### Decision Trees :

* Minimation metric: gini index or informaton gain for classification problems, Residual sum of squares (RSS) for regression
* Hyperparameters for tuning:
	* **minsplit**- minimum number of noes needed to attempt a split
	* **cp**- complexity parameter
	* **maxdepth**- depth of a decision tree
* Pruning

### Bagged Trees :

* **Bootstrap aggregation**(Bagging) is a general purpose procedure for reducing the variance of a statistical learning method.
* **Bootstrapping** is nothing sampling with replacement.
* Decision Trees have a high variance and low bias
* In bagged trees, we generate N different bootstrapped datasets and build N different decision trees and take the average of all the N trees
* Averaging the N trees reduces the variance
* Bagged trees are basically **ensemble** of decision trees built upon bootstrapped datasets 
* Interpretability is lost because of this though. However, the variable importance measures can be obtained by recording how much the RSS(for regression trees) or the gini index(for classification) has decreased on average across all N trees due to splits over the variable.

### Random Forests :

* Random forests provide an improvement over bagged trees by way of a small tweak that **decorrelates** the trees.
* In bagging, we build a number of decision trees on bootstrapped training samples. 
* In random forests, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors.
* The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose m ≈ √p


### Ridge & Lasso:
* Same as linear regression, but the loss function we try to minimize will have RSS+ lambda times the sum of squares of regression coefficients, where lamda isthe tuning parameter
* LASSO is similar of Ridge, only difference is instead of square of the coefficients, the absolute value multiplied by lambda is added to the loss function
* Regularization, significantly reduces the variance of the model, without substantial increase in its bias

### Questions testing understanding :
* Does feature standardization improve model performance? Why do we need it?
* What is the problem with having multicollinearity/ correlated features?
* Does coarse classification of continuous variables improve model performance?
* How can gbm be parallelized(xgboost) if the second tree is dependent on the first tree?
* How do class weights work? How is it added in the error function for regession and classification problems?
* What is the difference between an ordinary least squares linear model and generalized linear model(GLM)?
* Does keeping insignficant features impact model performance?
* What is curse of dimensionality?
* Does doing PCA improve model performance?
* What are the important hyperparameters for decision trees, random forests, gbm, xgboost?
* Are xgboost and gbm the same?
* What does bag.fraction and train.fraction do in gbm function?
* Why is the default distribution "bernoulli" in gbm function
* What will happen if the response actually doesn't follow a guassian or binomial distribution? What does the distributon parameter in the gbm function do? 
* How does regularization work in tree based models?
* How does regularization work in classification problems?
* How do you define the number of clusters in a clustering problem?
* Explain test-set and validation set approach
* What is decomposition of variance?