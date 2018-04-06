## Notes on Statistical learning

#### Types of machine learning algortihms

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

#### Linear Regression

#### Classification: 

* Logistic regression
* Multiple Logistic Regression
* Linear Discriminant Analysis
* Quadratic Discriminant Analysis
* K-Nearest Neighbors

#### Resampling methods:
 
 * Validation set approach
 * Leave-one-out cross validation
 * k-Fold Cross Validation
 * Train,test,validation
 * Bootstrap

#### Tree based models :

* Decision Trees
	* Splitting criterion: gini index/informaton gain
	* Hyperparameters for tuning:
		* **minsplit**- minimum number of noes needed to attempt a split
		* **cp**- complexity parameter
		* **maxdepth**- depth of a decision tree
* Bagging, Random Forests
* Gradient Boosted Models(Boosting)

#### Unsupervised Learning :

* Principal Component Analysis
* K-means clustering
* Hierarchical clustering

#### Evaluation metrics :

* RMSE
* log-loss/cross-entropy
* AUC
 

Totally not known things

Regularization, General Additive Models, Support Vector Machines



