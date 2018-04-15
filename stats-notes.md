### Inferential Statistics

* **Population mean**$$ \mu $$
* **Population Standard deviation** $$\sigma=\sqrt{(\sum_{i=1}^N (X_i-\mu)^2)/N}$$
* **Population Variance** $$\sigma^2$$

* **Sample Mean** $$\bar{x}=\sum_{i=1}^N X_i/N$$
*  **Sample Standard Deviation** $$s= \sqrt{(\sum_{i=1}^N (X_i-\bar{x})^2)/(N-1)}$$
* **Sample Variance** $$s^2$$

The **Standard Error** of a **statistic** is the standard deviation of its **samplng distribution**. Going by that analogy, the **standard error of the mean**(SEM) is an estimate of how far the **sample mean** is likely to be from the **population mean**. The **standard deviation** of the sample on the other hand is the degree to which individuals within the sample differ from the sample mean.

* **Standard Error**  $$\sigma/\sqrt{N}$$ 

Since population standard deviation is seldom known, the standard error is usually estimated as the sample standard deviation divided by the square root of sample size(N)

$$s/\sqrt{N}$$

**Law of large numbers**

Law of large numbers states that as a sample size grows, its mean gets closer to the average of the whole population. There are two laws: weak and strong. The weak law of large numbers refers to convergence in probability, whereas the strong law of large numbers refers to almost sure convergence. Basically there are two different mathematical equations I am not writing here. The main difference is that Almost sure convergence implies convergence in probability, but the converse is not true, hence the names strong and weak.

**68-95-99 rule**

For normally distributed data the standard deviation has some extra information, namely the **68-95-99.7** rule which tells us the percentage of data lying within 1, 2 or 3 standard deviation from the mean. Note that this rule does not hold good for skewed(left or right) distributions 

**Statistical Experiments and significance testing**

* **A/B testing** 
	* An A/B  test is an experiment with two groups to establish which one of the two treatments is superiror.
	*  Often one of the two treatments is the standard existing treatment, which in some cases may be no treatment.
	* The subjects exposed to standard treatment is called the *control group* and the other group is called *treatment group*
	* Example: Testing two prices to determine which yields more net profit
* **ANOVA**
* **Chi-squatre test**

**t-test and z-test**

In probability and statistics, the t-distribution is any member of a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown.

According to the **central limit theorem**, the sampling distribution of a statistic will follow a normal distribution, as long as the sample size is sufficiently large.

Therefore, when we know the standard deviation of the population, we can compute a z-statistic, and use the normal distribution to evaluate probabilities with the sample mean.

The t distribution is used when you have small samples. The larger the sample size, the more the t distribution looks like the normal distribution. In fact, for sample sizes larger than 20 (e.g. more degrees of freedom), the distribution is almost exactly like the normal distribution.

The t-distribution with degrees of freedom “N – 1” is given below.
$$ t^*=(\bar{x}-\mu_{0})/s/\sqrt{N}$$

where s is the sample standard deviation. Let's say below are our Null and Alternate hypotheses.

$$H_{0}: \mu=\mu_{0}$$$$H_{A}: \mu \neq\mu_{0}$$

The probability value corresponding to the t-statistic is looked up in the t-table corresponsing to N-1 degrees of freedom and if it is less than the  **significance level**(alpha-level usually 5%), then we can reject the Null Hypothesis. This probability (p-value) is basically the probability of observing a t-statistic which is greater than or equal to the absolute value of t-statistic we calculated in the formula above.

In a z-test, we know the population standard deviation. Instead of t-statistic, we calculate the z-statistic and its corresponding p-value in Normal distribution.

$$ z^*=(\bar{x}-\mu_{0})/\sigma/\sqrt{N}$$

As a thumb rule, for both t-test and z-test if a 95% confidence interval is being considered for hypothesis testing, any t/z statistic which is 2 SE's greater than or less than the Null Hypothesis value will have a probability less than 0.05. Meaning we can reject the Null Hypothesis.