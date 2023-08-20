# Introduction

Most if not all machine learning algorithms can be viewed as a form of empirical risk minimization where the object is to find the parameters $\theta$ that for some loss function $L$ minimize 

$\hat{\theta} = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L(f_{\theta}(x_i), y_i)$

where an underlying assumption is that minimizing this empirical risk corresponds to approximately minimizing the true risk which considers all examples in the populations which would imply that $f_\theta$ is approximately the true target function $f$.

In a multi-class setting, one can write

$\hat{\theta} = \arg\min_{\theta} \left( \frac{1}{N_1} \sum_{i \in \mathcal{C}_1} L(f_{\theta}(x_i), y_i) + \frac{1}{N_2} \sum_{i \in \mathcal{C}_2} L(f_{\theta}(x_i), y_i) + \ldots + \frac{1}{N_C} \sum_{i \in \mathcal{C}_C} L(f_{\theta}(x_i), y_i) \right)$

Class imbalance occurs when some classes have much fewer examples than other classes. In this case, the corresponding terms contribute minimally to the sum which makes it easier for any learning algorithm to find an approximate solution to the empirical risk that mostly only minimizes the over the significant sums. This yields a hypothesis $f_\theta$ that may be very different from the true target $f$ with respect to the minority classes which may be the most important for the application in question.

# Naive Random Oversampling

One obvious possible remedy is to weight the smaller sums so that a learning algorithm can avoid approximate solutions that exploit their insignificance. This can be shown to be equivalent (on average) to repeating examples by naive random oversampling of the observations in such classes which is offered by this package along with other more advanced oversampling methods.

# ROSE


# SMOTE


# SMOTE-N


# SMOTEN-C
