
### Generalizing to Multiclass
Papers often propose the resampling algorithm for the case of binary classification only. In many cases, the algorithm only expects a set of points to resample and has nothing to do with the existence of a majority class (e.g., estimates the distribution of points then generates new samples from it) so it can be generalized by simply applying the algorithm on each class. In other cases, there is an interaction with the majority class (e.g., a point is borderline in `BorderlineSMOTE1` if the majority but not all its neighbors are from the majority class). In this case, a one-vs-rest scheme is used as proposed in [1]. For instance, a point is now borderline if most but not all its neighbors are from a different class. 

### Generalizing to Real Ratios
Papers often proposes the resampling algorithm using integer ratios. For instance, a ratio of `2` would mean to double the amount of data in a class and a ratio of $2.2$ is not allowed or will be rounded. In `Imbalance.jl` any appropriate real ratio can be used and the ratio is relative to the size of the majority or minority class depending on whether the algorithm is oversampling or undersampling. The generalization occurs by randomly choosing points instead of looping on each point. That is, if a $2.2$ ratio corresponds to $227$ examples then $227$ examples are chosen randomly by replacement then applying resampling logic to each. Given an integer ratio $k$, this falls back to be on average equivalent to looping on the points $k$ times.

[1] Fernández, A., López, V., Galar, M., Del Jesus, M. J., and Herrera, F. (2013). Analysing the classifi-
cation of imbalanced data-sets with multiple classes: Binarization techniques and ad-hoc approaches.
Knowledge-Based Systems, 42:97–110.