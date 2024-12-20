### L1Loss

**Definition**: Measures the mean absolute error between predicted values ($\hat{y}_i$) and target values ($y_i$).

$ L_{L1}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i| $

### MSELoss

**Definition**: Measures the mean squared difference between predicted values ($\hat{y}_i$) and target values ($y_i$).

$ L_{MSE}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 $

### CrossEntropyLoss

**Definition**: Combines softmax and negative log-likelihood to measure the distance between probability distributions.

$L_{CE}(\hat{y}, y) = - \frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$

Where:
*  $y_{i,c}$ represents the true label for the i-th sample for class c
* $\hat{y}_{i,c}$ represents the predicted probability for the i-th sample for class c
* C is the total number of classes.

### CTCLoss

**Definition**: Used for sequence-to-sequence tasks with alignment-free labels.

$L_{CTC} = - \log P(y | \hat{y})$

Where:
* $P(y | \hat{y})$ is the probability of the target sequence ($y$) given the predicted sequence ($\hat{y}$), calculated by summing probabilities over all possible alignments between the two sequences.

### NLLLoss

**Definition**: Measures the negative log-likelihood of a target under a predicted distribution.

$L_{NLL}(\hat{y}, y) = - \sum_{i=1}^n \log(\hat{y}_{i, y_i})$

Where:
* $\hat{y}_{i, y_i}$ represents the predicted probability of the true class label ($y_i$) for the i-th sample.

### PoissonNLLLoss

**Definition**: Used for count data modeled by a Poisson distribution.

$ L_{PoissonNLL}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n \hat{y}_i - y_i \log(\hat{y}_i) $


### GaussianNLLLoss

**Definition**: Models a Gaussian distribution for target data, taking into account the predicted mean ($\hat{y}_i$) and variance ($\sigma_i^2$)

$L_{GaussianNLL}(\hat{y}, y, \sigma) = \frac{1}{n} \sum_{i=1}^n \left(\frac{(\hat{y}_i - y_i)^2}{2\sigma_i^2} + \log(\sigma_i^2)\right)$


### KLDivLoss

**Definition**: Measures the difference between two probability distributions, P and Q.

$L_{KL}(P, Q) = \sum_{i=1}^n P(x_i) \log\frac{P(x_i)}{Q(x_i)}$

Where:
* $P(x_i)$ represents the probability of event $x_i$ under distribution *P*
* $Q(x_i)$ represents the probability of event $x_i$ under distribution *Q*.

### BCELoss

**Definition**: Used for binary classification tasks, measuring the difference between predicted probabilities ($\hat{y}_i$) and target labels ($y_i$).

$L_{BCE}(\hat{y}, y) = - \frac{1}{n} \sum_{i=1}^n \left(y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right)$

### BCEWithLogitsLoss

**Definition**: Combines a sigmoid layer and binary cross-entropy loss for numerical stability. Applies the sigmoid function ($\sigma$) to the predicted values ($\hat{y}$) before computing the BCELoss.

$L_{BCEWithLogits}(\hat{y}, y) = L_{BCE}(\sigma(\hat{y}), y)$

### MarginRankingLoss

**Definition**: Used for ranking tasks, where the goal is to learn a ranking function that assigns higher scores to more relevant items. The loss encourages the difference between the scores of a positive example ($x_1$) and a negative example ($x_2$) to be greater than a specified margin.

$ L_{Margin}(x_1, x_2, y) = \max(0, -y (x_1 - x_2) + \text{margin}) $

Where: 
* $y$ is a binary indicator (1 or -1) indicating whether $x_1$ should be ranked higher than $x_2$
* *margin* is a hyperparameter that determines the desired separation between positive and negative examples. 

### HingeEmbeddingLoss

**Definition**: Measures the loss for binary classification tasks with embedding vectors. Encourages the norm of the embedding vector ($||x||$) to be large for positive examples (y = 1) and small for negative examples (y = -1), with a margin separating the two cases.

$L_{HingeEmbedding}(x, y) = \begin{cases} \|x\|, & \text{if } y = 1, \\ \max(0, \text{margin} - \|x\|), & \text{if } y = -1. \end{cases}$

### MultiLabelMarginLoss

**Definition**: Calculates the loss for multi-label classification tasks with integer labels. For each correct class label ($y_{i,j}$ = 1), it encourages the predicted score for that class ($\hat{y}_i$) to be higher than the predicted scores for all incorrect classes ($\hat{y}_j$) by at least a margin.

$ L_{MultiLabelMargin}(\hat{y}, y) = \frac{1}{C} \sum_{i=1}^C \sum_{j \neq i} \max(0, 1 - (\hat{y}_i - \hat{y}_j) y_{i,j}) $

Where:
* C represents the number of classes.

### HuberLoss

**Definition**: Combines L1 and L2 loss, providing robustness to outliers. For small errors (absolute difference between predicted and target values less than or equal to δ), it behaves like MSE Loss, while for larger errors, it behaves like L1 Loss.

$L_{Huber}(\hat{y}, y) = \begin{cases} \frac{1}{2}(\hat{y} - y)^2, & \text{if } |\hat{y} - y| \leq \delta, \\ \delta |\hat{y} - y| - \frac{\delta^2}{2}, & \text{otherwise}. \end{cases}$

Where:
* δ is a hyperparameter that controls the transition point between the L2 and L1 loss regions.

### SmoothL1Loss

**Definition**:  A variant of Huber Loss commonly used in object detection, also providing robustness to outliers. It smoothly transitions between L2 Loss for small errors (absolute difference between predicted and target values less than 1) and L1 Loss for larger errors.

$L_{SmoothL1}(\hat{y}, y) = \begin{cases} \frac{1}{2}(\hat{y} - y)^2, & \text{if } |\hat{y} - y| < 1, \\ |\hat{y} - y| - \frac{1}{2}, & \text{otherwise}. \end{cases}$

### SoftMarginLoss

**Definition**: Allows for soft margins in binary classification tasks, employing a smoother penalty compared to the hinge loss. 

$ L_{SoftMargin}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i \hat{y}_i)) $

Where:
* $y_i$ represents the true label (1 or -1) for the i-th sample
* $\hat{y}_i$ represents the predicted score for the i-th sample.

### MultiLabelSoftMarginLoss

**Definition**: Extends soft margin loss to multi-label classification. For each sample and each class, it calculates the binary cross-entropy loss between the predicted probability for that class ($\sigma(\hat{y}_i)$) and the true label ($y_i$).

$L_{MultiLabelSoftMargin}(\hat{y}, y) = - \frac{1}{n} \sum_{i=1}^n \left( y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right)$

### CosineEmbeddingLoss

**Definition**: Measures the loss between embeddings based on their cosine similarity. It encourages the cosine similarity to be high for positive pairs (y = 1) and low for negative pairs (y = -1).

$L_{Cosine}(x_1, x_2, y) = \begin{cases} 1 - \cos(x_1, x_2), & \text{if } y = 1, \\ \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1. \end{cases}$

Where:
* $x_1$ and $x_2$ are the embedding vectors being compared
* $\cos(x_1, x_2)$ is their cosine similarity.

### MultiMarginLoss

**Definition**: Used for classification tasks with multiple margins. For each sample, it encourages the predicted score for the correct class to be higher than the predicted scores for all other classes by at least a margin.

$L_{MultiMargin}(\hat{y}, y) = \frac{1}{n} \sum_{i \neq y} \max(0, \text{margin} + \hat{y}_i - \hat{y}_y)$

Where: 
* $\hat{y}_y$ is the predicted score for the true class
* $\hat{y}_i$ is the predicted score for other class (i).

### TripletMarginLoss

**Definition**: Aims to ensure that an anchor data point is closer to a positive data point (from the same class) than it is to a negative data point (from a different class) by at least a specified margin.

$ L_{Triplet}(a, p, n) = \max(0, \|a - p\|_2^2 - \|a - n\|_2^2 + \text{margin}) $

Where: 
* $a$ represents the anchor data point
* $p$ represents the positive data point 
* $n$ represents the negative data point.

### TripletMarginWithDistanceLoss

**Definition**: Generalizes the triplet margin loss by allowing the use of a custom distance function ($d$) instead of the Euclidean distance. 

$ L_{TripletDistance}(a, p, n) = \max(0, d(a, p) - d(a, n) + \text{margin}) $

Where:
* $a$ is the anchor data point
* $p$ is the positive data point 
* $n$ is the negative data point. 

