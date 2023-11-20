## **TreeModelsLab**

The goal of this project was to implement tree-based machine learning models from scratch. This version includes Decision Trees, Random Forests and Gradient Boosting algorithms. Decision trees and random forests supports both regression and classification and gradient boosting supports only regression.

The models are implemented in Python and the code is organized into separate folders for each model.

The notebooks are not well explained, I admit, but I hope that the codes in the notebooks are straightforward.

## Table of contents

- Decision Trees
- Random Forests
- Gradient Boosting
- Limitations
- References

## Decision Trees

The decision_tree folder contains the implementation of the Decision Tree algorithm.

`build_tree(X, y, current_depth)`: A function that builds a decision tree model with the specified parameters. The criterion argument determines the quality of a split. The max_depth limits the maximum depth of the tree, while the min_samples_split argument sets the minimum number of samples required to split an internal node.

**gini_index**: Gini index measures the probability of a random sample being incorrectly classified based on the distribution of the target variable. It ranges from 0 to 1, where 0 represents a perfect split (all samples belong to the same class) and 1 represents a random split (samples are equally distributed among all classes).

**entropy**: Entropy measures the impurity of a split by calculating the information gain, which is the reduction in entropy of the target variable after the split. It ranges from 0 to 1, where 0 represents a perfect split (all samples belong to the same class) and 1 represents a random split (samples are equally distributed among all classes).

**variance**: Variance measures the homogeneity of a split by calculating the variance of the target variable within each node. It ranges from 0 to 1, where 0 represents a perfect split (all samples have the same target value) and 1 represents a random split (samples have different target values within each node).

**mae**: Mean Absolute Error (MAE) measures the average absolute difference between the predicted and actual target values. It is a loss function commonly used in regression problems.

**poisson**: Poisson deviance is a measure of the goodness of fit of a Poisson regression model. It is a loss function commonly used in count data regression problems

`fit(X, y)`: A method that trains the decision tree model on the input data X and target values y.

`predict(X)`: A method that predicts the target values for the input data X based on the trained decision tree model.

`predict_proba(X)`: A method that returns the class probabilities for the input data X based on the trained decision tree model. This method is only applicable for classification problems.

`feature_importance()`: A method that computes the feature importance scores of each feature used in the decision tree model. The method argument specifies the feature importance measure to be used, with "impurity_decrease" being the default option and "n_splits" being the alternative.

## Random Forests

The main idea behind Random Forests is to randomly select a subset of features from the dataset and a subset of data points from the training set with replacement (i.e., bootstrap) to build each decision tree. This is known as bootstrap aggregating, or bagging. This process is repeated multiple times to create a "forest" of decision trees, where each tree is trained on a different subset of the data. The final prediction is made by aggregating the predictions of all the individual trees, either by majority voting in classification tasks or by averaging in regression tasks. This approach helps to reduce overfitting and improve the generalization performance of the model.

`n_estimators`: This parameter is the number of decision trees to be constructed in the random forest. Increasing the number of trees can improve the accuracy of the model, but it also increases the computational cost and may lead to overfitting.

`max_features`: In a random forest, each tree is constructed using a subset of the available features. This is done to reduce the correlation between the trees and make the forest more diverse. The max_features parameter determines the maximum number of features that can be used in each tree. In general, a smaller value of max_features can help to reduce overfitting, while a larger value can help to improve accuracy. This parameter may need to be tuned through experimentation. By default max_features is set to 'sqrt', then the maximum number of features used in each tree will be the square root of the total number of features.

## Gradient Boosting

Gradient Boosting is an iterative algorithm that combines several weak models (e.g., decision trees) to form a strong predictive model. Unlike bagging (used in Random Forests), which builds several models independently and then averages their predictions, boosting builds models sequentially, with each subsequent model attempting to correct the errors of the previous one.

The basic idea behind Gradient Boosting is to minimize a loss function by iteratively adding models to the ensemble. At each step, the algorithm fits a new model to the pseudo-residuals of the current model. The pseudo-residuals are the negative gradient (partial derivative) of the loss function with respect to the predicted values. The new model is then added to the ensemble with a weight that depends on the learning rate (all models have the same learning rate in this implementation), which controls the contribution of each model to the final prediction.

The current implementation don't supports classification, but I will implement in the future (feel free to contribute btw).

The general formula for Gradient Boosting is:

    F_0(x) = argmin_c sum(L(y_i, c))

where F_0(x) is the initial prediction, argmin_c finds the value of c that minimizes the loss function L for the training set, and y_i is the target variable for the i-th sample in the training set.

To find the value of c that minimizes the loss function L, we can take the derivative of L with respect to c, set it to zero, and solve for c.
For regression it's the same of calculating the mean value if the loss is the loss detailed below. Thus, at the first step the algorithm starts predicting the y mean for all samples.

The algorithm also needs a differentiable loss function, which for regression is usually:

    L = 1/2 * (y_i - y_i_pred) ^ 2

And it's gradient is:

    dL / dy_i_pred = -(y_i - y_i_pred)

The algorithm then proceeds to iteratively improve the model by adding new models to the ensemble. At each iteration t, the algorithm computes the negative gradient of the loss function with respect to the predicted values, which gives the pseudo-residuals:

    r_i(t) = -[dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)]

where F_{t-1}(x_i) is the prediction of the ensemble at iteration t-1, and dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i) is the derivative of the loss function with respect to the predicted values.

The algorithm then fits a new model g_t(x) to the pseudo-residuals r_i(t) (r_i(t) as function of x), and adds it to the ensemble with a weight determined by the learning rate *alpha*:

    F_t(x) = F_{t-1}(x) + alpha * g_t(x)

where F_{t-1}(x) is the prediction of the ensemble at iteration t-1.

In each node Jm of each g_t(x) the predicted values (*gamma*) are found by:

    gamma_j = argmin_gamma [ L(y_i, F_{t-1}(x_i) + gamma) ]

where gamma_j is the optimal predicted value for the j-th child node, L is the loss function, y_i is the target variable for the i-th sample, F_{t-1}(x_i) is the predicted value of the ensemble up to the (t-1)-th iteration for the i-th sample, and gamma is the predicted value for the current node.

After the optimal split point is found, the predicted values for the child nodes are set to be the optimal predicted values that minimize the loss function in each child node. These optimal predicted values are denoted as gamma_j for the j-th child node. For regression, it's the mean if the loss is L = 1/2 * (y_i - y_i_pred) ^ 2.

Therefore, gamma_j is computed at each node of the tree as the predicted value that minimizes the loss function in the child node, given the current ensemble up to the (t-1)-th iteration.

The process is repeated until a stopping criterion (e.g., a maximum number of iterations or a minimum improvement in performance) is met.

## Limitations

It's important to note that this project is primarily intended for educational and experimental purposes, and is not optimized for real-life applications. As such, the following limitations should be taken into consideration:

- The use of a recursive implementation of decision trees in Python can cause errors when dealing with large datasets due to the recursion limit. Users may need to adjust the maximum recursion depth.

- The decision tree implementation used in this project does not include stopping parameters to help mitigate overfitting such as minimum samples per leaf and maximum number of features per split.

- Classification was not implemented for Gradient Boosting in this project. 

- The learning rate in Gradient Boosting is fixed for all steps.

It's worth noting that there are many other limitations and considerations that should be taken into account. However, these limitations are particularly relevant for this specific project.

## References

- Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324

- Jerome H. Friedman. "Greedy function approximation: A gradient boosting machine.." Ann. Statist. 29 (5) 1189 - 1232, October 2001. https://doi.org/10.1214/aos/1013203451 

- https://scikit-learn.org/stable/