---
layout: post
title: Regularization Techniques in Deep Learning
---

In deep learning, the training process is important as it allows for a multi-parameter relationship between the input and the expected output to be learned. There are several techniques, in addition to the structure of the model itself, that have an effect on the accuracy and performance of the model. In this post, three papers will be explored that each present interesting ideas about learning. In one, a loss function is introduced for an application of semi-supervised deep learning that allows for training with minimal labelled data, but still obtains good results. In the second paper, non-convex regularization penalties are explored, with the hope that they would be able to reduce bias that arises from typical convex penalty functions. Finally, generalization in deep learning is explored, which includes an investigation into the role of regularization.


## Regularization Technique for Semi-Supervised Deep Learning

Due to their high complexity, convolutional neural net- works (CNNs) are often used for computer vision tasks, as they are able to achieve state-of-the-art accuracy. With such a large number of parameters, there is a risk of overfitting with insufficiently large training datasets. Training data is difficult to obtain because it usually requires manual annotation. There is an interest in using unlabeled images for in semi- supervised techniques to get good accuracy without the need of hand labelling.

[Sajjadi, Javanmardi and Tasdizen](https://arxiv.org/abs/1606.04586) propose two loss functions for use during semi-supervised training. A CNN model should make the same prediction for a given training sample, even if the sample has undergone a random transformation (e.g., linear or non-linear data augmentation), or the model has been perturbed (e.g., stochastic gradient descent, dropout, randomized pooling). The proposed loss functions enforce that predictions be the same, regardless of changes to the training sample or the model. The technique is unsupervised because the correct prediction does not need to be known.

Assume a dataset of \\( N \\) training samples and \\( C \\) classes. \\( \mathbf{f}^j(\mathbf{x}_i) \\) is the classifier's prediction vector on the \\( i \\)th training sample during the \\( j \\)th pass through the network and each training sample is passed \\( n \\) times through the network. \\( T^j(\mathbf{x}_i) \\) is a random transformation on \\( \mathbf{x}_i \\).
The first loss function is for transformation stability and is as follows:

$$ \begin{equation}
\label{eq:1}
l^{TS} = \sum_{i=1}^{N} \sum_{j=1}^{n-1} \sum_{k=j+1}^{n} 
\| \mathbf{f}^{i}(T^j(\mathbf{x}_i)) - \mathbf{f}^k(T^k(\mathbf{x}_i)) \| _2^2
\end{equation} $$

\\( T^j \\) produces a different input each pass through the network and the loss function attempts to minimize the sum of squared differences between each pair of predictions.

The second loss function is for mutual exclusivity and is as follows:

$$
\begin{equation}
\label{eq:2}
    l^{ME} = \sum_{i=1}^N \sum_{j=1}^n \left( -\sum_{k=1}^C f_k^j(\mathbf{x}_i)
    \prod_{l=1, l \neq k}^C (1 - f_l^j(\textbf{x}_i))\right)
\end{equation}
$$

Where \\( f_k^j(\mathbf{x}_i) \\) is the \\( k \\)th element of prediction vector \\( f^j(\mathbf{x}_i)\\). Again, this loss function attempts to reduce the sum of squared difference between classifications due to perturbations in the model.

It has been found that by using a weighted sum of the two loss functions in equations \eqref{eq:1} and \eqref{eq:2}, as in equation \eqref{eq:3} further improvements in accuracy can be obtained. The authors report that they were able to achieve the best results with \\( \lambda_1 = 0.1 \\) and \\( \lambda_2 = 1 \\). A close value to the state-of-the-art error rate is obtained on MNIST with only 100 labeled samples.

$$
\begin{equation} \label{eq:3}
    l = \lambda_1l^{ME} + \lambda_2l^{TS}
\end{equation}
$$

## Deep Learning with Non-Convex Regularization

Regularization in deep learning is done to prevent overfitting of the model to the training data. The two most common methods are using the \\( L_1 \\) and \\( L_2 \\) penalties. Both of these are convex, which means that, during optimization, a local optima will always be a global optima. 

However, these two regularization methods can introduce bias into training. The use of \\( L_1 \\) sets many parameters to zero and the use of \\( L_2 \\) shrinks all parameters towards zero. Non-convex penalty functions may reduce this bias. It has also been recently shown that local optima from optimization of non-convex regularizers are as good as a global optima from a theoretical statistical perspective, since the distance from the global optima is not statistically significant.

[Vettam and John](https://arxiv.org/abs/1909.05142) compare the performance of four non-convex regularization functions (Laplace (equation \eqref{eq:4}), Arctan (equation \eqref{eq:5}), SCAD and MCP), \\( L_1 \\), \\( L_2 \\), and training with no regularization penalty.

$$
\begin{equation} \label{eq:4}
    \sum_{j=1}^p p_{\theta}(w_i) = \lambda\sum_{j=1}^p(1-\epsilon^{|w_i|}), \theta = (\lambda,\epsilon), \epsilon \in (0,1), \lambda >0
\end{equation}
$$

$$
\begin{equation} \label{eq:5}
    \sum_{j=1}^p p_{\theta}(w_i) = \lambda\sum_{j=1}^p \frac{2}{\pi} \arctan(\gamma |w_i|), \theta = (\lambda,\gamma), \gamma > 0, \lambda >0
\end{equation}
$$

Training on the MNIST dataset gives comparable test errors for Arctan and Laplace as the convex penalties. There is no further exploration into the effects of bias on the results, so it is unknown whether these non-convex penalty functions actually have any advantages over convex functions.

## Understanding Generalization in Deep Learning

In this [paper by Zhang et al.](https://arxiv.org/abs/1611.03530), the authors set out to understand generalization in deep learning. Despite deep networks having many more parameters than training samples, some are still able to have a small generalization error (the difference between training error and test error). It is also quite easy to train a network that generalizes poorly. It has been suggested that explicit and/or implicit regularization is needed to reduce generalization error.

One of the main findings of the paper is that deep neural networks are easily able to learn and converge with random labels. It is possible to train on a randomly labelled dataset and obtain zero training error and a small increase in training time. Neural networks are also able to learn with input images that are just random noise. This shows that stochastic gradient descent is able to learn a relationship between the images and labels, even though no relationship exists.

It is also not possible to explain the generalization error of a network based exclusively on explicit regularization (e.g., weight decay, dropout, and data augmentation). Including regularization during the training of a model improves performance, but models that have no regularization are still somewhat able to generalize. Testing was performed with the CIFAR10 and ImageNet datasets and Inception, Alexnet and MLPs as the deep neural networks. When regularization is properly tuned, generalization is improved, but it cannot fully explain the generalization.

The authors propose that stochastic gradient descent (SGD) can also act as an implicit regularizer in linear models, as it will often converge to the solution with minimum norm.

## Conclusions

Three papers were explored in this report. For semi-supervised image classification tasks, a loss function that attempts to minimize the differences in model output between classes for different passes of the input through the model seems to improve performance. Non-convex penalty functions for regularization were able to obtain similar results to traditional convex penalty functions. These non-convex penalty functions are supposed to reduce the bias in the weights that typically arise with convex penalty functions. The role of regularizers for generalization is poorly understood, and models are still able to achieve some level of generalization, even without regularizers. These new and interesting ideas can be implemented when training and should be explored in further experiments.
