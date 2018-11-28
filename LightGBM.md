# LightGBM : A Highly Efficient Gradient Boosting Decision Tree

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu

<br>

### Abstract

* But the efficiency and scalability are stil unsatisfactory when 
  * the feature dimension is high
  * data size is large


<br>

---

## Introduction

<br>

**Gradient Boosting Decision Tree** [(GBDT)](Gradient_Boosting.md)
* a widely-used machine learning algorithm, due to its efficiency, accuracy, and interpretability
* state-of-the-art-performances in many machine learning tasks
  
<br>

**Need to**, <br>
for every feature, scan all the data instances to estimate the information gain of all the possible split points. <br>
(very time consuming when handling big data)

<br>

### Is it a good idea to reduce the number of data instances and features?

<br> . <br> . <br> . <br> 

### NO!
For example, it is unclear how to perform data sampling for GBDT.

<br>

**Two novel techniques towards this goal**
* Gradient-based One-Side Sampling (GOSS)
* Exclusive Feature Bundling (EFB)

<br>

---

## Preliminaries

<br>

* GBDT
  * an ensemble model of decision trees, which are trained in sequence.
  * learn the decision trees by fitting the negative gradients in each iteration.
  * the main cost : learning the decision trees
  * the most time-consuming part in learning a decision tree is to find the best split points
  
* Pre-sorted Algorithm
  * one of the most popular algorithms to find split points
  * enumerate all possible split points on the pre-sorted feature values
  * simple and can find the optimal split points
  * inefficient in both training speed and memory consumption
  
* Histogram-based Algorithm
  * the popular algorithms to find split points
  * Instead of finding the split points on the sorted feature values, bucket continuous feature values into discrete bins and uses these bins to construct feature histograms during training
  * more efficient in both memory consumption and training speed
  * IMAGE
  
<br>

To reduce the size of the training data, a common approach is to down sample the data instances.

cannot be directly applied to GBDT since there are no native weights for data instances in GBDT.
