# LightGBM : A Highly Efficient Gradient Boosting Decision Tree

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu

<br>

### Abstract


<br>

---

### Introduction

* Gradient Boosting Decision Tree [(GBDT)](Gradient_Boosting.md)
  * a widely-used machine learning algorithm, due to its efficiency, accuracy, and interpretability
  * a few effective implementations such as XGBoost and pGBRT
  
* But the efficiency and scalability are stil unsatisfactory when 
  * the feature dimension is high
  * data size is large

**Need to**, <br>
for every feature, scan all the data instances to estimate the information gain of all the possible split points.


