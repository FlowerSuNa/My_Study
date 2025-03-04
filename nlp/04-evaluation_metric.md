# Language Model Evaluation Metric

## PPL (Perplexity)

- PPL은 Language Model을 평가하기 위한 평가 지표로, 낮은 수치일수록 좋은 성능을 의미함

- 문장 $ W $의 길이가 $ N $이라고 가정하였을 때, <br> $ PPL(W) = P(w_1, w_2, w_3, ..., w_N)^{-\frac{1}{N}} $

- 문장의 확률에 chain rule을 적용하면, <br> $ PPL(W) = {\prod_{i=1}^{N}P(w_i|w_1, w_2, ..., w_{i-1})}^{-\frac{1}{N}} $

- n-gram에도 적용할 수 있음

    - bigram의 경우, $ PPL(W) = {\prod_{i=1}^{N}P(w_i|w_{i-1})}^{-\frac{1}{N}}$

- PPL은 선택할 수 있는 가능한 경우의 수를 의미함 ➡️ branching factor (분기 계수)

    - Language Model이 특정 시점에서 평균적으로 몇 개의 선택지를 가지고 고민하고 있는지를 의미함

    - 단, PPL 값이 낮다는 것은 테스트 데이터 상에서 높은 정확도를 보인다는 것이지, 사람이 직접 느끼기에 좋은 Language model 이라는 것을 의미하지 않음
    