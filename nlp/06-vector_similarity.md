# Vector Similarity

- 문서들 간에 얼마나 비슷한 단어가 사용되는지나 단어 벡터 간의 유사도 확인 시에 vector similarity를 계산함

## Cosine Similarity

- 두 벡터 간의 코사인 각도를 구할 수 있음

    - 두 벡터의 방향이 완전히 동일하면 1, 
    - 90도 각도를 이루면 0,
    - 180도 반대 방향이면 -1

- 두 벡터 $ A $와 $ B $에 대해 <br> $ cos(\theta) = \frac{A \cdot B}{||A||||B||} = \frac{\sum_{i=1}^{n}A_i \times B_i}{\sqrt{\sum_{i=1}^{n} (A_i)^2} \times \sqrt{\sum_{i=1}^{n} (B_i)^2}} $


<br>

## Euclidean Distance

- 다차원 공간에서 두개의 점 $ p $와 $ q $ 사이의 거리를 계산함

- $ \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$

<br>

## Jaccard Similarity

- 두 개의 집합 $ A $와 $ B $의 합집합에서 교집합의 비율을 계산함

- $ J(A,B) = \frac{|A \cap B|}{|A \cup B|} $