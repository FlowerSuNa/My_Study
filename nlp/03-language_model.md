# Language Model Overview

- 언어라는 현상을 모델링하고자 단어 시퀀스에 확률을 할당하는 모델
    - = 가장 자연스러운 단어 시퀀스를 찾아내는 모델

## Statistical Language Model (SLM)

### Count Based

- **Conditional Probability**

    - 한 사건이 일어났다는 전체 하에서 다른 사건이 일어날 확률

    - 한 사건 A가 발생했다는 전제 하에서 다른 사건 B가 발생할 확률
        - $ P(A,B) = P(A)P(B|A) $
        - $ P(A) > 0 $

    - **Chain Rule**
        - $ P(x_1, x_2, x_3, ..., x_n) $ <br> $= P(x_1)P(x_2|x_1)P(x_3|x_1, x_2) ... P(x_n|x_1, ...,x_{n-1}) $ <br> $ = \prod_{i=1}^{n} P(x_n|x_1, ..., x_{n-1})$

- 문장(단어 시퀀스) ***An adorable little boy is spreading smiles***의 확률 <br>
    - $ P(An, adorable, little, boy, is, spreading, smiles)$ <br>$ = P(An) \times P(adorable|An) \times P(little|An, adorable) \times P(boy|An, adorable, little) \times P(is|An, adorable, little, boy) \times P(spreading|An, adorable, little, boy, is) \times P(smiles|An, adorable, little, boy, is, spreading) $
    - 단어의 확률은 corpus 내 단어 빈도 수에 기반하여 구할 수 있음

- 빈도 기반 접근의 한계 - Sparsity Problem

    - 기계에서 많은 corpus를 훈련시켜 Language Model이 현실에서의 확률 분포에 근사하는 것이 목표임
    - 빈도 수 기반으로 접근하려면 훈련 데이터는 방대한 양이 필요함
    - corpus에 특정 단어 시퀀스가 없다면 이 확률이 0으로 정의됨
    - 즉, 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제가 발생함 ➡️ **Sparsity Problem**

## N-gram Language Model

- SLM는 훈련 Corpus에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있음
    - 확률을 계산하고 싶은 문장이 길어질수록 훈련 corpus에 그 문장이 존재하지 않을 가능성이 높음
    - 따라서 참고하는 단어 수를 줄이면 카운트 할 가능성을 높일 수 있음

- n개의 연속적인 단어를 사용하여 추론하는 기법이 n-gram 임
    - unigrams, bigrams, trigrams, 4-grams

- EX. 문장 ***An adorable little boy is spreading smiles***
    - unigrams : An, adorable, little, boy, is, spreading, smiles
    - bigrams : An adorable, adorable little, little boy, boy is, is spreading, spreading smiles
    - trigrams : An adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
    - 4-grams : An adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

- n-gram을 통해 다음 단어를 추론할 때는 오직 n-1개의 단어에만 의존함
    - 4-gram 모델을 사용하여 spreading 다음에 올 단어를 예측한다면, boy is spreading 단어 시퀀스를 이용함

- n-gram은 앞의 단어 몉 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생김
    - 문장의 앞부분과 뒷부분의 문맥이 연결 안 될 수도 있음

- 또한, 일부 단어만을 보는 것이 현실적으로 corpus에서 빈도 수 기반 확률 측정을 높일 수는 있지만, 여전히 sparsity problem이 발생함

- 몇 개의 단어를 볼지 n을 정하는 것에 trade-off 문제가 발생함
    - n이 작아지면 language model의 성능이 감소함
    - n이 커지면 sparsity problem이 심해짐
    - 적절한 n을 선택해야하며, 최대 5를 넘지 않는 것을 권장하고 있음

<br>

## 인공 신경망을 이용하는 방법

-  GPT, BERT


## Reference

- [링크1](https://blog.naver.com/mykepzzang/220834864348)