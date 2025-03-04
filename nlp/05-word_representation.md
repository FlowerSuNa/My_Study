# Word Representation

## Bag of Words (BoW)

- BoW는 단어들의 순서를 고려하지 않고, 단어들의 출현 빈도에만 집중하여 텍스트 데이터를 수치화 방법임

- Ex. 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.
    - 단어 : 정부, 가, 발표, 하는, 물가상승률, 과, 소비자, 느끼는, 은, 다르다
    - BoW : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]

- BoW는 각 단어가 등장한 횟수를 표기하므로 주로 문서가 어떤 성격의 문서인지 판단하는 작업에 쓰임

    - 즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰임

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
bow = vector.fit_transform(corpus).toarray()

print('vocabulary :',vector.vocabulary_)
print('bag of words vector :', bow) 
```

```output
vocabulary : {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2,  because': 0}
bag of words vector : [[1 1 2 1 2 1]]
```

📌 띄어쓰기를 기준으로 단어 토큰화가 진행되므로, 한국어에는 적합하지 않음

<br>

## DTM (Document-Term Matrix)

- DTM은 BoW 표현을 다수의 문서에 대해 행렬로 표현한 방법임

- 매우 간단하게 구현할 수 있지만, **sparse representation** 문제가 있음

    - One-hot vector는 공간적 낭비와 계산 리소스 증가의 문제가 있는데, DTM도 같은 문제를 지님
    - 방대한 corpus에서 전체 단어 집합의 크기는 크지만, 대부분의 값이 0일 수 있기 때문임

- 또한, 문서에서 불용어가 높은 빈도수를 가지게 되면 중요한 단어와 혼재되어 표기될 수 있음
    - 예를 들어, 영어에서 문서간 the의 빈도수가 높아 유사한 문서라고 판단하는 경우가 생길 수 있음
    - 불용어와 중요한 단어에 대해 가중치를 줄여 줄 수 있는 방법이 필요함

<br>

## TF-IDF (Term Frequency-Inverse Document Frequency)

- TF-IDF는 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법임
    - DTM을 만든 후, TF-IDF 가중치를 부여함

- TF-IDF는 문서의 유사도 계산, 검색 시스템에서 검색 결과의 중요도 계산, 문서 내 특정 단어의 중요도 계산 등의 작업에 쓰일 수 있음

- 문서를 $ d $, 단어를 $ t $, 문서의 개수를 $ n $이라고 표현할 때
    - $ TF(d, t) $ : 특정 문서 $ d $에서의 특정 단어 $ t $의 등장 횟수
    - $ DF(t) $ : 특정 단어 $ t $가 등장하는 문서의 수
    - $ IDF(t) = ln(\frac{n}{1+DF(t)})$
        - 분모에 1을 더하는 이유 : $ DF(t) $가 0이 나올 수도 있기 때문임
        - $ ln $을 사용하는 이유 : 문서의 개수 $ n $이 커질수록 희귀 단어에 엄청난 가중치가 부여될 수 있음 ➡️ $ \frac{n}{1+DF(t)} $ 값이 급격히 커질 수 있음
    - $ TFIDF(d, t) = TF(d, t) \times IDF(t) $

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.vocabulary_)
print(tfidfv.transform(corpus).toarray())
```

```output
 {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
 [[0.         0.46735098 0.         0.46735098 0.         0.46735098 0.         0.35543247 0.46735098] 
  [0.         0.         0.79596054 0.         0.         0.         0.         0.60534851 0.        ]
  [0.57735027 0.         0.         0.         0.57735027 0.         0.57735027 0.         0.        ]]
```

<br>

# Word Embedding

- 단어를 dense vector 형태로 표현하는 방법임 ➡️ distributed representation

    - One-hot encoding 방법은 단어간 관계를 표현할 수 없음 (cosine similarity 모두 0임)

    - DTM, TF-IDF도 sparse matrix으로 단어의 의미를 표현하지 못함

<br>

## Word2Vec

### CBOW (Continuous Bag of Words)

- 주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법

### Skip-gram

- 중간에 있는 단어들을 입력으로 주변 단어들을 예측하는 방법

<br>

## GloVe

## FastText

<br>

# Subword Tokenizer

- 기계에서 아무리 많은 단어를 학습시켜도 모르는 단어가 등장하며, 주어진 문제를 풀기 까다로워짐 ➡️ OVV(Out-of-Vocabulary) 문제

- Subword segmentation 작업은 하나의 단어를 더 작은 단위의 의미 있는 여러 subword로 분리해서 단어를 embedding 하는 전처리 방법임

- 실제로 희귀 단어나 신조어는 subword segmentation 시도 시, 어느 정도 완화하는 효과가 있음

<br>

## BPE (Byte Pair Encoding)

## WordPrice Tokenizer

- BERT를 훈련하기 위해서 사용되기도 함

## Unigram Language Model Tokenizer

- 구글은 BPE 알고리즘과 Unigram Language Model Tokenizer를 구현한 `SentencePiece`를 깃허브에 공개함
    - 사전 tokenization 작업 없이 subword segmentation을 수행할 수 있음 ➡️ 언어 종속성 없음