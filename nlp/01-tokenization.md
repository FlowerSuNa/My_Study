# Tokenization

: 주어진 corpus에서 의미있는 단위인 token으로 나누는 작업

- English : `nltk` 패키지 사용

- Korean : `konlpy`, `kss` 패키지 사용

## Word Tokenization

### - English

- `-`(hyphen)으로 구성된 단어는 하나의 token으로 유지해야 함
- doesn't와 같은 `'`(apostrophe)로 접어가 함께하는 단어는 분리해야 함

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = """
Starting a home-based restaurant may be an ideal. 
it doesn't have a food chain or restaurant of their own.
"""
print(tokenizer.tokenize(text))
```

> ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', <br>
> 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', <br>
> 'restaurant', 'of', 'their', 'own', '.']


### - Korean

- 한국어는 띄어쓰기만으로 tokenization 지양함
    - 한국어는 교착어(조사, 어미 등이 붙여져 말을 만드는 언어)이기 때문에 띄어쓰기 단위는 어절 단위임 ➡️ 단어 단위 아님
    - 또한, 한국어는 띄어쓰기가 English 보다 어려우며, 잘 지켜지지 않음

- 한국어는 morpheme(형태소, 뜻을 가진 가장 작은 말의 단위) 개념을 반드시 이해해야 함
    - English의 word tokenization과 유사한 형태는 한국어의 ***morpheme tokenization***임

- `konlpy` 패키지에는 morpheme 분석기로 Okt, Mecab(메캅), Komoran(코모란), Hannanum(한나눔), Kkma(꼬꼬마)가 있음
    - 각 분석기의 성능과 결과가 다르므로, 적절히 판단하여 사용해야 함
    - 속도가 중요하다면, Mecab을 사용할 수 있음

```python
from konlpy.tag import Okt

okt = Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

> ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']

<br>

## Part-of-speech tagging (품사 태깅)

- 단어는 표기는 같지만 품사에 따라 의미가 달라지는 경우가 있음
    - fly : 날다 (동사), 파리 (명사)
    - 못 : 망치를 사용해서 목재 따위를 고정하는 물건 (명사), 동작 동사를 할 수 없다는 의미 (부사)

- 따라서, word tokenization 과정에서 각 단어가 어떤 품사로 쓰이는지 구분하기도 함 ➡️ ***Part-of-speech tagging***

### - English

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print(pos_tag(tokenized_sentence))
```

> [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), <br>
> ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), <br>
> ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), <br>
> ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]

### - Korean

```python
from konlpy.tag import Okt

okt = Okt()
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

> [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), <br>
> (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), <br>
> ('을', 'Josa'), ('가봐요', 'Verb')]

<br>

➕ Nouns (명사) 추출

```python
from konlpy.tag import Okt

okt = Okt()
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

> ['코딩', '당신', '연휴', '여행']

<br>

## Sentence Tokenization

- 보통 corpus가 잘 정제되어 있지 않아 문장 단위로 구분하는 작업이 필요하여 sentence tokenization가 진행됨

### - English

```python
from nltk.tokenize import sent_tokenize

text = """
His barber kept his word. But keeping such a huge secret to himself was driving him crazy. 
Finally, the barber went up a mountain and almost to the edge of a cliff. 
He dug a hole in the midst of some reeds. He looked about, to make sure no one was near.
"""
print(sent_tokenize(text))
```

> ['His barber kept his word.', <br>
> 'But keeping such a huge secret to himself was driving him crazy.', <br>
> 'Finally, the barber went up a mountain and almost to the edge of a cliff.', <br>
> 'He dug a hole in the midst of some reeds.', <br>
> 'He looked about, to make sure no one was near.']

<br>

⚠️ `LookupError` 발생 시, 추가로 다운로드가 필요함

```python
import nltk
nltk.download()
```

<br>

### - Korean

```python
import kss

text = """
딥 러닝 자연어 처리가 재미있기는 합니다.
그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.
이제 해보면 알걸요?
"""
print(kss.split_sentences(text.replace('\n', ' ')))

```

> ['딥 러닝 자연어 처리가 재미있기는 합니다.', 
> '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', 
> '이제 해보면 알걸요?']

<br>

⚠️ `kss` 패키지 설치 시 아래와 같은 오류가 발생하면, 버전을 낮추어 설치 진행해야 함 ➡️ `pip install kss=3.5.4`

```
note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

<br>

## Reference

- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/21698)
