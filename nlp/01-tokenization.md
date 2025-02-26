# Tokenization

: 주어진 corpus에서 의미있는 단위인 token으로 나누는 작업

- English : `nltk` 패키지 사용

- Korean : `kss` 패키지 사용

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

> ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']


### Korean

- 한국어는 띄어쓰기만으로 tokenization 지양함
    - 한국어는 교착어(조사, 어미 등이 붙여져 말을 만드는 언어)이기 때문에 띄어쓰기 단위는 어절 단위임 ➡️ 단어 단위 아님
    - 한국어는 morpheme(형태소) 개념을 반드시 이해해야 함

<br>

## Sentence Tokenization

- 보통 copus가 잘 정제되어 있지 않아 문장 단위로 구분하는 작업이 필요하여 sentence tokenization가 진행됨

### English

```python
from nltk.tokenize import sent_tokenize

text = """
His barber kept his word. But keeping such a huge secret to himself was driving him crazy. 
Finally, the barber went up a mountain and almost to the edge of a cliff. 
He dug a hole in the midst of some reeds. He looked about, to make sure no one was near.
"""
print(sent_tokenize(text))
```

> ['His barber kept his word.', 
> 'But keeping such a huge secret to himself was driving him crazy.', 
> 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 
> 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']

- `LookupError` 발생 시, 추가로 다운로드가 필요함

    ```python
    import nltk
    nltk.download()
    ```

### Korean

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

- `kss` 패키지 설치 시 아래와 같은 오류가 발생하면, 버전을 낮추어 설치 진행해야 함 ➡️ `pip install kss=3.5.4`

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
