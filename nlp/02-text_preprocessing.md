# Text Preprocessing

- Corpus로부터 복잡성을 줄이기 위해 text를 전처리 해야함


## Cleaning (정제)

: 가지고 있는 corpus로부터 noisy data를 제거하는 것임

    - Noisy data : 분석하고자 하는 목적에 맞지 않는 불필요한 단어
    - 불용어, 등장 빈도가 적은 단어, 길이가 짧은 단어를 제거하는 방법이 있음

### 1. 불용어 제거

- 자주 등장하지만 분석에 있어 큰 의미가 없는 token을 제거하는 작업업

### 2. 등장 빈도가 적은 단어 제거

- Corpus에서 너무 적게 등장해서 자연어 처리에 도움이 되지 않는 단어는 제거하여 문서 내 단어 수를 줄임

### 3. 길이 짧은 단어 제거

- 영어권 언어에서 길이가 짧은 단어는 대부분 불용어에 해당됨

- 한국어는 글자 하나에 함축적 의미를 담는 경우가 많아 사용하지 않는 방법임

## Normalization (정규화)

: 표현 방법이 다른 단어들을 통합시켜 같은 단어로 만들어 주는 것임 (대,소문자 통합 등)

### 1. Lemmatization (표제어 추출)

- Lemmatization은 단어들이 다른 형태를 가지더라고, 그 뿌리 단어를 찾아주는 역할을 함

    - am, are, is는 서로 다른 스펠링이지만, 그 뿌리 단어는 be 임 ➡️ 이 단어의 lemma는 be 임

### 2. Stemming (어간 추출)

## 한국어 전처리 패키지

- PyKoSpacing

    - 띄어쓰기가 되어 있지 않은 문장을 띄어쓰기를 한 문장으로 변환해 주는 패키지

```bash
pip install git+https://github.com/haven-jeon/PyKoSpacing.git

```
