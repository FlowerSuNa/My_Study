
# 1. Concept

- `facebook/bart-large-mnli`은 Text Classification 모델로, BART 모델을 기반으로 MNLI 작업에 fine-tuning 된 모델임

- BART는 [Denosing](#denosing) 기법을 seq2seq에 적용시켜 자연어 생성, 번역, 이해를 하는 모델임


### 1) BART

- Bidirectional and Auto-Regressive Transformers

- 페이스북이 만든 AI 모델로, BERT의 이해 능력과 GPT의 생성 능력을 모두 결합한 하이브리드 모델임

    - seq2seq의 Encoder는 BERT(with bidirectional encoder)의 AE 특성을 가지고 있고, Decoder는 GPT(with left-to-right decoder)의 AR 특성을 가지고 있음

- Denosing 기법을 seq2seq에 적용시켜 자연어 생성, 번역, 이해를 하는 모델임

- 구조적으로는 Encoder-Decoder 형태임





### 2) MNLI

- Multi-Genre Natural Language Inference

- MNLI는 entailment, contradiction, neutral로 레이블이 지정된 많은 수의 텍스트 쌍을 포함하는 데이터 세트임
    - 이 데이터 세트는 BART와 같은 모델을 훈련하여 문장 간의 관계를 이해하는 데 사용됨


- Premise 문장과 Hypothesis 문장의 관계를 파악하는 작업을 수행함

- Ex.
| Premise | Hypothesis | Label |
|---|---|---|
| 휴대폰이 빠르게 충전돼요. | 이 제품은 충전 속도가 빨라요. | entailment (포함) |
| 배터리가 빨리 닳아요. | 배터리가 오래 간다. | contradiction (모순) |
| 터치감이 좋아요. | 성능이 좋다. | neutral (중립) |


# 2. BART Thesis Summary 

- 2019년 10월 Facebook에서 발표한 논문임

### 0) Abstract

- BART는 seq2seq 구조를 기반으로 한 [Denosing AutoEncoder](#denosing-autoencoder)임
    - text에 임의의 noise를 주어 훼손시키고, 원본 text로 복구하며 모델을 학습시킴

### 1. Introduction

- Self-supervised 방식은 NLP 작업에서 좋은 성과를 보여줬음




---

# Dictionaly

### Denosing

- Denoising은 데이터에서 불필요한 노이즈를 제거하는 전처리 과정임
- 이미지에서는 배경이 같아도 픽셀 단위의 미세한 RGB 차이가 노이즈로 작용할 수 있음
- 이러한 노이즈를 제거하면 이미지의 일관성이 높아지고, 분석이나 모델 성능이 향상됨
- 텍스트에서도 의미 없는 기호, 중복 표현, 오탈자 등이 노이즈로 간주됨
- 불필요한 요소를 제거함으로써 핵심 정보 전달력이 높아지고, 모델이 더 정확하게 작동함

### Autoencoder

- Autoencoder는 입력 데이터를 압축했다가 다시 복원하는 비지도 학습 모델임
- Encoder는 입력을 잠재 공간(latent space)으로 압축하고, Decoder는 이를 원래 데이터 형태로 복원함
- 주로 차원 축소, 노이즈 제거, 데이터 재구성 등에 활용됨
- 입력과 출력이 동일하도록 학습되며, 그 과정에서 데이터의 중요한 특징을 학습하게 됨
- 변형된 형태로는 Variational Autoencoder(VAE), Denoising Autoencoder 등이 있음

### Denosing Autoencoder

- 일반 Autoencoder는 원본 데이터를 그대로 입력해서 다시 복원하는 방식임
- Denoising Autoencoder는 원본에 일부러 노이즈를 섞은 데이터를 입력으로 사용함
- 출력은 노이즈 없는 원본 데이터를 목표로 학습함
- 즉, DAE는 입력이 손상되더라도 원래 모습을 잘 복원할 수 있도록 학습되는 구조임
- 따라서 일반 Autoencoder보다 더 견고하게 중요한 특징만 잘 추출하는 데 유리함




---

# Reference

- [Hugging Face](https://huggingface.co/facebook/bart-large-mnli)
- [논문](https://arxiv.org/pdf/1910.13461)
- [논문 요약](https://velog.io/@tobigs-nlp/BART-Denoising-Sequence-to-Sequence-Pre-training-for-Natural-Language-Generation-Translation-and-Comprehension)