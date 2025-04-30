# BART

- 2019년 10월 Facebook에서 발표한 논문임

### 0. Abstract

- BART는 Seq2seq 구조를 기반으로 한 [Denosing Autoencoder](#denosing-autoencoder)임
    - Text에 임의의 Noise를 주어 변형시킨 후, 원본 Text로 복원하며 모델을 학습시킴
    - 표준 Transformer 기반 MNT(Neural Machine Translation) 구조를 사용함
    - 단순한 구조임에도 BERT와 GPT를 포함한 다양한 사전 학습 방식을 일반화함

- 원본 문장들의 순서를 무작위로 섞고, Text의 [Spans](#Spans)가 하나의 Mask Token으로 치환되는 [In-filling](#22-pre-training-bart) 방식을 사용할 때 가장 우수한 성능을 보였음

- BART는 Text 생성 작업을 위한 Fine-tuning에서 특히 효과적이었지만, 이해력이 요구되는 작업에서도 잘 작동함
    - [GLUE](#glue) 및 [SQuAD](#squad) benchmark에서는 [RoBERTa](#roberta)와 유사한 학습 자원 하에서 비슷한 성능을 기록함
    - 추상적 대화, 질의응답, 요약 작업에서는 최고 성능(최대 6 ROUNGE 포이트 향상)을 달성함
    - 또한, 기계 번역을 위한 역번역([Back-translation](#back-translation)) 작업에서도 Target 언어에 대한 사전 학습만으로 1.1 [BLEU](#bleu) (Bilingual Evaluation Understudy) 향상함

---

### 1. Introduction

- Self-supervised 기법들은 NLP 분야에서 뛰어난 성과를 보여주었음
    - 가장 성공적인 접근법은 무작위로 Masked Text를 복원하도록 학습하는 Denosing Autoencoder, 즉 Masked 언어 모델의 변형들임

- 최근 연구에서는 Masked Token의 분포, 예측 순서, 대체 가능한 문맥을 개선하는 성과를 달성함
    - 하지만 이 기법들은 특정 End Task(Span 예측, 생성 등)에만 집중하여 적용 가능성이 제한적임

- 본 논문에서는 양방향(Bidirectional)과 자기회귀적(Auto-regressive) Transformer를 결합한 사전 학습 모델인 BART를 제안함

- BART는 다양한 End Task에 적용 가능한 Seq2seq 기반 Denosing Autoencoder임
    - 사전 학습은 Text를 임의의 Nosing 함수로 변형한 후, Seq2seq 모델이 원본 Text로 복구하도록 학습하는 방식임

- BART는 표준 Transformer 기반 NMT 구조를 사용함
    - 단순한 구조임에도 양방향 Encoder인 BERT와 단방향(좌→우) Decoder인 GPT를 포함하여 다양한 최신 사전 학습 방식들을 일반화함 (Figure 1)

![figure 1](img/bart-fig-01.png)

**[Figure 1] BERT, GPT, BART 방식 비교**
```
(a) BERT는 임의 Token를 Masking 후 양방향으로 문서를 Encoding하고, Masked Token를 독립적으로 예측하기 때문에 생성 작업에는 적합하지 않음
(b) GPT는 Token을 자기회귀적으로 예측하여 생성 작업에는 적합하지만, 좌측 문맥만 활용할 수 있어 양방향 상호작용을 학습할 수 없음
(c) BART는 Encoder 입력과 Decoder 출력이 일치할 필요가 없어 임의의 Noise 변환을 적용할 수 있음
    - 이때 문서는 일부 Spans가 Masking 기호로 대체되어 변형된 상태임
    - 변형된 문서(왼쪽)는 양방향 모델로 Encoding되고, 이후 원본 문서(오른쪽)로 자기회귀적으로 Decording하여 복원함
    - Fine-tuning 시에는 손상되지 않은 문서를 Encoder와 Decoder에 입력하며, Decoder의 마지막 Hidden State에서 표현(Representation)을 추출해 사용함
```

- 이 방식의 주요 장점은 Noising의 유연성으로, 원본 Text 길이 변경을 포함한 다양한 변형이 가능함

- 다양한 Noising 접근법을 평가한 결과, 원본 문장을 랜덤하게 섞은 후, 길이 0을 포함한 임의 길이의 Spans를 하나의 Mask Token으로 대체하는 새로운 In-filling 방식을 사용하는 것이 가장 우수한 성능 보임
    - 이 접근법은 모델이 전체 문장 길이에 대해 더 깊이 추론하고 입력보다 더 긴 변환을 하도록 강제하여, BERT의 원본 단어 Masking과 NSP(Next Sentence Prediction) 목표를 일반화함

- BART는 Text 생성 작업을 위해 Fine-tuning 되었을 때 특히 효과적이었으며, 이해력이 요구되는 작업에서도 잘 작동함
    - GLUE 및 SQuAD와 같은 Benchmark에서는 RoBERTa와 유사한 학습 자원 하에서 비슷한 성능을 기록하고, 추상적 대화, 질의응답, 요약 작업에서 최고 성능을 달성함
    - 예를 들어, XSum Benchmark에서는 이전 연구 대비 ROUNGE 점수 6점이 향상됨

- 또한, BART는 Fine-tuning에 대한 새로운 접근 방식을 제안함
    - BART 모델 위에 몇 개의 추가 Transformer Layer를 쌓는 새로운 기계 번역 방식을 제안함
    - 이 Layer들은 BART의 전파(Propagation)를 통해 외국어를 Noisy 영어로 번역하도록 학습되며, 이를 통해 BART를 사전 학습된 Target 언어 모델로 활용함

        > - BART는 사전 학습할 때 영어 문장을 손상시키고, 그 손상된 영어를 다시 원래 영어로 복원하는 방식으로 학습된 모델임
        > - 즉, Noisy 영어를 De-noised 영어로 복구를 잘하는 모델임
        > - 따라서 추가된 Transformer Layer로 외국어를 Noisy 영어로 변환해주면, BART는 Noisy 영어를 De-noised 영어로 자연스럽게 복원함 

    - 이 방식은 WMT Romanian-English Benchmark에서 기존 역번역 모델 대비 1.1 BLEU 만큼 성능을 향상시킴

- BART는 다양한 Task에 걸쳐 일관되게 강력한 성능을 보임 

---

### 2. Model

- BART는 손상된 문서를 원본 문서로 복원하는 Denoising Autoencoder임
    - 손상된 Text를 입력으로 받아, 양방향 Encoder와 단반향(좌→우) 자기회귀적 Decoder를 갖춘 Seq2seq 모델로 구현됨
    - 사전 학습은 원본 문서에 대한 Negative Log Likelihood를 최소화하는 방식으로 진행됨

##### 2.1 Architecture

- BART는 표준 Seq2seq Transformer 구조를 사용함
    - 단, GPT와 동일하게 Activation 함수를 ReLU 대신 [GeLU](#gelu)로 변경하고, Parameter를 N(0, 0.02) 분포로 초기화함

- 기본 모델은 6개의 Encoder와 Decoder Layer를 사용하고, 대형 모델은 12개의 Layer를 사용함

- 이 구조는 BERT와 밀접한 관련이 있지만 다음과 같은 차이점이 있음
    - Decoder의 각 Layer는 Encoder의 마지막 Hidden Layer에 대해 추가적으로 Cross-attention을 수행함
    - BERT는 단어 예측 전에 Feed-forword Network를 추가로 사용하지만, BART는 사용하지 않음

- 전체적으로 BART는 동일 크기의 BERT 모델보다 약 10% 더 많은 Parameter를 가짐

##### 2.2 Pre-training BART

- BART는 문서를 손상시킨 후, Decoder 출력과 원본 문서 간의 cross-entropy loss를 최소화하는 방식으로 학습됨

- 특정 Noising 방식에만 최적화된 Denoising Autoencoder와 달리, BART는 다양한 유형의 손상된 문서를 학습에 활용할 수 있음
    - 원본 정보를 모두 잃은 경우, BART는 일반적인 언어 모델과 동일하게 동작함

- 잠재력이 있는 새로운 Noising 변형 기법을 실험함

**Token Masking**
- BERT와 동일하게, 무작위로 Token을 선택하여 `Mask`로 교체함

**Token Delection**
- 입력에서 무작위로 Token을 제거하고, 어떤 위치에서 입력이 누락되었는지 판단함

**Text Infilling**
- Poisson 분포 (λ = 3)를 기반으로 Span 길이를 Sampling하여 무작위로 Text Span을 선택하고, 각 Span을 하나의 `Mask` Token으로 교체함
- 길이가 0인 Span도 `Mask` Token으로 대응될 수 있음
- SpanBERT에서는 Geometric 분포를 통해 Sampling하고, 각 Span을 동일 길이의 `Mask` Token으로 교체하는 방식을 제안함
- Text Infilling은 모델이 하나의 Span으로 누락된 Token 수를 예측하도록 학습함

**Sentence Permutation**
- 문서를 full stop(마침표) 기준으로 문장을 나누고, 그 문장들을 무작위로 섞음

**Document Rotation**
- 문서에서 무작위로 하나의 Token을 선택하여 해당 Token으로 문서가 시작되도록 회전시키고, 모델이 문서의 시작을 식별하도록 학습시킴

![figure 2](img/bart-fig-02.png)

**[Figure 2]** 입력 Text에 대한 여러 형태의 Noisy 변환은 조합하여 적용할 수 있음

---

### 3. Fine-tuning BART

- BART가 생성한 표현은 다양한 Downstream 작업에 활용될 수 있음

![figure 3](img/bart-fig-03.png)

**[Figure 3] 분류 및 번역을 위한 BART의 Fine-tuning 구조**

```
(a) 분류 문제에서 BART는 Encoder와 Decoder에 동일한 입력을 주고, 마지막 출력의 표현을 사용함
(b) 기계 번역에서는 BART 앞에 word embedding을 위한 작은 Encoder를 추가로 학습시키며, 
    이 추가된 Encoder는 별도의 단어 집합(vocabulary)으로 사용할 수 있음
```

##### 3.1 Sequence Classification Tasks

- Sequence 분류 시에는 Encoder와 Decoder에 동일한 입력을 주고, Decoder의 마지막 Hidden State를 Multi-class 분류기에 입력함

- 이 접근법은 BERT의 [CLS](#CLS) Token을 사용하는 방식과 유사하지만, BART는 Decoder 입력 끝에 특정 Token을 추가하여 해당 Token 표현이 전체 입력으로부터 생성된 Decoder의 Hidden State에 Attention 할 수 있도록 함 (Figure 3a)

    > - Decoder 입력 끝에 Token을 하나 더 붙여, 그 Token이 전체 입력 내용을 요약한 표현을 갖도록 만드는 구조임
    >- 이 요약된 Hidden State를 문장 대표 표현으로 씀(?)

##### 3.2 Token Classification Tasks

- Token 분류 시에는 SQuAD의 답변 Endpoint 분류와 유사하게, 전체 문서를 Encoder와 Decoder에 입력한 뒤, 각 단어의 표현이 포함된 Decoder의 상단 Hidden State를 활용해 분류를 수행함

##### 3.3 Sequence Generation Tasks

- BART는 자기회귀적 Decoder를 갖추고 있어, 추상적인 질의응답과 요약과 같은 Sequence 생성 작업을 직접 수행할 수 있음
- 이 두 작업은 정보를 입력으로부터 복사해 활용하지만, Denoising 사전 학습 방식과 밀접하게 연관되어 작동함

    > - BART의 사전 학습 방식은 Noisy 데이터를 복원하는 학습임 (denosing)
    > - 따라서 질의응답이나 요약처럼 입력 일부를 기반으로 출력을 생성하는 작업은 학습 구조 자체가 잘 맞아 떨어짐

- 이때 Encoder 입력은 입력 Sequence이며, Decoder는 자기회귀적 출력을 생성함

##### 3.4 Machine Translation

- BART는 영어로 번역하는 기계 번역 Decoder의 성능을 향상시킴
    - 기존 연구 Edunov(2019)에서는 사전 학습된 Encoder를 통합하여 성능을 높였지만, Decoder의 언어 모델은 이점이 제한적임
- BiText로 학습된 Encoder Parameter Set을 새로 추가하여, BART의 Encoder와 Decoder가 기계 번역을 위한 하나의 사전 학습된 Decoder로 사용 가능함 (Figure 3b)
    - 보다 정확히는, BART의 embedding layer를 무작위로 초기화된 새로운 Encoder로 대체함
    - 추가된 Encoder는 외국어 단어를 De-noised 영어로 변환되도록 학습함

- 추가된 Encoder는 두 단계로 학습하며, 역전파(Backpropagation)는 BART 출력에 대한 Cross-entory Loss임
    - 먼저 BART의 대부분 Parameter를 고정하고, 무작위로 초기화된 Encoder와 BART의 위치 Embedding, BART의 첫 Encoder Layer의 Self-attention 입력인 Projection Matrix를 업데이트함
    - 다음으로 작은 수의 Iteration으로 모든 모델의 Parameter를 학습함 (end-to-end 학습)

---

### 4. Comparing Pre-training Objectives

- BART는 이전 연구 보다 다양한 noising 방식을 사전 학습에 활용할 수 있음

- 기본 모델(6개의 Encoder와 Decoder Layer, 768개의 Hidden)을 사용하여 다양한 선택지를 비교함
    - 5장에서 다룰 대규모 실험의 일부 작업을 기준으로 평가됨

##### 4.1 Comparision Objectives

- 

---

# Dictionaly

### Autoencoder

- Autoencoder는 입력 데이터를 압축했다가 다시 복원하는 비지도 학습 모델임
- Encoder는 입력을 잠재 공간(latent space)으로 압축하고, Decoder는 이를 원래 데이터 형태로 복원함
- 주로 차원 축소, 노이즈 제거, 데이터 재구성 등에 활용됨
- 입력과 출력이 동일하도록 학습되며, 그 과정에서 데이터의 중요한 특징을 학습하게 됨
- 변형된 형태로는 Variational AutoEncoder(VAE), Denoising Autoencoder 등이 있음

### Denosing Autoencoder

- 일반 Autoencoder는 원본 데이터를 그대로 입력해서 다시 복원하는 방식임
- DeNoising AutoEncoder는 원본에 일부러 노이즈를 섞은 데이터를 입력으로 사용함
- 출력은 노이즈 없는 원본 데이터를 목표로 학습함
- 즉, DAE는 입력이 손상되더라도 원래 모습을 잘 복원할 수 있도록 학습되는 구조임
- 따라서 일반 Autoencoder보다 더 견고하게 중요한 특징만 잘 추출하는 데 유리함

### Spans

- 연속된 단어나 문장의 조각을 의미함
- 단어 하나가 아니라 여러 단어가 묶여 있는 덩어리를 가리킴
- 단어 단위가 아니라 구(phrase)나 문장 조각 단위로 다뤄질 수 있음

### RoBERTa

- RoBERTa는 페이스북에서 개발한 BERT 개선 버전 모델임
- BERT랑 구조는 같지만 학습 데이터를 훨씬 많이 사용하고, 마스킹 방식도 더 정교하게 조정함
- NSP(Next Sentence Prediction)을 제거하여 성능을 높였음 <br>(NSP Task가 실제로 언어 이해 능력 향상에는 큰 도움이 안됨을 실험으로 확인함)

### GLUE

- GLUE는 다양한 NLP 과제를 모아놓은 benchmark 테스트임
- 문장 추론, 유사도, 감정 분류 등 총 9개의 Task로 구성되어 있음
- 모델이 얼마나 언어를 잘 이해하는지를 평가하는 데 쓰임
- 많은 모델들이 이 점수를 기준으로 성능을 비교함

### SQuAD

- SQuAD는 QA Task를 위한 데이터셋임
- 주어진 지문에서 사용자가 묻는 질문에 답을 찾아야 하는 형식임
- 정답은 보통 지문 안에 존재하는 문장 일부임
- Stanford에서 만든 데이터셋으로 Stanford Question Answering Dataset의 약자임

### Back-translation

- Back-translation은 기계번역에서 자주 쓰이는 데이터 증강 기법임
- Target language 문장을 source langauge로 역번역하여 새로운 훈련 데이터를 생성함

### BLEU

- 기계번역 품질 평가 지표 중 가장 널리 쓰이는 점수임
- 0~100 사이의 점수로, 높을수록 사람 번역과 유사하다는 의미임
- 모델이 생성한 번역 결과와 정답 간의 n-gram 단위 일치도를 평가함

### GeLU

- Gaussian Error Linear Units
- $ GeLU(x) = xP(X \leq x) =  x \Phi(x) = x \cdot \frac{1}{2}[1+erf(\frac{x}{\sqrt{2}})]$
    - $ \Phi(x) $ : CDF (Standard Gaussian Cumulative Distribution Function)
    - CDF를 사용하여 입력값 크기에 따라 확률적으로 가중치를 부여함
    - $ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} \, dt $ : Error Function
- $ GeLU(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715x^3 \right) \right) \right) $
    - 계산 비용이 비싸기 때문에 $ tanh $ 기반 근사식을 사용하기도 함

### CLS
---

# Reference

- [논문](https://arxiv.org/pdf/1910.13461)
- [논문 요약](https://velog.io/@tobigs-nlp/BART-DeNoising-Sequence-to-Sequence-Pre-training-for-Natural-Language-Generation-Translation-and-Comprehension)
- [논문 요약2](https://velog.io/@dutch-tulip/BART)
- [GeLU](https://jik9210.tistory.com/14)

---

# Original Text

### 0. Abstract

```
We present BART, a deNoising autoEncoder for pretraining sequence-to-sequence models.

BART is trained by (1) corrupting Text with an arbitrary Noising function, 
and (2) learning a model to reconstruct the original Text.

It uses a standard Tranformer-based neural machine translation architecture which, 
despite its simplicity, can be seen as generalizing BERT (due to the bidirectional Encoder), 
GPT (with the left-to-right Decoder), and many other more recent pretraining schemes.

We evaluate a number of Noising approaches, 
finding the best performance 
by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, 
where Spans of Text are replaced with a single Mask Token.

BART is particularly effective when fine tuned for Text generation
but also works well for comprehension Tasks.

It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, 
achieves new stateof-the-art results on a range of 
abstractive dialogue, question answering, and summarization Tasks, with gains of up to 6 ROUGE.

BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, 
with only target language pretraining.

We also report ablation experiments that replicate other pretraining schemes within the BART framework, 
to better measure which factors most influence end-Task performance.
```

### 1. Introduction

```
Self-supervised methods have achieved remarkable success in a wide range of NLP Tasks
(Mikolov et al., 2013; Peters et al., 2018; Devlin et al., 2019; Joshi et al., 2019; Yang et al., 2019; Liu et al., 2019).

The most successful approaches have been variants of Masked language models, 
which are deNoising autoEncoders that are trained to reconstruct Text 
where a random subset of the words has been Masked out.

Recent work has shown gains by improving the distribution of Masked Tokens (Joshi et al., 2019), 
the order in which Masked Tokens are predicted (Yang et al., 2019), 
and the available conText for replacing Masked Tokens (Dong et al., 2019).

However, these methods typically focus on particular types of end Tasks (e.g. Span prediction, generation, etc.), 
limiting their applicability.

In this paper, we present BART, 
which pre-trains a model combining Bidirectional and Auto-Regressive Transformers.

BART is a deNoising autoEncoder built with a sequence-to-sequence model that is applicable
to a very wide range of end Tasks.

Pretraining has two stages (1) Text is corrupted with an arbitrary Noising function,
and (2) a sequence-to-sequence model is learned to reconstruct the original Text. 

BART uses a standard Tranformer-based neural machine translation architecture which, 
despite its simplicity, can be seen as generalizing BERT (due to the bidirectional Encoder),
GPT (with the left-to-right Decoder), and many other more recent pretraining schemes (see Figure 1).

Figure 1: A schematic comparison of BART with BERT (Devlin et al., 2019) and GPT (Radford et al., 2018).

(a) BERT: Random Tokens are replaced with Masks, and the document is encoded bidirectionally.
Missing Tokens are predicted independently, so BERT cannot easily be used for generation.

(b) GPT: Tokens are predicted auto-regressively, meaning GPT can be used for generation.
However words can only condition on leftward conText, so it cannot learn bidirectional interactions.

(c) BART: Inputs to the Encoder need not be aligned with Decoder outputs, allowing arbitary Noise transformations.
Here, a document has been corrupted by replacing Spans of Text with Mask symbols.
The corrupted document (left) is encoded with a bidirectional model,
and then the likelihood of the original document (right) is calculated with an autoregressive Decoder.
For Fine-tuning, an uncorrupted document is input to both the Encoder and Decoder,
and we use representations from the final hidden state of the Decoder.

A key advantage of this setup is the Noising flexibility;
arbitrary transformations can be applied to the original Text, including changing its length.

We evaluate a number of Noising approaches, finding the best performance 
by both randomly shuffling the order of the original sentences and using a novel in-filling scheme,
where arbitrary length Spans of Text (including zero length) are replaced with a single Mask Token.

This approach generalizes the original word Masking and next sentence prediction objectives in BERT 
by forcing the model to reason more about overall sentence length and make longer range transformations to the input.

BART is particularly effective when fine tuned for Text generation but also works well for comprehension Tasks. 

It matches the performance of RoBERTa (Liu et al., 2019) 
with comparable training resources on GLUE (Wang et al., 2018) and SQuAD (Rajpurkar et al., 2016),
and achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization Tasks. 

For example, it improves performance by 6 ROUGE over previous work on XSum (Narayan et al., 2018).

BART also opens up new ways of thinking about fine tuning.

We present a new scheme for machine translation
where a BART model is stacked above a few additional transformer layers.

These layers are trained to essentially translate the foreign language to Noised English,
by propagation through BART, thereby using BART as a pre-trained target-side language model.

This approach improves performance over a strong back-translation MT baseline
by 1.1 BLEU on the WMT Romanian-English benchmark.

To better understand these effects,
we also report an ablation analysis that replicates other recently proposed training objectives. 

This study allows us to carefully control for a number of factors, 
including data and optimization parameters, which have been shown
to be as important for overall performance as the selection of training objectives (Liu et al., 2019). 

We find that BART exhibits the most consistently strong performance across the full range of Tasks we consider.
```

### 2. Model

```
BART is a deNoising autoEncoder that maps a corrupted document to the original document it was derived from.

It is implemented as a sequence-to-sequence model with a bidirectional Encoder 
over corrupted Text and a left-to-right autoregressive Decoder.

For pre-training, we optimize the negative log likelihood of the original document.
```

##### 2.1 Architecture

```
BART uses the standard sequence-to-sequence Transformer architecture from (Vaswani et al., 2017), 
except, following GPT, that we modify ReLU activation functions to GeLUs (Hendrycks & Gimpel, 2016)
and initialise parameters from N (0, 0.02).

For our base model, we use 6 layers in the Encoder and Decoder, and for our large model we use 12 layers in each. 

The architecture is closely related to that used in BERT, with the following differences: 
(1) each layer of the Decoder additionally performs cross-attention over the final hidden layer of the Encoder (as in the transformer sequence-to-sequence model); 
and (2) BERT uses an additional feed-forward network before wordprediction, which BART does not. 

In total, BART contains roughly 10% more parameters than the equivalently sized BERT model.
```

##### 2.2 Pre-training BART

```
BART is trained by corrupting documents and then 
optimizing a reconstruction loss—the cross-entropy between the Decoder’s output and the original document.

Unlike existing deNoising autoEncoders, which are tailored to specific Noising schemes, 
BART allows us to apply any type of document corruption. 

In the extreme case, where all information about the source is lost,
BART is equivalent to a language model.

We experiment with several previously proposed and novel transformations, 
but we believe there is a significant potential for development of other new alternatives. 

The transformations we used are summarized below, and examples are shown in Figure 2.

Token Masking
Following BERT (Devlin et al., 2019), random Tokens are sampled and replaced with [Mask] elements.

Token Deletion 
Random Tokens are deleted from the input.
In contrast to Token Masking, the model must decide which positions are missing inputs.

Text Infilling 
A number of Text Spans are sampled, with Span lengths drawn from a Poisson distribution (λ = 3).
Each Span is replaced with a single [Mask] Token. 
0-length Spans correspond to the insertion of [Mask] Tokens. 
Text infilling is inspired by SpanBERT (Joshi et al., 2019), 
but SpanBERT samples Span lengths from a different (clamped geometric) distribution, 
and replaces each Span with a sequence of [Mask] Tokens of exactly the same length. 
Text infilling teaches the model to predict how many Tokens are missing from a Span.

Sentence Permutation 
A document is divided into sentences based on full stops, 
and these sentences are shuffled in a random order.

Document Rotation 
A Token is chosen uniformly at random, 
and the document is rotated so that it begins with that Token. 
This Task trains the model to identify the start of the document.

Figure 2: Transformations for Noising the input that we experiment with.
These transformations can be composed.
```

### 3. Fine-tuning BART

```
The representations produced by BART can be used in several ways for downstream applications.

Figure 3: Fine tuning BART for classification and translation.

(a) To use BART for classification problems, the same input is fed into the Encoder and Decoder, 
and the representation from the final output is used.

(b) For machine translation, we learn a small additional Encoder that replaces the word embeddings in BART. 
The new Encoder can use a disjoint vocabulary.
```

##### 3.1 Sequence Classification Tasks

```
For sequence classification Tasks, the same input is fed into the Encoder and Decoder, 
and the final hidden state of the final Decoder Token is fed into new multi-class linear classifier. 

This approach is related to the CLS Token in BERT;
however we add the additional Token to the end 
so that representation for the Token in the Decoder can attend to Decoder states from the complete input (Figure 3a).
```

##### 3.2 Token Classification Tasks

```
For Token classification Tasks, such as answer endpoint classification for SQuAD, 
we feed the complete document into the Encoder and Decoder, 
and use the top hidden state of the Decoder as a representation for each word. 
This representation is used to classify the Token.
```

##### 3.3 Sequence Generation Tasks

```
Because BART has an autoregressive Decoder, it can be directly fine tuned for sequence generation Tasks
such as abstractive question answering and summarization.

In both of these Tasks, information is copied from the input but manipulated, 
which is closely related to the deNoising pre-training objective. 

Here, the Encoder input is the input sequence, and the Decoder generates outputs autoregressively.
```

##### 3.4 Machine Translation

```
We also explore using BART to improve machine translation Decoders for translating into English. 

Previous work Edunov et al. (2019) has shown that models can be improved by incorporating pre-trained Encoders,
but gains from using pre-trained language models in Decoders have been limited.

We show that it is possible to use the entire BART model (both Encoder and Decoder) 
as a single pretrained Decoder for machine translation, 
by adding a new set of Encoder parameters that are learned from biText (see Figure 3b).

More precisely, we replace BART’s Encoder embedding layer with a new randomly initialized Encoder.

The model is trained end-to-end, 
which trains the new Encoder to map foreign words into an input that BART can de-Noise to English. 

The new Encoder can use a separate vocabulary from the original BART model.

We train the source Encoder in two steps,
in both cases backpropagating the cross-entropy loss from the output of the BART model. 

In the first step, we freeze most of BART parameters and only update the randomly initialized source Encoder, the BART positional embeddings, 
and the self-attention input projection matrix of BART’s Encoder first layer.

In the second step, we train all model parameters for a small number of iterations.
```

### 4. Comparing Pre-training Objectives

```
BART supports a much wider range of noising schemes during pre-training than previous work. 

We compare a range of options using base-size models (6 encoder and 6 decoder layers, with a hidden size of 768), 
evaluated on a representative subset of the tasks we will consider for the full large scale experiments in §5.
```

##### 4.1 Comparision Objectives

```
While many pre-training objectives have been proposed, 
fair comparisons between these have been difficult to perform, 
at least in part due to differences in training data, training resources, architectural differences between models, and fine-tuning procedures. 

We re-implement strong pre-training approaches recently
proposed for discriminative and generation tasks. We
aim, as much as possible, to control for differences unrelated to the pre-training objective. However, we do
make minor changes to the learning rate and usage of
layer normalisation in order to improve performance
(tuning these separately for each objective). For reference, we compare our implementations with published
numbers from BERT, which was also trained for 1M
steps on a combination of books and Wikipedia data.
We compare the following approaches:

Language Model Similarly to GPT (Radford et al.,
2018), we train a left-to-right Transformer language
model. This model is equivalent to the BART decoder,
without cross-attention.
Permuted Language Model Based on XLNet (Yang
et al., 2019), we sample 1/6 of the tokens, and generate them in a random order autoregressively. For consistency with other models, we do not implement the
relative positional embeddings or attention across segments from XLNet.
Masked Language Model Following BERT (Devlin
et al., 2019), we replace 15% of tokens with [MASK]
symbols, and train the model to independently predict
the original tokens.
Multitask Masked Language Model As in UniLM
(Dong et al., 2019), we train a Masked Language
Model with additional self-attention masks. Self attention masks are chosen randomly in with the follow
proportions: 1/6 left-to-right, 1/6 right-to-left, 1/3 unmasked, and 1/3 with the first 50% of tokens unmasked
and a left-to-right mask for the remainder.
Masked Seq-to-Seq Inspired by MASS (Song et al.,
2019), we mask a span containing 50% of tokens,
and train a sequence to sequence model to predict the
masked tokens.

For the Permuted LM, Masked LM and Multitask
Masked LM, we use two-stream attention (Yang et al.,
2019) to efficiently compute likelihoods of the output
part of the sequence (using a diagonal self-attention
mask on the output to predict words left-to-right).

We experiment with (1) treating the task as a standard sequence-to-sequence problem, where the source
input to the encoder and the target is the decoder output, or (2) adding the source as prefix to the target in
the decoder, with a loss only on the target part of the
sequence. We find the former works better for BART
models, and the latter for other models.
To most directly compare our models on their ability
to model their fine-tuning objective (the log likelihood
of the human text), we report perplexity in Table 1.
```