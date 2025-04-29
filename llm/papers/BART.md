# BART

- 2019년 10월 Facebook에서 발표한 논문임

### 0. Abstract

- BART는 seq2seq 구조를 기반으로 한 [Denosing AutoEncoder](#denosing-autoencoder)임
    - text에 임의의 noise를 주어 변형시킨 후, 원본 text로 복원하며 모델을 학습시킴
    - 표준 Transformer 기반 MNT(Neural Machine Translation) 구조를 사용함
    - 단순한 구조임에도 BERT(bidirectional encoder)와 GPT(left-to-right decoder)를 포함한 다양한 사전 학습 방식을 일반화함

- 원본 문장들의 순서를 무작위로 섞고, text의 [spans](#spans)가 하나의 mask token으로 치환되는 [Text Infilling](#text-infilling) 방식을 사용할 때 가장 우수한 성능을 보였음

- BART는 text 생성 작업을 위한 fine-tuning에서 특히 효과적이었지만, 이해력이 요구되는 작업에서도 잘 작동함
    - [GLUE](#glue) 및 [SQuAD](#squad) benchmark에서는 [RoBERTa](#roberta)와 유사한 학습 자원 하에서 비슷한 성능을 기록함
    - 추상적 대화, 질의응답, 요약 작업에서는 최고 성능(최대 6 ROUNGE 포이트 향상)을 달성함
    - 또한, 기계 번역을 위한 [Back-translation](#back-translation) task에서도 target language에 대한 사전학습만으로 1.1 [BLEU](#bleu) (Bilingual Evaluation Understudy) 향상함

---

### 1. Introduction

- Self-supervised 기법들은 NLP 분야에서 뛰어난 성과를 보여주었음
    - 가장 성공적인 접근법은 랜덤으로 masked text를 복원하도록 학습하는 denosing autoencoder, 즉 masked language model의 변형들임

- 최근 연구들은 masked token의 분포, 예측 순서, 대체 가능한 context를 개선하는 성과를 달성함
    - 하지만 이 기법들은 특정 end task(span 예측, 생성 등)에만 집중하여 적용 가능성이 제한적임

- 본 논문에서는 bidirectional과 auto-regressive transformer를 결합한 사전 학습 모델인 BART를 제안함

- BART는 다양한 end task에 적용 가능한 seq2seq 기반 denosing autoencoder임
    - 사전 학습은 text를 임의의 nosing function으로 변형한 후, seq2seq 모델이 원본 text로 복구하도록 학습하는 방식임

- BART는 표준 transformer 기반 NMT 구조를 사용함
    - 단순한 구조임에도 bidirectional encoder인 BERT와 left-to-right decoder인 GPT를 포함한한 다양한 최신 사전 학습 방식들을 일반화함 (Figure 1)

![figure 1](img/bart-fig-01.png)

- Figure 1 : BERT, GPT, BART 방식 비교
    - BERT는 임의 token를 masking 후 bidirectionally 문서를 encoding하고, masked token를 독립적으로 예측하기 때문에 생성 작업에는 적합하지 않음
    - GPT는 token을 auto-regressively 예측하여 생성 작업에는 적합하지만, 좌측 context만 활용할 수 있어 bidirectional 상호작용을 학습할 수 없음
    - BART는 encoder 입력과 decoder 출력이 일치할 필요가 없어 임의의 noise 변환을 적용할 수 있음
        - 이때 문서는 일부 spans가 masking 기호로 대체되어 변형된 상태임
        - 변형된 문서(왼쪽)는 bidirectional 모델로 encoding되고, 이후 원본 문서(오른쪽)로 autoregressive decording하여 복원함
        - fine-tuning 시에는 손상되지 않은 문서를 encoder와 decoder 모두에 입력하며, decoder의 마지막 hidden state에서 표현을 추출해 사용함

- 이 방식의 주요 장점은 noising의 유연성으로, 원본 text 길이 변경을 포함한 다양한 변형이 가능함

- 다양한 noising 접근법을 평가한 결과, 원본 문장을 랜덤하게 섞은 후, 길이 0을 포함한 임의 길이의 spans를 하나의 mask token으로 대체하는 새로운 text filling 방식을 사용하는 것이 가장 우수한 성능 보임
    - 이 접근법은 모델이 전체 문장 길이에 대해 더 깊이 추론하고, 입력보다 더 긴 변환을 하도록 강제함으로써, BERT의 기존 단어 masking과 NSP(Next Sentence Prediction) 목표를 일반화함

- BART는 text 생성 작업을 위해 fine-tuning 되었을 때 특히 효과적이었으며, 이해력이 요구되는 작업에서도 잘 작동함
    - GLUE 및 SQuAD와 같은 benchmark에서는 RoBERTa와 유사한 학습 자원 하에서 비슷한 성능을 기록하고, 추상적 대화, 질의응답, 요약 작업에서 최고 성능을 달성함
    - 예를 들어, XSum benchmark에서는 이전 연구 대비 ROUNGE 점수 6점이 향상됨

- 또한, BART는 fine-tuning에 대한 새로운 접근 방식을 제안함
    - BART 모델 위에 몇 개의 추가 transformer layer를 쌓는 새로운 기계 번역 방식을 제안함
    - 이 layer들은 BART의 propagation을 통해 외국어를 noisy 영어로 번역하도록 학습되며, 이를 통해 BART를 사전 학습된 target 쪽 언어 모델로 활용함

        > - BART는 사전 학습할 때 영어 문장을 손상시키고, 그 손상된 영어를 다시 원래 영어로 복원하는 방식으로 학습된 모델임
        > - 즉, noisy 영어를 깨끗한 영어로 복구를 잘하는 모델임
        > - 따라서 추가된 transformer layer로 외국어를 noisy 영어로 변환해주면, BART는 noisy 영어를 fluent 영어로 자연스럽게 복원함 

    - 이 방식은 WMT Romanian-English benchmark에서 기존 back-translation 기반 기계 번역 기준 대비 1.1 BLEU 만큼 성능을 향상시킴

- BART는 다양한 task에 걸쳐 일관되게 강력한 성능을 보임 

### 2. Model

- BART는 손상된 문서를 원래 문서로 복원하는 denoising autoencoder임
    - 손상된 text를 입력으로 받아, bidirectional encoder와 left-to-right autoregressive decoder를 갖춘 seq2seq 모델로 구현됨
    - 사전 학습은 원래 문서에 대한 negative log likelihood를 최소화하는 방식으로 진행됨

##### 2.1 Architecture

- BART는 표준 seq2seq Transformer 구조를 사용함
    - 단, GPT를 따라 activation 함수를 ReLU 대신 [GeLU](#gelu)로 변경하고, parameter를 N(0, 0.02) 분포로 초기화함

- 기본 모델은 6개의 encoder와 decoder layer를 사용하고, 대형 모델은 12개의 layer를 사용함

- 이 구조는 BERT와 밀접한 관련이 있지만 다음과 같은 차이점이 있음
    - decoder의 각 layer는 encoder의 마지막 hidden layer에 대해 추가적으로 cross-attention을 수행함
    - BERT는 단어 예측 전에 feed-forword network를 추가로 사용하지만, BART는 사용하지 않음

- 전체적으로 BART는 동일 크기의 BERT 모델보다 약 10% 더 많은 parameter를 가짐

##### 2.2 Pre-training BART

- BART는 문서를 손상시킨 후, decoder 출력과 원본 문서 간의 cross-entropy loss를 최소화하는 방식으로 학습됨

- 특정 noising 방식에만 최적화된 denoising autoencoder와 달리, BART는 다양한 유형의 손상된 문서를 학습에 활용할 수 있음
    - 원본의 모든 정보를 잃은 경우, BART는 일반적인 언어 모델과 동일하게 동작함

- 잠재력이 있는 새로운 noising 변형 기법을 실험함
    - Token Masking : BERT와 같이, 무작위로 token을 선택하여 `MASK`로 교체함
    - Token Delection : 입력에서 무작위로 token을 제거하고, 어떤 위치에서 입력이 누락되었는지 판단함
    - Test Infilling : Poisson 분포 (λ = 3)를 기반으로 span 길이를 sampling하여 무작위로 text span을 선택하고, 각 span을 하나의 `MASK` token으로 교체함
        - 길이가 0인 span도 `MASK` token으로 대응될 수 있음
        - SpanBERT에서는 geometric 분포를 통해 sampling하고, 각 span을 동일 길이의 `MASK` token으로 교체하는 방식을 제안함
        - Text Infilling은 모델이 하나의 span으로 누락된 token 수를 예측하도록 학습함

    - Sentence Permutation : 문서는 full stop(마침표)을 기준으로 문장이 나뉘고, 그 문장들을 무작위로 섞음
    - Document Rotation : 문서 내 무작위로 하나의 token이 선택하여, 해당 token으로 문서가 시작되도록 회전시키고, 모델이 문서의 시작을 식별하도록 학습시킴

![figure 2](img/bart-fig-02.png)

- Figure 2 : 입력 text에 대한 여러 형태의 noisy 변환은 조합하여 적용할 수 있음



---

# Dictionaly

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

### Spans

- 연속된 단어나 문장의 조각을 의미함
- 단어 하나가 아니라 여러 단어가 묶여 있는 덩어리를 가리킴
- 단어 단위가 아니라 구(phrase)나 문장 조각 단위로 다뤄질 수 있음

### Text Infilling

- 문장의 일부를 연속적으로 비우고, 그 빈칸을 채우는 작업으로 학습시키는 방식임
- 즉, 단어 하나를 마스킹하는 BERT의 방식이 아니라, 문장 중간을 통째로 비워서 모델이 자연스럽게 복원하도록 학습시키는 방식임

### RoBERTa

- RoBERTa는 페이스북에서 개발한 BERT 개선 버전 모델임
- BERT랑 구조는 같지만 학습 데이터를 훨씬 많이 사용하고, 마스킹 방식도 더 정교하게 조정함
- NSP(Next Sentence Prediction)을 제거하여 성능을 높였음 <br>(NSP task가 실제로 언어 이해 능력 향상에는 큰 도움이 안됨을 실험으로 확인함)

### GLUE

- GLUE는 다양한 NLP 과제를 모아놓은 benchmark 테스트임
- 문장 추론, 유사도, 감정 분류 등 총 9개의 task로 구성되어 있음
- 모델이 얼마나 언어를 잘 이해하는지를 평가하는 데 쓰임
- 많은 모델들이 이 점수를 기준으로 성능을 비교함

### SQuAD

- SQuAD는 QA task를 위한 데이터셋임
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

---

# Reference

- [논문](https://arxiv.org/pdf/1910.13461)
- [논문 요약](https://velog.io/@tobigs-nlp/BART-Denoising-Sequence-to-Sequence-Pre-training-for-Natural-Language-Generation-Translation-and-Comprehension)
- [논문 요약2](https://velog.io/@dutch-tulip/BART)
- [GeLU](https://jik9210.tistory.com/14)

---

# Original Text

### 0. Abstract

```
We present BART, a denoising autoencoder for pretraining sequence-to-sequence models.

BART is trained by (1) corrupting text with an arbitrary noising function, 
and (2) learning a model to reconstruct the original text.

It uses a standard Tranformer-based neural machine translation architecture which, 
despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), 
GPT (with the left-to-right decoder), and many other more recent pretraining schemes.

We evaluate a number of noising approaches, 
finding the best performance 
by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, 
where spans of text are replaced with a single mask token.

BART is particularly effective when fine tuned for text generation
but also works well for comprehension tasks.

It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, 
achieves new stateof-the-art results on a range of 
abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, 
with only target language pretraining.

We also report ablation experiments that replicate other pretraining schemes within the BART framework, 
to better measure which factors most influence end-task performance.
```

### 1. Introduction

```
Self-supervised methods have achieved remarkable success in a wide range of NLP tasks
(Mikolov et al., 2013; Peters et al., 2018; Devlin et al., 2019; Joshi et al., 2019; Yang et al., 2019; Liu et al., 2019).

The most successful approaches have been variants of masked language models, 
which are denoising autoencoders that are trained to reconstruct text 
where a random subset of the words has been masked out.

Recent work has shown gains by improving the distribution of masked tokens (Joshi et al., 2019), 
the order in which masked tokens are predicted (Yang et al., 2019), 
and the available context for replacing masked tokens (Dong et al., 2019).

However, these methods typically focus on particular types of end tasks (e.g. span prediction, generation, etc.), 
limiting their applicability.

In this paper, we present BART, 
which pre-trains a model combining Bidirectional and Auto-Regressive Transformers.

BART is a denoising autoencoder built with a sequence-to-sequence model that is applicable
to a very wide range of end tasks.

Pretraining has two stages (1) text is corrupted with an arbitrary noising function,
and (2) a sequence-to-sequence model is learned to reconstruct the original text. 

BART uses a standard Tranformer-based neural machine translation architecture which, 
despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder),
GPT (with the left-to-right decoder), and many other more recent pretraining schemes (see Figure 1).

Figure 1: A schematic comparison of BART with BERT (Devlin et al., 2019) and GPT (Radford et al., 2018).

(a) BERT: Random tokens are replaced with masks, and the document is encoded bidirectionally.
Missing tokens are predicted independently, so BERT cannot easily be used for generation.

(b) GPT: Tokens are predicted auto-regressively, meaning GPT can be used for generation.
However words can only condition on leftward context, so it cannot learn bidirectional interactions.

(c) BART: Inputs to the encoder need not be aligned with decoder outputs, allowing arbitary noise transformations.
Here, a document has been corrupted by replacing spans of text with mask symbols.
The corrupted document (left) is encoded with a bidirectional model,
and then the likelihood of the original document (right) is calculated with an autoregressive decoder.
For fine-tuning, an uncorrupted document is input to both the encoder and decoder,
and we use representations from the final hidden state of the decoder.

A key advantage of this setup is the noising flexibility;
arbitrary transformations can be applied to the original text, including changing its length.

We evaluate a number of noising approaches, finding the best performance 
by both randomly shuffling the order of the original sentences and using a novel in-filling scheme,
where arbitrary length spans of text (including zero length) are replaced with a single mask token.

This approach generalizes the original word masking and next sentence prediction objectives in BERT 
by forcing the model to reason more about overall sentence length and make longer range transformations to the input.

BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. 

It matches the performance of RoBERTa (Liu et al., 2019) 
with comparable training resources on GLUE (Wang et al., 2018) and SQuAD (Rajpurkar et al., 2016),
and achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks. 

For example, it improves performance by 6 ROUGE over previous work on XSum (Narayan et al., 2018).

BART also opens up new ways of thinking about fine tuning.

We present a new scheme for machine translation
where a BART model is stacked above a few additional transformer layers.

These layers are trained to essentially translate the foreign language to noised English,
by propagation through BART, thereby using BART as a pre-trained target-side language model.

This approach improves performance over a strong back-translation MT baseline
by 1.1 BLEU on the WMT Romanian-English benchmark.

To better understand these effects,
we also report an ablation analysis that replicates other recently proposed training objectives. 

This study allows us to carefully control for a number of factors, 
including data and optimization parameters, which have been shown
to be as important for overall performance as the selection of training objectives (Liu et al., 2019). 

We find that BART exhibits the most consistently strong performance across the full range of tasks we consider.
```

### 2. Model

```
BART is a denoising autoencoder that maps a corrupted document to the original document it was derived from.

It is implemented as a sequence-to-sequence model with a bidirectional encoder 
over corrupted text and a left-to-right autoregressive decoder.

For pre-training, we optimize the negative log likelihood of the original document.
```

##### 2.1 Architecture

```
BART uses the standard sequence-to-sequence Transformer architecture from (Vaswani et al., 2017), 
except, following GPT, that we modify ReLU activation functions to GeLUs (Hendrycks & Gimpel, 2016)
and initialise parameters from N (0, 0.02).

For our base model, we use 6 layers in the encoder and decoder, and for our large model we use 12 layers in each. 

The architecture is closely related to that used in BERT, with the following differences: 
(1) each layer of the decoder additionally performs cross-attention over the final hidden layer of the encoder (as in the transformer sequence-to-sequence model); 
and (2) BERT uses an additional feed-forward network before wordprediction, which BART does not. 

In total, BART contains roughly 10% more parameters than the equivalently sized BERT model.
```

##### 2.2 Pre-training BART

```
BART is trained by corrupting documents and then 
optimizing a reconstruction loss—the cross-entropy between the decoder’s output and the original document.

Unlike existing denoising autoencoders, which are tailored to specific noising schemes, 
BART allows us to apply any type of document corruption. 

In the extreme case, where all information about the source is lost,
BART is equivalent to a language model.

We experiment with several previously proposed and novel transformations, 
but we believe there is a significant potential for development of other new alternatives. 

The transformations we used are summarized below, and examples are shown in Figure 2.

Token Masking
Following BERT (Devlin et al., 2019), random tokens are sampled and replaced with [MASK] elements.

Token Deletion 
Random tokens are deleted from the input.
In contrast to token masking, the model must decide which positions are missing inputs.

Text Infilling 
A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3).
Each span is replaced with a single [MASK] token. 
0-length spans correspond to the insertion of [MASK] tokens. 
Text infilling is inspired by SpanBERT (Joshi et al., 2019), 
but SpanBERT samples span lengths from a different (clamped geometric) distribution, 
and replaces each span with a sequence of [MASK] tokens of exactly the same length. 
Text infilling teaches the model to predict how many tokens are missing from a span.

Sentence Permutation 
A document is divided into sentences based on full stops, 
and these sentences are shuffled in a random order.

Document Rotation 
A token is chosen uniformly at random, 
and the document is rotated so that it begins with that token. 
This task trains the model to identify the start of the document.

Figure 2: Transformations for noising the input that we experiment with.
These transformations can be composed.
```