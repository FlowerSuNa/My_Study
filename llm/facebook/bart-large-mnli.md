
# 1. Concept

- `facebook/bart-large-mnli`은 Text Classification 모델로, BART 모델을 기반으로 MNLI 작업에 fine-tuning 된 모델임

    - 또한, NLI 기반 Zero Shot Text classification 모델임

- BART는 [Denosing](#denosing) 기법을 seq2seq에 적용시켜 자연어 생성, 번역, 이해를 하는 모델임


### 1) BART

- Bidirectional and Auto-Regressive Transformers

- 페이스북이 만든 AI 모델로, BERT의 이해 능력과 GPT의 생성 능력을 모두 결합한 하이브리드 모델임

    - seq2seq의 Encoder는 BERT(with bidirectional encoder)의 AE 특성을 가지고 있고, Decoder는 GPT(with left-to-right decoder)의 AR 특성을 가지고 있음
    - 구조적으로 Encoder-Decoder 형태임

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

### 3) Bart-large-mnli

-  hypothesis를 라벨 후보들로 직접 만들어서 입력

- 각각의 hypothesis마다 entailment 점수만 사용
- 즉, softmax 결과 중 entailment 확률만 뽑아서 비교



---

# 2. Code Example

### 1) Model Load

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
```

### 2) Single-Label Classification

```python
text = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(text, candidate_labels)
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}
```

### 3) Multi-Label Classification

```python
text = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
classifier(text, candidate_labels, multi_label=True)
#{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
# 'scores': [0.9945111274719238,
#  0.9383890628814697,
#  0.0057061901316046715,
#  0.0018193122232332826],
# 'sequence': 'one day I will see the world'}
```

### 4) When using `PyTorch`

```python
# pose sequence as a NLI premise and label as a hypothesis
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'facebook/bart-large-mnli'
nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

premise = text
hypothesis = f'This example is {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(
    premise, hypothesis, 
    return_tensors='pt',
    truncation_strategy='only_first'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]
```

---

# Dictionaly

### Denosing

- Denoising은 데이터에서 불필요한 노이즈를 제거하는 전처리 과정임
- 이미지에서는 배경이 같아도 픽셀 단위의 미세한 RGB 차이가 노이즈로 작용할 수 있음
- 이러한 노이즈를 제거하면 이미지의 일관성이 높아지고, 분석이나 모델 성능이 향상됨
- 텍스트에서도 의미 없는 기호, 중복 표현, 오탈자 등이 노이즈로 간주됨
- 불필요한 요소를 제거함으로써 핵심 정보 전달력이 높아지고, 모델이 더 정확하게 작동함

---

# Reference

- [Hugging Face](https://huggingface.co/facebook/bart-large-mnli)
- [논문](https://arxiv.org/pdf/1910.13461)
- [논문 요약](https://velog.io/@tobigs-nlp/BART-Denoising-Sequence-to-Sequence-Pre-training-for-Natural-Language-Generation-Translation-and-Comprehension)