# Few-shot Prompt

- Few-shop은 LLM에 몇 가지 예시를 제공하여 특정 태스크를 수행하도록 유도하는 기법임

- LLM이 주어진 예제들을 참고하여 더 정확하고 일관된 응답을 생성할 수 있음
    - 특정 도메인이나 형식의 질문에 대해 답변 성능을 향상시키는 데 효과적임

### 라이브러리 설치

```
pip install langchain langchain-openai langchain-community chromadb
```

### 예제 선택기 사용

```python
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings



## Few-shop 생성
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
examples = [
    {
        "question": "question1",
        "answer": "answer1"
    },
    {
        "question": "question2",
        "answer": "answer2"
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,              # 사용할 예제들
    example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
    suffix="Question: {input}",     # 예제 뒤에 추ㅇㄷㄷ가될 접미사
    input_variables=["input"],      # 입력 변수 지정
)

## 예제 선택

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,            # 사용할 예제들
    OpenAIEmbeddings(),  # 임베딩 모델
    Chroma,              # 벡터 저장소
    k=1,                 # 선택할 예제 수
)

question = "your_question"
selected_examples = example_selector.select_examples({"question": question})

## 유사한 예제 출력
print(f"Question : {question}\nSimilar Example : ")
for example in selected_examples:
    for k, v in example.items():
        print(f"- {k} : {v}")

```
