# Few-shot Prompt

- Few-shop은 LLM에 몇 가지 예시를 제공하여 특정 태스크를 수행하도록 유도하는 기법임

- LLM이 주어진 예제들을 참고하여 더 정확하고 일관된 응답을 생성할 수 있음
    - 특정 도메인이나 형식의 질문에 대해 답변 성능을 향상시키는 데 효과적임

## 예시 코드

### 라이브러리 설치

```
pip install langchain langchain-openai langchain-chroma
```

### OPENAI API Key 설정

```python
import os

os.environ['OPENAI_API_KEY'] = 'openai_api_key'

```

### 예제 선택기 사용

```python
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

# 벡터 저장소 생성
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# 예제 선택기 생성
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are ..."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 모델과 체인 생성
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
chain = final_prompt | model

# 모델에 질문하기
result = chain.invoke({"input": "your question"})
print(result.content)
```

## Reference

- [링크1](https://wikidocs.net/231153)