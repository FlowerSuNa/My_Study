# OpenAI 사용법

- OpenAI에서 제공하는 LLM을 사용하려면 `langchain-openai` 라이브러리를 설치해야 함
    ```bash
    pip install langchain-openai tiktoken
    ```

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4.1-mini",
    # model="gpt-4.1-mini-2025-04-14", # 정확한 버전 표기
    temperature=0.4,
    top_p=0.7
)
```

- 학습날짜를 포함하여 모델 버전을 정확하게 선언해 사용하는 것이 좋음
- 모델 버전만 선언하여 사용할 경우 업데이트되어 모델의 특성이 조금 변경될 수 있음
