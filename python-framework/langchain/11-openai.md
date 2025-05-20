# OpenAI 사용법

- OpenAI에서 제공하는 LLM을 사용하려면 `langchain-openai` 라이브러리를 설치해야 함

    ```bash
    pip install langchain-openai tiktoken
    ```

- `ChatOpenAI`는 인스턴스로 생성해 LLM 응답을 처리할 수 있음
    - [Document](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
    - [Guide](https://python.langchain.com/docs/integrations/chat/openai/)

    ```python
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        # model="gpt-4.1-mini-2025-04-14", # 정확한 버전 표기
        temperature=0.4,
        top_p=0.7
    )
    ```

- 모델 버전은 학습 날짜를 명시해 정확히 선언하는 것이 좋음
    - [OpenAI Model Version](https://platform.openai.com/docs/pricing)
    - 단순히 모델명만 지정하면, OpenAI 측의 업데이트로 인해 모델 차이가 발생할 수 있음
    - 동일한 모델명이라도 버전에 따라 응답 성향이나 성능이 달라질 수 있음

