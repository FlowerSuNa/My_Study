# OpenAI 사용법

- `openai` 패키지를 사용하여 API를 호출할 수 있음

    ```python
    from openai import OpenAI

    client = OpenAI(
        api_key = OPENAI_API_KEY # 환경 변수로 등록되어 있는 경우 입력하지 않아도 됨
    )

    # Completion 요청 (prompt -> completion)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            # developer 역할 - 전반적인 동작 방식 정의
            {"role": "developer", "content": "You are a assistant."},
            # user 역할 - 실제 요청 내용
            {"role": "user", "content": "..."},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    ```

- OpenAI에서 제공하는 LLM을 LangChain과 사용하려면 `langchain-openai` 라이브러리를 설치하여 사용할 수 있음

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

- 모델 선택 시 고려사항
    - gpt-4.1 : 높은 성능 및 비용
    - gpt-4.1-mini : 빠른 속도, 낮은 비용
    - o1 계열 : 복잡한 추론 가능

