# LangChain Setup

>  ***Python*** 기반으로 작성함

- `langchain` 프레임워크를 설치하면 `langchain-core`, `langchain-community`, `langsmith` 등 프로젝트 수행에 필수적인 라이브러리들이 함께 설치됨

    ```bash
    pip install langchain

    ```

- 단, 다양한 외부 모델 제공자와 데이터 저장소 등과 통합을 위해서는 의존성 설치가 따로 필요함
    - 만약 OpenAI에서 제공하는 LLM을 사용하려면 `langchain-openai`, `tiktoken` 라이브러리를 설치해야 함

    ```bash
    pip install langchain-openai tiktoken
    ```

    - `langchain-openai` : GPT-3.5, GPT-4 등 LLM 모델과 기타 보조 도구

    - `tiktoken` : OpenAI 모델이 사용하는 Tokenizer 