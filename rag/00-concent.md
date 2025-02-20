# RAG (Retrieval-Augmented Generation)


## 등장 배경

- LLM이 매일 학습하지는 않음
    - ChatGPT4는 2023년 2월까지의 데이터로 학습됨
    - 학습 이후에 일어난 사건이나 지식에 대해 알지 못함

- 이런 LLM이 보다 정확하고 최신 상태로 유지하는데 도움을 주는 도구가 RAG임

## 개념

- RAG는 LLM이 알고 있는 정보에만 의존하는 것이 아니라 ***지식 콘텐츠 저장소***를 통해 추가 정보와 정보의 출처를 알려줌줌
    - 즉, LLM이 답변을 생성하기 전에 ***지식 콘텐츠 저장소***에서 사용자 질의와 관련된 정보를 검색하도록 지시함

- RAG 프레임워크 플로우

    ```
        ① Prompt + Query   ② Query
    User --------> Service --------> Knowledge Source
                           <--------
                            ③ Relevant Information for Enhanced Context

                            ④ Prompt + Query + Enhanced Context
                   Service --------> LLM
    User <---------------- <--------
                            ⑤ Generated Text Response
    ```

- RAG를 사용하면 LLM을 다시 훈련할 필요없이 ***지식 콘텐츠 저장소***에 새로운 정보, 최신 정보를 지속적으로 업데이트하면 됨
    - 따라서, RAG는 다양한 상황에서 관련성, 정확성, 유용성 유지를 위한 비용 효율적인 접근 방식임

- 또한, 사용자의 질의에 대해 신뢰할 수 없는 답변(ex. Halluciation)이 발생되되지 않도록 할 수 있음

- 양질의 지식 컨텐츠 저장소 구축 뿐만 아니라, LLM의 고품질 정보를 제공하기 위해 Retriever을 개발해야 함

- 답변을 생성할 때 최종적으로 사용자에게 최상의 답변을 제공할 수 있도록 Generation 개선을 위한 노력도 필요함

## AWS 서비스

- Amazon Bedrock : 정보 보호를 유지 관리하고 개발을 간소화하여 생산형 AI 애플리케이션을 구축할 수 있음

- Amazon Kendra : 자체 RAG를 관리하는 조직을 위해 기계 학습 기반의 매우 정확한 엔터프라이즈 검색 서비스임
    - Kendra Retrive API를 제공함


## Reference

- [링크1](https://brunch.co.kr/@ywkim36/146)
- [링크2](https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/)