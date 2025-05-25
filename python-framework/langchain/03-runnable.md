# LangChain의 실행 구조 : Runnable과 LCEL

## Runnable

- LangChain의 모든 컴포넌트는 `Runnable` 인터페이스를 구현하여 일관된 방식으로 실행됨  
- 실행 메서드로는 `.invoke()` (단일 입력), `.batch()` (여러 입력), `.stream()` (스트리밍 처리) 등을 지원하며, 동기/비동기 처리 방식에 따라 다양하게 활용 가능  
- 모든 `Runnable` 컴포넌트는 `|` 연산자를 사용해 연결할 수 있으며, 이를 통해 재사용성과 조합성이 높은 체인을 구성할 수 있음 (LCEL 기반)

💡 **Tip**: 하나의 Runnable, 프롬프트, 함수는 되도록 하나의 명확한 기능만 수행하도록 구성하는 것이 좋음
- 복잡한 로직을 여러 단계로 나누어 구성하면 가독성과 유지보수성이 크게 향상됨

## LCEL (LangChain Expression Language)

- LCEL은 LangChain에서 컴포넌트들을 `|` 연산자로 선언적으로 연결하는 방식임
- 컴포넌트는 왼쪽에서 오른쪽으로 순차적으로 실행되며, 이전 출력이 다음 입력으로 전달됨  
- 정의된 체인은 하나의 `Runnable`로 간주되어 다른 체인의 구성 요소로 재활용 가능함  
- 배치 실행 시 내부 최적화를 통해 리소스를 절약하고 처리 속도를 향상시킬 수 있음  
- LCEL은 테스트, 실험, 복잡한 흐름 제어 등 다양한 시나리오에서 구조화된 체인을 빠르게 구성할 수 있는 효율적인 표현 방식임

## RunnableSequence

- 여러 `Runnable`을 순차적으로 연결하여 실행함
- LCEL로 연결한 체인은 내부적으로 `RunnableSequence`로 컴파일됨
- 일반적으로는 LCEL 문법을 활용하여 선언적으로 구현하는 방식을 선호함

```python
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 컴포넌트 정의
prompt = PromptTemplate.from_template("'{text}'를 영어로 번역해주세요. 번역된 문장만을 출력해주세요.")
translator = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
output_parser = StrOutputParser()

# RunnableSequence 생성 - 클래스 생성 방식
translation_chain = RunnableSequence(
    first=prompt,
    middle=[translator],
    last=output_parser
)

# RunnableSequence 생성 - LCEL 방식
# translation_chain = prompt | translator | output_parser

result = translation_chain.invoke({"text": "안녕하세요"})
print(result) 
```

## RunnableParallel

- 여러 `Runnable` 객체를 딕셔너리 형태로 구성하여 병렬처리 가능함
- 동일한 입력값이 각 `Runnable`에 전달되며, 결과는 키-값 형태로 반환됨
- 주로 데이터 전처리, 변환, 포맷 조정 등에 활용되며, 다음 파이프라인 단계에서 요구하는 출력 형식으로 조정 가능함

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter

# 질문 템플릿 정의
question_template = """
다음 카테고리 중 하나로 입력을 분류하세요:
- 화학(Chemistry)
- 물리(Physics)
- 생물(Biology)

# 예시:
Q: 사람의 염색체는 모두 몇개가 있나요?
A: 생물(Biology)

Q: {question}
A: """

# 언어 분류 템플릿 정의
language_template = """
입력된 텍스트의 언어를 다음 카테고리 중 하나로 분류하세요:
- 영어
- 한국어
- 기타

# 예시:
입력: How many protons are in a carbon atom?
답변: English

입력: {question}
답변: """

# 답변 템플릿 정의
answer_template = """
당신은 {topic} 분야의 전문가입니다. {topic}에 관한 질문에 {language}로 답변해주세요.
질문: {question}
답변: """

# 프롬프트 및 체인 구성
answer_prompt = ChatPromptTemplate.from_template(answer_template)
output_parser = StrOutputParser()

# LLM model
llm = ChatOpenAI(
    model="gpt-4.1-mini", 
    temperature=0.3
)

# 병렬 처리 체인 구성
answer_chain = RunnableParallel({
    "topic": question_chain,            # 주제 분류 체인
    "language": language_chain,         # 언어 감지 체인
    "question": itemgetter("question")  # 원본 질문 추출
}) | answer_prompt | llm | output_parser

# 체인 실행 예시
result = answer_chain.invoke({
    "question": "탄소의 원자 번호는 무엇인가요?"
})
print(f"답변: {result}")
```

## RunnableLambda

- 사용자 정의 파이썬 함수를 Runnable로 래핑하여 체인에 포함

## RunnablePassthrough

- 입력값을 그대로 다음 단계로 전달함
- `RunnablePassthrough`과 함께 사용되어 입력 데이터를 새로운 키로 매핑할 수 있음
- 투명한 데이터 프름으로 파이프라인 디버깅과 구성이 용이함
