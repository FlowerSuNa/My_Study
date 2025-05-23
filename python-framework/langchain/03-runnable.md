# LangChain의 실행 구조 : Runnable과 LCEL

## Runnable

- LangChain의 모든 컴포넌트는 `Runnable` 인터페이스를 구현하여 일관된 방식으로 실행됨  
- 실행 메서드로는 `.invoke()` (단일 입력), `.batch()` (여러 입력), `.stream()` (스트리밍 처리) 등을 지원하며, 동기/비동기 처리 방식에 따라 다양하게 활용 가능  
- 모든 `Runnable` 컴포넌트는 `|` 연산자를 사용해 연결할 수 있으며, 이를 통해 재사용성과 조합성이 높은 체인을 구성할 수 있음 (LCEL 기반)

### 주요 Runnable 클래스

- `RunnablePassthrough` : 입력값을 그대로 다음 단계로 전달 (디버깅 또는 테스트용)  
- `RunnableParallel` : 여러 Runnable을 병렬로 실행하여 결과를 동시에 처리  
- `RunnableLambda` : 사용자 정의 파이썬 함수를 Runnable로 래핑하여 체인에 포함  

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


💡 **Tip**: 하나의 Runnable, 프롬프트, 함수는 되도록 하나의 명확한 기능만 수행하도록 구성하는 것이 좋음
- 복잡한 로직을 여러 단계로 나누어 구성하면 가독성과 유지보수성이 크게 향상됨.