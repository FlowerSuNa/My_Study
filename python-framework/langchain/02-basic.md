# LangChain 기본 기능 사용

- [API 문서](https://python.langchain.com/api_reference/)

**1. Setup** : `.env` 파일에 API 키 등을 입력하고, 환경 변수로 불러와 사용함

```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()
```

**2. Model** : `langchain_openai` 라이브러리의 `ChatOpenAI` 클래스를 통해 OpenAI 모델을 호출하여 사용할 수 있음

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini")

response = model.invoke("Glory를 한국어로 번역해주세요.")
print("답변: ", response.content)
print("메타데이터: ", response.response_metadata)
```

**3. Message** : 메시지 타입을 나누어 사용함으로써 역할을 명확히 구분하고, 대화 맥락을 구조화하며, 프롬프트를 유연하게 설계할 수 있음

- `SystemMessage`: 대화의 전반적인 맥락이나 규칙을 설정하는 메시지로, LLM의 응답 스타일, 역할, 목적 등을 정의할 때 사용함 (role: system)
- `HumanMessage`: 사용자가 입력한 메시지를 나타냄 (role: user)
- `AIMessage`: AI 모델이 생성한 응답 메시지를 나타내며, 이전 응답을 기록하거나 프롬프트에 포함시킬 때 사용함 (role: assistant)

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")
human_message = HumanMessage(content="Glory")

response = model.invoke([system_msg, human_message])
print("답변: ", response.content)
```

**4. Prompt** : 프롬프트를 일관된 형식으로 작성하고 재사용 가능하게 관리할 수 있음

- `PromptTemplate` : 단일 텍스트 입력을 변수로 구성해 포맷팅할 수 있는 기본 프롬프트 템플릿
- `ChatPromptTemplate` : 여러 메시지(Human, AI, System 등)를 포함하는 대화형 프롬프트 템플릿
    - `MessagesPlaceholder` : 기존 메시지 목록을 템플릿 내 특정 위치에 삽입할 수 있도록 도와주는 클래스

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {subject}에 능숙한 비서입니다"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
```

**5. Memory** : 대화 기록을 저장 및 관리하며, 컨텍스트 유지를 위해 다양한 메모리 타입을 지원함 (대화 요약, 버퍼 저장 등 포함)

- `BaseChatMessageHistory` : 메시지 히스토리를 저장하고 불러오는 기본 클래스
- `RunnableWithMessageHistory` : 체인이나 에이전트 실행 시, 자동으로 메시지 기록을 연동해 사용하는 래퍼 클래스
- `ConversationBufferMemory` : 최근 대화 내용을 버퍼 형태로 유지하여 간단한 맥락을 지속적으로 제공하는 클래스
- `SummaryMemory` : 이전 대화 내용을 요약해 저장함으로써 긴 대화 히스토리도 압축된 형태로 관리할 수 있는 클래스



```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List

# 메모리 기반 히스토리 구현
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# 세션 저장소
store = {}

# 세션 ID로 히스토리 가져오기
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
```

**6. Chain** : 여러 구성 요소(LLM, 프롬프트, 툴 등)를 순차적으로 연결하여 복잡한 작업 흐름을 구성할 수 있음

```python

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# 체인 구성
chain = prompt | model | parser

# 히스토리 관리 추가
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 체인 실행, 히스토리 이용해서 대화 진행
response = chain_with_history.invoke(
    {"subject": "수학", "question": "1+2는 얼마인가요?"},
    config={"configurable": {"session_id": "user1"}}
)
```