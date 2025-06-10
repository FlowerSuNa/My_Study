# Pydantic

**[Github](https://github.com/pydantic/pydantic)**

**[Document](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel)**

- Python의 타입 힌트(`str`, `int`, `List` 등)를 기반으로 데이터 유효성 검사 및 파싱을 자동으로 처리해줌
- **FastAPI, LangChain** 등 최신 Python 프레임워크에서 데이터 모델 정의 시 널리 사용됨

## 주요 특징

- 직관적인 데이터 모델링 가능함
    - 클래스 정의만으로 JSON-like 구조를 표현하고 검증할 수 있음
- 자동 타입 캐스팅 지원함
    - "123"과 같은 문자열도 `int` 필드에 자동 변환됨
- `dict` ↔ `object` 간 변환 쉬움
    - `.dict()`, `.model_dump()` 등을 통해 손쉽게 변환 가능함
- 입력 데이터 유효성 검사 자동 수행함
    - 필드 누락, 타입 불일치, 범위 초과 등 오류를 사전에 방지할 수 있음
- 오류 발생 시 상세한 메시지 제공함
    - 어떤 필드에서 어떤 문제가 발생했는지 명확하게 출력함
- pydantic은 내부적으로 Cython으로 최적화되어 매우 빠름
