# Vector DB

- 벡터화된 데이터를 효율적으로 저장하고 검색하기 위한 특수 데이터베이스 시스템
- 텍스트나 이미지 등의 비정형 데이터를 고차원 벡터 공간에 매핑하여 저장
- 유사도 기반 검색을 통해 의미적으로 가까운 데이터를 빠르게 검색 가능

**주요 기능**
- 벡터 색인화 : 효율적인 검색을 위한 데이터 구조화 수행
- 근접 이웃 검색 : 주어진 쿼리와 가장 유사한 벡터 검색
- 메타데이터 관리 : 벡터와 관련된 부가 정보를 함께 저장하고 검색

**사용 사례**
- 시맨틱 문서 검색 : 문서의 의미를 이해하여 검색
- 추천 시스템 : 유사한 아이템을 추천
- 중복 데이터 감지 : 유사한 콘텐츠 검색
- 질의응답 시스템 : 관련 문서에서 답변을 생성하는 데 필요한 근거 검색

## Chroma

- 사용자 편의성이 우수한 오픈소스 벡터 저장소
- 경량화된 임베딩 데이터베이스로 로컬 개발에 적합
- `langchain-chroma` 패키지 설치하여 사용할 수 있음

```python
# 벡터 저장소에 문서를 저장할 때 적용할 임베딩 모델
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 벡터 저장소 생성
from langchain_chroma import Chroma

chroma_db = Chroma(
    collection_name="ai_smaple_collection",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)
```

## FAISS
- Facebook AI가 개발한 고성능 유사도 검색 라이브러리

## Pinecone

- 완전 관리형 벡터 데이터베이스 서비스

## Milvus

- 분산 벡터 데이터베이스로 대규모 데이터 처리에 적합

## PostgreSQL
- pgvector 화장을 통해 벡터 저장 및 검색 기능을 제공



