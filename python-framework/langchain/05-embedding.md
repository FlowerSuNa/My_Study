


## 문서 임베딩

### 방법 1 : OpenAI 

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np

# OpenAIEmbeddings 모델 생성
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 사용할 모델 이름
    dimensions=1024, # 원하는 임베딩 차원 수를 지정 가능 (기본값: None)
)

# 문서 컬렉션
documents = [
    "인공지능은 컴퓨터 과학의 한 분야입니다.",
    "머신러닝은 인공지능의 하위 분야입니다.",
    "딥러닝은 머신러닝의 한 종류입니다.",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.",
    "컴퓨터 비전은 컴퓨터가 디지털 이미지나 비디오를 이해하는 방법을 연구합니다."
]

# 문서 임베딩
document_embeddings_openai = embeddings_openai.embed_documents(documents)

# 임베딩 결과 출력
print(f"임베딩 벡터의 개수: {len(document_embeddings_openai)}")
print(f"임베딩 벡터의 차원: {len(document_embeddings_openai[0])}")
print(document_embeddings_openai[0])

embedded_query_openai = embeddings_openai.embed_query("인공지능이란 무엇인가요?")

# 쿼리 임베딩 결과 출력
print(f"쿼리 임베딩 벡터의 차원: {len(embedded_query_openai)}")
print(embedded_query_openai)

# 쿼리와 가장 유사한 문서 찾기 함수
def find_most_similar(
        query: str, 
        doc_embeddings: np.ndarray,
        embeddings_model=OpenAIEmbeddings(model="text-embedding-3-small")
        ) -> tuple[str, float]:
    
    # 쿼리 임베딩: OpenAI 임베딩 사용 
    query_embedding = embeddings_model.embed_query(query)

    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # 가장 유사한 문서 인덱스 찾기
    most_similar_idx = np.argmax(similarities)

    # 가장 유사한 문서와 유사도 반환: 문서, 유사도
    return documents[most_similar_idx], similarities[most_similar_idx]

# 예제 쿼리
queries = [
    "인공지능이란 무엇인가요?",
    "딥러닝과 머신러닝의 관계는 어떻게 되나요?",
    "컴퓨터가 이미지를 이해하는 방법은?"
]

# 각 쿼리에 대해 가장 유사한 문서 찾기
for query in queries:
    most_similar_doc, similarity = find_most_similar(
        query, 
        document_embeddings_openai, 
        embeddings_model=embeddings_openai
        )
    print(f"쿼리: {query}")
    print(f"가장 유사한 문서: {most_similar_doc}")
    print(f"유사도: {similarity:.4f}")
```

### 방법 2 : Huggingface

```python
```