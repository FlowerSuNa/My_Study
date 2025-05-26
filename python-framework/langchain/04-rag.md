# RAG (Retrieval-Augmented Generation)

**등장 배경**
- LLM이 매일 학습하지는 않음
    - ChatGPT4는 2023년 2월까지의 데이터로 학습됨
    - 학습 이후에 일어난 사건이나 지식에 대해 알지 못함
- 이런 LLM이 보다 정확하고 최신 상태로 유지하는데 도움을 주는 도구가 RAG임

**개념**
- RAG는 LLM이 알고 있는 정보에만 의존하는 것이 아니라 ***지식 콘텐츠 저장소***를 통해 추가 정보와 정보의 출처를 알려줌
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
- 또한, 사용자의 질의에 대해 신뢰할 수 없는 답변(ex. Hallucination)이 발생되지 않도록 할 수 있음
- 양질의 지식 컨텐츠 저장소 구축 뿐만 아니라, LLM의 고품질 정보를 제공하기 위해 ***Retriever***을 개발해야 함
- 답변을 생성할 때 최종적으로 사용자에게 최상의 답변을 제공할 수 있도록 ***Generation*** 개선을 위한 노력도 필요함

**장점**
- **환각(Hallucination) 현상 감소**
    - 모델이 자체적으로 추론하기보다는 실제 문서에서 근서를 검색해 응답을 생성하므로 사실에 기반한 응답률이 크게 향상됨
- **최신 정보 반영 가능**
    - 사전 학습된 언어 모델은 최신 정보를 반영하지 못하는 한계가 있음
    - RAG는 외부 문서나 DB에서 실시간으로 정보 검색 후 활용하므로 최신 데이터 기반 응답 생성이 가능함
- **도메인 특화 응답 생성**
    - 특정 분야(법률, 의료, 산업 등)의 문서를 벡터화해 연결하면 범용 모델로는 어려운 도메인 지식 기반 응답 생성이 가능함
- **출처 추적 가능성 확보**
    - 검색된 문서를 기반으로 응답을 생성하기 때문에 사용자는 어떤 자료에서 유래된 응답인지 추적 가능함

**AWS 서비스**
- Amazon Bedrock : 정보 보호를 유지 관리하고 개발을 간소화하여 생산형 AI 애플리케이션을 구축할 수 있음
- Amazon Kendra : 자체 RAG를 관리하는 조직을 위해 기계 학습 기반의 매우 정확한 엔터프라이즈 검색 서비스임
    - Kendra Retrive API를 제공함

## Document Indexing

**1. Load Document**
- `langchain_community.document_loaders` 모듈의 다양한 로더 클래스를 활용하여 문서 데이터를 로드할 수 있음
    - `PyPDFLoader`: PDF 파일 로더 (`pypdf` 패키지 설치 필요)
    - `WebBaseLoader`: 웹 문서 로더 (`beautifulsoup4` 패키지 설치 필요)
    - `JSONLoader`: JSON 파일 로더 (`jq` 패키지 설치 필요)
    - `CSVLoader`: CSV 파일 로더

**2. Split Texts** : RAG 파인프라인에서 적절한 문서 청크 크기 조정은 검색 성능과 LLM 응답 품질에 큰 영향을 줌
- `langchain_text_splitters` 모듈의 다양한 `TextSplitter` 클래스를 활용하여 텍스를 구조화된 청크 단위로 분할할 수 있음
    - `CharacterTextSplitter`: 지정한 문자 수를 기준으로 텍스트를 일정한 길이로 분할하며, 정규표현식을 구분자로 설정하면 문단이나 문장 단위로 자연스럽게 나누는 것도 가능함
    - `RecursiveCharacterTextSplitter`: 여러 구분자를 우선순위 순으로 순차 적용해 분할하는 방식으로, 글자 수나 토큰 수 기준 모두 가능하며 토큰 단위 분할 시에는 토크나이저가 필요함
- `langchain_experimental.text_splitter` 모듈의 `SemanticChunker`를 활용하여 텍스트를 분할할 수 있음
    - 정량적인 기준이 아닌 의미 기반 분할을 수행하기 때문에 도메인 특화 RAG 성능을 높이는 데 효과적임
    - 다만 임베딩 계산이 존재하여 계산 비용이 존재하며, 처리량이 많을 경우 리소스 관리 필요함
    - **Gradient** 방식: 임베딩 벡터 간의 **기울기 변화**를 기준으로 의미가 크게 달라지는 지점을 찾아 분할함
    - **Percentile** 방식: 임베딩 거리 분포의 **백분위수**를 기준으로 급격한 의미 변화가 감지되는 구간을 분할함
    - **Standard Deviation** 방식: 임베딩 거리 분포의 **표준편차**를 활용하여 유의미한 변화 지점을 찾아 분할함
    - **Interquartile** 방식: **사분위수 범위**를 기준으로 이상치를 지점을 찾아 분할함

## Vectorstore



## Retriever

## Example 1

**1. Document Indexing**

1\) Load Document
- 웹 문서 로드

```python
from langchain_community.document_loaders import WebBaseLoader

def load_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load() 
    print(f"Document 개수: {len(docs)}")
    return docs[0]

web_urls = [
    "https://n.news.naver.com/mnews/article/029/0002927209",
    "https://n.news.naver.com/mnews/article/092/0002358620",
    "https://n.news.naver.com/mnews/article/008/0005136824",
]
docs = [load_docs(url) for url in web_urls]
```

2\) Split Texts
- 텍스트를 1000자씩 잘라서 200자씩 겹치는 Document로 변환

```python
from langchain_text_splitters import CharacterTextSplitter

# 
text_splitter = CharacterTextSplitter(
    separator="\n\n",    # 문단 구분자
    chunk_size=1000,     # 문단 길이
    chunk_overlap=200,   # 겹치는 길이
    length_function=len, # 길이 측정 함수
    is_separator_regex=False,   # separator가 정규식인지 여부
)
splitted_docs = text_splitter.split_documents(docs)

# 결과 확인
print(f"Document 개수: {len(splitted_docs)}\n\n")
for i, doc in enumerate(splitted_docs):
    print(f"Document {i} 길이: {len(doc.page_content)}")
    print(f"Document {i} 내용: {doc.page_content[:10]}...")
    print("-"*50)
```

3\) Embedding and Saving Vectors
- 문서 임베딩은 `OpenAI`의 **text-embedding-3-small** 모델을 사용함

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
)
vector_store = InMemoryVectorStore(embedding_model)
document_ids = vector_store.add_documents(splitted_docs)
```

**2. Rag Chain**
- RAG 기반 QA 체인을 구현함

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)
    
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4.1-mini"),
    chain_type="stuff",
    retriever=retriever,
)
```

**3. Output**

```python
query = "AGI관련 기사 뽑아줘."
response = qa_chain.invoke(query)
print(f"Q: {response['query']}")
print(f"A: {response['result']}")
```

Q: AGI관련 기사 뽑아줘. <br>
A: 아래는 AGI(Artificial General Intelligence, 범용 인공지능) 관련 기사 요약입니다.
<br>
**"그래서 AGI 왔다고?"…오픈AI 올트먼, X에 오묘한 메시지**  
2025년 1월 5일, 오픈AI CEO 샘 올트먼이 소셜 미디어 플랫폼 X(옛 트위터)에 올린 메시지가 AGI 도래를 암시하는 듯한 내용으로 화제가 되고 있다.  
기사 원문에 따르면 올트먼은 AGI의 정의와 관련해 수익성 중심의 인공지능 협약에 대해 언급하며, MS 등 주요 협력사들과 함께 AGI 실현과 상용화를 위한 논의를 진행 중임을 시사했다.
<br>
필요하시면 기사의 상세 내용을 더 알려드릴 수 있습니다.

## Example 2

**1. Document Indexing**

1\) Load Document
- BART 논문을 로드함

```python
import requests
from langchain_community.document_loaders import PyPDFLoader

# PDF 다운로드
url = "https://arxiv.org/pdf/1910.13461.pdf"
with open("bart_paper.pdf", "wb") as f:
    f.write(requests.get(url).content)

# 로컬에서 PDF 로드
loader = PyPDFLoader("bart_paper.pdf")
docs = loader.load()
print(f'PDF 문서 개수: {len(docs)}')
```

2\) Split Texts
- **Semantic Chunking** 방식으로 텍스트를 분할함<br>
    → 임베딩 벡터 간의 **기울기(gradient)** 변화를 기준으로 의미 단위(semantic unit)를 구분함<br>
    → 청크 길이에 일관성이 없으며, 문맥에 따라 길이가 유동적으로 결정됨

```python
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="gradient",  # 임계값 타입 설정 (gradient, percentile, standard_deviation, interquartile)
)
chunks = text_splitter.split_documents(docs)
print(f"생성된 청크 수: {len(chunks)}")
print(f"각 청크의 길이: {list(len(chunk.page_content) for chunk in chunks)}")
```

- 길이가 100자 미만인 청크는 이미지 기반 텍스트(OCR 등)로 간주하여 제거함<br>
    → 주요 텍스트가 아닌 부가 정보일 가능성이 높기 때문임

```python
selected_chunks = []
for idx, chunk in enumerate(chunks):
    content = chunk.page_content
    if len(chunk.page_content) < 100:
        print(f'{idx}: {content}')
    else:
        selected_chunks.append(chunk)

print(f"생성된 청크 수: {len(selected_chunks)}")
print(f"각 청크의 길이: {list(len(chunk.page_content) for chunk in selected_chunks)}")
```

- 1차 분할된 청크는 길이 편차가 크므로, 문자열 길이 기준으로 재귀적으로 분할하여 최종적으로는 일관된 길이의 청크를 구성함

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                      
    chunk_overlap=100,
    separators=[" \n", ".\n", ". "],
)
final_chunks = text_splitter.split_documents(selected_chunks)
print(f"생성된 텍스트 청크 수: {len(final_chunks)}")
print(f"각 청크의 길이: {list(len(chunk.page_content) for chunk in final_chunks)}")
```

3\) Embedding
- 문서 임베딩은 `OpenAI`의 **text-embedding-3-small** 모델을 사용함

```python
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)
documents = [chunk.page_content for chunk in final_chunks]
document_embeddings = embeddings_model.embed_documents(documents)
print(f"임베딩 벡터의 개수: {len(document_embeddings)}")
print(f"임베딩 벡터의 차원: {len(document_embeddings[0])}")
```

- **코사인 유사도(Cosine Similarity)**를 기반으로 입력 질의와 의미적으로 가장 가까운 문서 청크를 정확히 탐색하는지 검증함

```python
from langchain_community.utils.math import cosine_similarity
import numpy as np

def find_most_similar(
        query: str, 
        documents: list,
        doc_embeddings: np.ndarray,
        embeddings_model
    ) -> tuple[str, float]:
    """ 쿼리와 가장 유사한 문서를 반환하는 함수 (코사인 유사도 사용) """
    query_embedding = embeddings_model.embed_query(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    return documents[most_similar_idx], similarities[most_similar_idx]

# 유사도 확인
query = "What is BART architecture?"
most_similar_doc, similarity = find_most_similar(
    query, 
    documents,
    document_embeddings, 
    embeddings_model=embeddings_model
)
print(f"쿼리: {query}")
print(f"가장 유사한 문서: {most_similar_doc}")
print(f"유사도: {similarity:.4f}")
```

4\) Save Vectors
- 임베딩된 벡터는 벡터스토어로 `ChromaDB` 사용하여 저장함

```python
from langchain_chroma import Chroma

chroma_db = Chroma(
    collection_name="my_task02",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)
chroma_db.get()

# 문서를 벡터 저장소에 저장
doc_ids = [f"DOC_{i}" for i in range(len(final_chunks))]
added_doc_ids = chroma_db.add_documents(documents=final_chunks, ids=doc_ids)
print(f"{len(added_doc_ids)}개의 문서가 성공적으로 벡터 저장소에 추가되었습니다.")
```

5\) Retriever
- **MMR** 기반의 Retriever를 사용하여 문맥 다양성을 고려한 상위 3개 문서 청크를 검색함

```python
chroma_mmr = chroma_db.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 3,                 # 검색할 문서의 수
        'fetch_k': 8,           # mmr 알고리즘에 전달할 문서의 수 (fetch_k > k)
        'lambda_mult': 0.3,     # 다양성을 고려하는 정도 (1은 최소 다양성, 0은 최대 다양성을 의미. 기본값은 0.5)
        },
)
```

**2. Prompt**
- 모든 답변은 제공된 컨텍스트에만 기반하여 작성되도록 함
- 외부 지식이나 사전 학습된 일반 상식은 사용하지 않도록 함
- 컨텍스트 내 명확한 근거가 없을 경우, **답변할 수 없음**으로 응답하도록 함

```python
from langchain.prompts import ChatPromptTemplate

translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following into English: {query}"
)
work_prompt = ChatPromptTemplate.from_template("""
Please answer following these rules:
1. Answer the questions based only on [Context].
2. If there is no [Context], answer that you don't know.
3. Do not use external knowledge.
4. If there is no clear basis in [Context], answer that you don't know.
5. You can refer to the previous conversation.

[Context]
{context}

[Question] 
{question}

[Answer]
""")
output_prompt = ChatPromptTemplate.from_template(
    "Translate the following into Korean: {output}"
)
```

**3. Chain**

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.8,
    top_p=0.7
)
output_parser = StrOutputParser()

def format_docs(docs):
    """ 참고 문서 연결 """
    return "\n\n".join([f"{i}: \n{doc.page_content}" for i, doc in enumerate(docs)])

def format_result(answer):
    """ 최종 응답 처리 """
    output = answer['output']
    context = answer['context']
    return f"{output}\n\n[Context]\n{context}"

# 체인 생성
translate_chain = translate_prompt | llm | output_parser
rag_chain = chroma_mmr | RunnableLambda(format_docs)
output_chain = work_prompt | llm | output_parser | output_prompt | llm | output_parser

main_chain = (
    translate_chain |
    RunnableParallel(
        question=RunnablePassthrough(),
        context=lambda x: rag_chain.invoke(x),
    ) | 
    RunnableParallel(
        context=lambda x: x['context'],
        output=output_chain
    ) | RunnableLambda(format_result)
)
```

**4. Output**

```python
# 체인 실행
query = "BART의 강점이 모야?"
answer = main_chain.invoke({"query": query})
print(f"쿼리: {query}")
print("답변:")
print(answer)
```

쿼리: BART의 강점이 모야? <br>
답변: <br>
[Context]를 바탕으로, BART의 강점은 다음과 같습니다:
- 모든 ROUGE 지표에서 약 6.0점 가량 이전 BERT 기반 연구를 크게 능가하여 텍스트 생성 작업에서 뛰어난 성능을 보입니다 (Context 0).
- 고품질의 샘플 출력을 생성합니다 (Context 0).
- CONVAI2 데이터셋의 자동 평가 지표에서 이전 연구들을 능가하며 대화 응답 생성에서 우수한 성과를 보입니다 (Context 0).
- BERT와 GPT 사전학습 방식을 모두 일반화한 Transformer 기반 신경망 기계번역 아키텍처를 사용하여 손상 및 재구성(corruption and reconstruction) 접근법으로 학습됩니다 (Context 1).
- 판별 작업에서 RoBERTa 및 XLNet과 비슷한 성능을 보여, 단방향 디코더 레이어가 이러한 작업에서 성능 저하를 일으키지 않음을 증명합니다 (Context 2).

[Context] <br>
0: 
BART outperforms the
best previous work, which leverages BERT, by roughly
6.0 points on all ROUGE metrics—representing a sig-
niﬁcant advance in performance on this problem. Qual-
itatively, sample quality is high (see §6). Dialogue We evaluate dialogue response generation
on C ONVAI2 (Dinan et al., 2019), in which agents
must generate responses conditioned on both the pre-
vious context and a textually-speciﬁed persona. BART
outperforms previous work on two automated metrics.
<br>
1: 
BART is trained by (1) corrupting text with an
arbitrary noising function, and (2) learning a
model to reconstruct the original text. It uses
a standard Tranformer-based neural machine
translation architecture which, despite its sim-
plicity, can be seen as generalizing BERT (due
to the bidirectional encoder), GPT (with the
left-to-right decoder), and many other more re-
cent pretraining schemes
<br>
2: 
. BART performs comparably to RoBERTa and
XLNet, suggesting that BART’s uni-directional decoder layers do not reduce performance on discriminative tasks

## Reference

- [링크1](https://brunch.co.kr/@ywkim36/146)
- [링크2](https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/)