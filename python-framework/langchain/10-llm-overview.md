# LLM Overview

## LM (Language Model)

- 언어 모델(LM)은 주어진 문맥(Context)이 주어졌을 때, 다음 단어로 적합할 조건부 확률을 계산함
- 다음으로 단어를 선택하는 방법에 따라 단어를 선택하고, 문장을 생성하거나 완성하는 등의 작업을 수행함
    - 결정적 방법 (Deterministic)은 가장 확률이 높은 단어를 선택하는 방법으로, 항상 같은 결과가 나옴 (Greedy Selection)
    - 확률적 방법 (Probabilistic)은 단어의 확률 분포에 따라 단어를 랜덤으로 선택하여, 각 실행마다 다른 결과가 나옴 (Random Sampling)

## LLM 작동 방식

- 프롬프트 입력 (Prompt) → 모델 처리 (LLM) → 응답 생성 (Completion)
- 우리는 보통 Foundation Model(만들어진 모델)을 사용함
    - Gemma (이미지 처리), Lammma, Q-won (한국어 처리), Exaone 등
    
