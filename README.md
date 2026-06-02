# building_gpt2

간단한 요약

이 저장소는 nanoGPT를 기반으로 GPT-2 모델을 다루는 실습 자료와 체크포인트를 포함합니다. 주된 목적은 학습/실험용 노트북 실행과 사전학습된 GPT-2 체크포인트 활용입니다.

주요 내용

- 노트북: ch02.ipynb ~ ch07.ipynb — 실습 및 모델 학습/평가 흐름
- 데이터: `train.csv`, `validation.csv`, `test.csv`, `SMSSpamCollection.tsv`(sms_spam_collection)
- 참고: `gpt_download.py`는 라이선스 준수를 위해 제거되었습니다. [gpt_download.README.md](gpt_download.README.md)를 참고하세요.
- 모델 체크포인트: `model.pth`, `model_and_optimizer.pth`, `review_classifier.pth`
- 포함된 GPT-2 체크포인트: `gpt2/124M`, `gpt2/355M`

챕터별 구성

`ch02`부터 `ch07`까지는 디렉터리가 아니라 순서대로 실행하는 Jupyter Notebook입니다.

```text
building_gpt2/
├── ch02.ipynb                  # 텍스트 전처리와 임베딩
├── ch03.ipynb                  # attention 메커니즘 구현
├── ch04.ipynb                  # GPT 모델 구조 구현
├── ch05.ipynb                  # 사전 훈련과 GPT-2 가중치 로드
├── ch06.ipynb                  # 스팸 분류 fine-tuning
├── ch07.ipynb                  # instruction fine-tuning
├── the-verdict.txt             # 사전 훈련 실습용 텍스트
├── train.csv                   # 스팸 분류 훈련 데이터
├── validation.csv              # 스팸 분류 검증 데이터
├── test.csv                    # 스팸 분류 테스트 데이터
├── instruction-data.json       # instruction fine-tuning 데이터
└── local_chat/                 # 체크포인트를 실행하는 Streamlit UI
```

### `ch02.ipynb`: 텍스트 데이터 전처리

- **사용 데이터:** 단편 소설 *The Verdict*의 본문인 `the-verdict.txt`
- **학습 내용:** 모델 가중치를 훈련하지는 않습니다. 이후 챕터에서 다음 토큰 예측 학습에 사용할 입력과 타깃 데이터를 만드는 과정을 준비합니다.
- **구현 내용:** 정규표현식 기반 토큰 분리, 단어 사전 구축, 문자열과 토큰 ID 사이를 변환하는 `SimpleTokenizerV1` 및 `SimpleTokenizerV2`, `<|unk|>`와 `<|endoftext|>` 특수 토큰 처리를 구현합니다.
- **추가 실습:** GPT-2의 BPE 토크나이저인 `tiktoken`을 적용하고, sliding window 방식으로 입력 시퀀스와 한 토큰 뒤의 타깃 시퀀스를 만드는 `GPTDatasetV1` 및 `create_dataloader_v1`을 구현합니다. 마지막으로 토큰 임베딩과 위치 임베딩을 더해 모델 입력 텐서를 만듭니다.

### `ch03.ipynb`: Attention 메커니즘 구현

- **사용 데이터:** 별도의 외부 데이터셋 없이 작은 예제 임베딩 텐서를 사용합니다.
- **학습 내용:** 전체 모델 훈련은 진행하지 않습니다. GPT의 핵심 연산인 attention이 각 토큰의 문맥 벡터를 만드는 과정을 단계별로 확인합니다.
- **구현 내용:** 단순 dot product 기반 self-attention부터 시작하여 학습 가능한 Query, Key, Value 가중치를 가진 `SelfAttention_v1`과 `SelfAttention_v2`를 구현합니다.
- **추가 실습:** 미래 토큰을 참조하지 못하게 하는 causal mask와 dropout을 적용한 `CausalAttention`, 여러 attention head를 결합한 `MultiHeadAttentionWrapper`, QKV 행렬을 효율적으로 분할하는 `MultiHeadAttention`을 구현합니다.

### `ch04.ipynb`: GPT 모델 구조 구현

- **사용 데이터:** 구조 검증을 위한 임의 입력 텐서와 짧은 시작 문장을 사용합니다.
- **학습 내용:** 아직 데이터셋으로 모델을 훈련하지 않습니다. `GPT_CONFIG_124M` 설정을 기준으로 GPT-2 Small과 같은 형태의 모델을 직접 조립하고 출력 shape과 생성 흐름을 검증합니다.
- **구현 내용:** `LayerNorm`, `GELU`, `FeedForward`, residual connection, `MultiHeadAttention`, `TransformerBlock`을 구현합니다. 이를 토큰 임베딩, 위치 임베딩, transformer block, 최종 정규화 층, 출력 층으로 연결하여 `GPTModel`을 완성합니다.
- **추가 실습:** 현재 문맥에서 마지막 토큰의 logits를 선택하고 다음 토큰을 반복해서 붙이는 greedy decoding 함수 `generate_text_simple`을 구현합니다.

### `ch05.ipynb`: 다음 토큰 예측 사전 훈련

- **사용 데이터:** `the-verdict.txt`를 앞부분 90%의 훈련 세트와 뒷부분 10%의 검증 세트로 분리합니다.
- **학습 내용:** `ch04`에서 만든 GPT 모델을 레이블 없는 텍스트로 자기지도학습합니다. 입력 시퀀스보다 한 토큰 뒤로 이동한 타깃 시퀀스를 사용하여, 각 위치에서 다음 토큰을 예측하도록 cross-entropy loss를 최소화합니다.
- **구현 내용:** 배치 및 데이터 로더 단위 손실 계산, 훈련 및 검증 평가, 샘플 텍스트 생성, AdamW 기반 훈련 루프 `train_model_simple`, loss 그래프 출력을 구현합니다.
- **추가 실습:** greedy decoding 외에 temperature scaling과 top-k sampling을 적용한 `generate` 함수를 구현합니다. 직접 학습한 모델과 optimizer를 체크포인트로 저장하고 다시 불러오는 방법도 다룹니다.
- **사전 훈련 가중치 활용:** 직접 학습과 별도로 OpenAI GPT-2 Small(124M)의 기존 가중치를 내려받아 직접 구현한 `GPTModel`에 매핑합니다.

### `ch06.ipynb`: 스팸 분류 Fine-tuning

- **사용 데이터:** UCI SMS Spam Collection의 문자 메시지 데이터입니다. `ham` 샘플을 undersampling하여 `spam`과 개수를 맞춘 뒤 `train.csv`, `validation.csv`, `test.csv`로 저장합니다. 실제 코드의 분할 비율은 훈련 70%, 검증 10%, 테스트 20%입니다.
- **학습 내용:** OpenAI GPT-2 Small(124M)의 사전 훈련 가중치를 불러온 뒤 문자 메시지가 정상 메시지인지 스팸인지 분류하도록 fine-tuning합니다.
- **구현 내용:** 메시지를 GPT-2 BPE로 토큰화하고 `<|endoftext|>` 토큰으로 길이를 맞추는 `SpamDataset`을 구현합니다. 기존 언어 모델 출력 층을 두 개 클래스의 logits를 출력하는 분류 헤드로 교체합니다.
- **Fine-tuning 범위:** 대부분의 모델 파라미터를 동결하고 새 분류 헤드, 마지막 transformer block, 최종 LayerNorm을 학습합니다. causal attention에서 마지막 토큰이 이전 문맥 전체를 참고할 수 있으므로 마지막 토큰의 logits로 분류합니다.
- **평가 및 결과물:** cross-entropy loss와 정확도를 계산하고, 새로운 메시지를 분류하는 `classify_review`를 구현합니다. 학습된 분류기는 `review_classifier.pth`로 저장합니다.

### `ch07.ipynb`: Instruction Fine-tuning

- **사용 데이터:** `instruction`, 선택적 `input`, `output` 필드를 가진 공개 지시-응답 데이터 `instruction-data.json`입니다. 실제 코드의 분할 비율은 훈련 85%, 테스트 10%, 검증 5%입니다.
- **학습 내용:** OpenAI GPT-2 Medium(355M)의 사전 훈련 가중치를 불러온 뒤, 사용자의 지시에 적절한 응답을 생성하도록 supervised instruction fine-tuning을 진행합니다.
- **구현 내용:** 각 샘플을 Alpaca 스타일의 `### Instruction`, `### Input`, `### Response` 형식으로 변환합니다. 배치마다 길이가 다른 샘플을 패딩하고, 손실 계산에서 불필요한 패딩 토큰을 `-100`으로 마스킹하는 `InstructionDataset`과 `custom_collate_fn`을 구현합니다.
- **Fine-tuning 방식:** 사전 훈련과 마찬가지로 다음 토큰 예측 cross-entropy loss를 사용하지만, 이번에는 지시문과 기대 응답을 연결한 텍스트를 학습합니다.
- **평가 및 결과물:** 테스트 세트에 대한 모델 응답을 생성하여 `instruction-data-with-response.json`으로 내보내고, fine-tuning된 모델을 `gpt2-medium355M-sft.pth`로 저장합니다. 선택적으로 Ollama의 Llama 3 모델을 호출하여 생성 응답을 자동 채점합니다.

요구사항 (예시)

- Python 3.8+
- PyTorch
- Transformers, tokenizers (선택)
- Jupyter

간단 설치 예시

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch
```

VESSL CUDA 12.4 환경에서는 호환되는 PyTorch 빌드를 설치합니다:

```bash
pip install --upgrade pip
pip install -r requirements-vessl.txt
```

빠른 시작

1. 로컬에서 Jupyter Lab을 실행합니다:

```bash
jupyter lab
```

2. 원하는 노트북(`ch02.ipynb` 등)을 열고 셀을 순서대로 실행하세요.
3. 필요하면 `gpt_download.py`로 체크포인트를 가져오세요:

```bash
python gpt_download.py
```

로컬 모델 UI

- `ch05`, `ch06`, `ch07`에서 저장한 체크포인트를 브라우저에서 바꿔 가며
  실행하려면 [local_chat/README.md](local_chat/README.md)를 참고하세요.

참고

- 저장소에 이미 `gpt2/124M` 및 `gpt2/355M` 폴더가 있으므로 별도 다운로드가 필요하지 않을 수 있습니다.
- 노트북은 GPU 사용을 가정할 수 있으니, GPU 환경이 없으면 설정을 CPU 모드로 조정하세요.

Third-party code and license

- The file `gpt_download.py` (originally by Sebastian Raschka) has been removed
  from this repository to comply with licensing requirements. Please refer to
  [gpt_download.README.md](gpt_download.README.md) for how to download and use
  it directly from the original source.
- The original code is licensed under the Apache License 2.0. Full license text
  is included in the repository as `LICENSE`.
