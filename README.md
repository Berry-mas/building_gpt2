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

| 파일 | 주요 내용 |
| --- | --- |
| `ch02.ipynb` | 정규표현식 기반 토큰화부터 BPE 토크나이저, sliding window 데이터 로더, 토큰 및 위치 임베딩까지 LLM 입력 처리 과정을 다룹니다. |
| `ch03.ipynb` | self-attention을 단계별로 계산하고, 학습 가능한 QKV 가중치, causal mask, dropout, multi-head attention으로 확장합니다. |
| `ch04.ipynb` | LayerNorm, GELU, feed-forward network, residual connection, transformer block을 조합하여 GPT 모델과 기본 텍스트 생성 함수를 구현합니다. |
| `ch05.ipynb` | 다음 토큰 예측 손실을 계산하고 GPT를 사전 훈련합니다. temperature 및 top-k 샘플링, 모델 저장과 로드, OpenAI GPT-2 가중치 적용도 다룹니다. |
| `ch06.ipynb` | GPT-2에 분류 헤드를 추가하고 SMS 스팸 데이터로 fine-tuning합니다. 데이터 균형 조정, 정확도 평가, 분류기 저장과 추론 과정을 포함합니다. |
| `ch07.ipynb` | Alpaca 스타일 지시 데이터와 커스텀 collate 함수를 준비하고 GPT-2 Medium을 instruction fine-tuning합니다. 응답 저장과 Ollama 기반 자동 평가도 다룹니다. |

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
