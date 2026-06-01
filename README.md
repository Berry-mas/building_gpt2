# building_gpt2

간단한 요약

이 저장소는 nanoGPT를 기반으로 GPT-2 모델을 다루는 실습 자료와 체크포인트를 포함합니다. 주된 목적은 학습/실험용 노트북 실행과 사전학습된 GPT-2 체크포인트 활용입니다.

주요 내용

- 노트북: ch02.ipynb ~ ch07.ipynb — 실습 및 모델 학습/평가 흐름
- 데이터: `train.csv`, `validation.csv`, `test.csv`, `SMSSpamCollection.tsv`(sms_spam_collection)
- 참고: `gpt_download.py`는 라이선스 준수를 위해 제거되었습니다. [gpt_download.README.md](gpt_download.README.md)를 참고하세요.
- 모델 체크포인트: `model.pth`, `model_and_optimizer.pth`, `review_classifier.pth`
- 포함된 GPT-2 체크포인트: `gpt2/124M`, `gpt2/355M`

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
