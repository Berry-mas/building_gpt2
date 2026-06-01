# Local chat UI

`ch05`, `ch06`, `ch07`에서 저장한 체크포인트를 브라우저에서 바꿔 가며
실행하는 로컬 Streamlit 앱입니다.

## Checkpoints

저장소 루트에 사용할 체크포인트를 둡니다.

| Notebook | Checkpoint | Mode |
| --- | --- | --- |
| `ch05.ipynb` | `model.pth` | next-token text generation |
| `ch06.ipynb` | `review_classifier.pth` | SMS spam classification |
| `ch07.ipynb` | `gpt2-medium355M-sft.pth` | instruction response generation |

체크포인트는 `.gitignore`에 의해 Git에 추가되지 않습니다.

## Run

저장소 루트에서 실행합니다.

```bash
pip install -r local_chat/requirements.txt
streamlit run local_chat/app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. VESSL 워크스페이스에서는
포트 `8501`을 노출한 뒤 해당 URL로 접속합니다.

VESSL CUDA 12.4 워크스페이스를 새로 만들었다면 저장소 루트에서 다음과
같이 설치하고 실행합니다.

```bash
pip install --upgrade pip
pip install -r requirements-vessl.txt
streamlit run local_chat/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

워크스페이스를 만들 때 포트 `8501`을 함께 노출해야 합니다.

## Notes

- 모델을 변경하면 대화 기록이 초기화됩니다.
- `ch06` 분류 모델의 `분류 패딩 길이`는 노트북 학습 시 출력된
  `train_dataset.max_length`와 동일하게 설정해야 합니다.
- `ch05`는 작은 데이터로 학습 원리를 실습한 모델이므로 ChatGPT와 같은
  응답 품질을 기대하기 어렵습니다.
