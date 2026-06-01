from __future__ import annotations

import csv
import gc
from pathlib import Path

import streamlit as st
import tiktoken
import torch

from gpt_model import MODEL_SPECS, classify_sms, format_instruction, generate, load_model


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = tiktoken.get_encoding("gpt2")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(chapter: str, checkpoint_dir: str, device: torch.device):
    model_key = (chapter, checkpoint_dir, str(device))
    if st.session_state.get("model_key") != model_key:
        st.session_state.pop("model", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.session_state.model = load_model(
            MODEL_SPECS[chapter],
            Path(checkpoint_dir),
            device,
        )
        st.session_state.model_key = model_key
    return st.session_state.model


def infer_classifier_max_length(checkpoint_dir: str) -> int:
    train_csv = Path(checkpoint_dir) / "train.csv"
    if not train_csv.exists():
        return 120
    with train_csv.open(encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return max(len(TOKENIZER.encode(row["Text"])) for row in rows)


def generate_text(model, prompt: str, device: torch.device, max_new_tokens: int, temperature: float, top_k: int):
    encoded = TOKENIZER.encode(prompt, allowed_special={"<|endoftext|>"})
    inputs = torch.tensor(encoded, device=device).unsqueeze(0)
    output = generate(
        model=model,
        token_ids=inputs,
        max_new_tokens=max_new_tokens,
        context_size=model.pos_emb.weight.shape[0],
        temperature=temperature,
        top_k=top_k or None,
        eos_id=50256,
    )
    return TOKENIZER.decode(output.squeeze(0).tolist()[len(encoded):]).strip()


st.set_page_config(page_title="building_gpt2 local chat")
st.title("building_gpt2 local chat")
st.caption("ch05, ch06, ch07 체크포인트를 로컬에서 바꿔 가며 실행합니다.")

device = choose_device()
with st.sidebar:
    st.header("모델 설정")
    chapter = st.selectbox(
        "체크포인트",
        options=list(MODEL_SPECS),
        format_func=lambda key: MODEL_SPECS[key].label,
    )
    checkpoint_dir = st.text_input("체크포인트 폴더", value=str(ROOT))
    spec = MODEL_SPECS[chapter]
    st.caption(spec.description)
    st.code(str(Path(checkpoint_dir) / spec.checkpoint), language=None)
    st.write(f"실행 장치: `{device}`")

    if spec.task != "classify":
        max_new_tokens = st.slider("생성 토큰 수", 1, 300, 120)
        temperature = st.slider("temperature", 0.0, 2.0, 0.0, 0.1)
        top_k = st.slider("top-k (0은 미사용)", 0, 100, 0)
    else:
        classifier_max_length = st.number_input(
            "분류 패딩 길이",
            min_value=1,
            max_value=1024,
            value=infer_classifier_max_length(checkpoint_dir),
            help="train.csv가 있으면 자동 계산합니다. 없으면 ch06의 train_dataset.max_length 출력값을 넣으세요.",
        )

if "messages" not in st.session_state or st.session_state.get("chapter") != chapter:
    st.session_state.messages = []
    st.session_state.chapter = chapter

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            model = get_model(chapter, checkpoint_dir, device)
            with st.spinner("추론 중입니다..."):
                if spec.task == "classify":
                    answer = classify_sms(
                        prompt,
                        model,
                        TOKENIZER,
                        device,
                        max_length=int(classifier_max_length),
                    )
                elif spec.task == "instruction":
                    answer = generate_text(
                        model,
                        format_instruction(prompt),
                        device,
                        max_new_tokens,
                        temperature,
                        top_k,
                    )
                else:
                    answer = generate_text(
                        model,
                        prompt,
                        device,
                        max_new_tokens,
                        temperature,
                        top_k,
                    )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as error:
            st.error(str(error))
