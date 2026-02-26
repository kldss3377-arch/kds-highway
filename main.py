import base64
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from PIL import Image

# =========================
# Streamlit App Config
# =========================
st.set_page_config(
    page_title="중고차 추천 앱",
    page_icon="🚗",
    layout="wide",
)

# =========================
# Helper Functions
# =========================
def get_client() -> OpenAI:
    """
    OpenAI 클라이언트
    API Key는 Streamlit Secrets에서 읽어옵니다.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록해 주세요.")
        st.stop()
    return OpenAI(api_key=api_key)


def bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """이미지 바이트를 data URL로 변환"""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """안전하게 정수로 변환"""
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        return default


def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """안전하게 실수로 변환"""
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def strip_code_fences(text: str) -> str:
    """텍스트에서 코드 블록 제거"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_json_safely(text: str) -> Dict[str, Any]:
    """가능한 한 JSON을 파싱"""
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        return {}


def now_kst_str() -> str:
    """현재 시간 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# =========================
# LLM Core
# =========================
SYSTEM_PROMPT = """당신은 대한민국 중고차 구매를 돕는 전문가입니다.
사용자 입력(예산, 용도, 선호, 지역, 주행거리/연식, 옵션 등)을 바탕으로
'모델 추천'과 '구매 체크리스트', '리스크 경고', '협상 포인트', '다음 액션'을 한국어로 작성합니다.

결과는 반드시 JSON 단일 객체로만 출력.
"""

JSON_SCHEMA_HINT = """반드시 아래 형태의 JSON 객체로만 답하세요:

{
  "summary": "한 줄 요약",
  "user_profile": {
    "budget_manwon": 0,
    "purpose": "",
    "preferred_body": [],
    "fuel": [],
    "must_have_options": [],
    "region": "",
    "annual_mileage_km": null,
    "family_size": null
  },
  "image_observations": [
    {"item": "", "confidence": "낮음/중간/높음", "note": ""}
  ],
  "recommendations": [
    {
      "rank": 1,
      "model": "예: 아반떼 1.6",
      "why_fit": ["", ""],
      "target_year_range": "2020~2022",
      "target_mileage_km": "3만~8만",
      "expected_price_manwon": {"min": 0, "max": 0},
      "watch_out": ["", ""],
      "inspection_focus": ["", ""]
    }
  ],
  "deal_tips": [
    {"topic": "", "detail": ""}
  ],
  "risk_alerts": [
    {"level": "낮음/중간/높음", "title": "", "description": "", "check": ""}
  ],
  "next_actions": [
    "다음에 할 일 1",
    "다음에 할 일 2"
  ],
  "disclaimer": "면책 문구"
}
"""

# =========================
# UI
# =========================
st.title("🚗 중고 자동차 추천 앱")
st.caption("이미지(선택) + 조건 입력 → AI 분석 → 추천 보고서 출력")

with st.sidebar:
    st.header("설정")
    model = st.selectbox(
        "모델 선택",
        options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
        index=0,
        help="모델 선택"
    )
    temperature = st.slider("창의성(temperature)", 0.0, 1.0, 0.4, 0.05)
    st.divider()
    st.markdown("✅ API 키는 **Streamlit Secrets**에 `OPENAI_API_KEY`로 저장되어 있어야 합니다.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) 조건 입력")
    budget_manwon = st.number_input("예산(만원)", min_value=200, max_value=20000, value=2500, step=50)
    purpose = st.selectbox(
        "주요 용도",
        ["출퇴근", "가족용", "장거리/여행", "업무/영업", "초보 운전", "세컨카", "기타"],
        index=0,
    )
    preferred_body = st.multiselect(
        "선호 차종(바디타입)",
        ["경차", "소형", "준중형", "중형", "대형", "SUV", "미니밴", "왜건", "픽업"],
        default=["준중형", "SUV"]
    )
    fuel = st.multiselect(
        "선호 연료",
        ["가솔린", "디젤", "하이브리드", "전기", "LPG"],
        default=["가솔린", "하이브리드"],
    )
    region = st.text_input("거주/구매 지역(예: 대전, 서울)", value="대전")
    must_have_options = st.multiselect(
        "필수 옵션",
        ["후방카메라", "어댑티브 크루즈", "차선유지", "통풍시트", "열선시트", "썬루프"],
        default=["후방카메라", "블루투스"],
    )
    annual_mileage_km = st.number_input("연간 주행거리(km, 선택)", min_value=0, max_value=60000, value=12000, step=1000)
    family_size = st.number_input("가족 인원(선택)", min_value=0, max_value=10, value=2)

with col2:
    st.subheader("2) 사진 업로드(선택)")
    uploaded = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png", "webp"])
    image_data_url = None
    if uploaded is not None:
        image_bytes = uploaded.read()
        mime = uploaded.type if uploaded.type else "image/jpeg"
        image_data_url = bytes_to_data_url(image_bytes, mime=mime)
        img = Image.open(uploaded)
        st.image(img, caption="업로드한 이미지 미리보기", use_container_width=True)

st.divider()

run = st.button("🔎 추천 보고서 생성", type="primary", use_container_width=True)

if run:
    client = get_client()

    user_text = f"""
[사용자 조건]
- 예산(만원): {budget_manwon}
- 용도: {purpose}
- 선호 차종: {', '.join(preferred_body)}
- 연료: {', '.join(fuel)}
- 지역: {region}
- 필수 옵션: {', '.join(must_have_options)}
- 연간 주행거리(km): {annual_mileage_km}
- 가족 인원: {family_size}

[요청]
1) 위 조건에 맞는 중고차 후보를 추천해 주세요.
2) 사진을 바탕으로 차량 상태를 분석해 주세요.
    """.strip()

    with st.spinner("AI가 중고차 후보를 분석 중입니다..."):
        # OpenAI 호출 함수 (응답 받기)
        response = client.Completions.create(
            model=model,
            prompt=user_text,
            temperature=temperature,
        )
        
        # 응답 처리
        result = parse_json_safely(response['choices'][0]['text'])

    # =========================
    # 추천 결과 출력
    # =========================
    st.subheader("3) 추천 보고서")
    if result.get("summary"):
        st.success(result["summary"])

    st.markdown("### 추천 차량")
    if "recommendations" in result:
        for rec in result["recommendations"]:
            st.write(f"**모델**: {rec['model']}")
            st.write(f"**예상 가격**: {rec['expected_price_manwon']['min']} ~ {rec['expected_price_manwon']['max']}만원")
            st.write(f"**추천 이유**: {', '.join(rec['why_fit'])}")
    else:
        st.warning("추천 결과가 비어 있습니다.")

    st.markdown("### 거래 팁")
    if "deal_tips" in result:
        for tip in result["deal_tips"]:
            st.write(f"- **{tip['topic']}**: {tip['detail']}")

    st.markdown("### 리스크 경고")
    if "risk_alerts" in result:
        for risk in result["risk_alerts"]:
            st.write(f"**{risk['title']}**: {risk['description']}")

    st.markdown("### 다음 액션")
    if "next_actions" in result:
        for action in result["next_actions"]:
            st.write(f"- {action}")

    st.divider()
    st.download_button(
        "결과 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="recommendations.json",
        mime="application/json",
    )
