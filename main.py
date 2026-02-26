import base64
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
# Helpers
# =========================
def get_client() -> OpenAI:
    """
    OpenAI v1+ client.
    API Key is loaded from Streamlit secrets.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록해 주세요.")
        st.stop()
    return OpenAI(api_key=api_key)


def bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip().replace(",", "")
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace(",", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def strip_code_fences(text: str) -> str:
    """
    모델이 ```json ... ``` 형태로 답을 줄 때를 대비.
    """
    text = text.strip()
    # Remove triple backtick fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_json_safely(text: str) -> Dict[str, Any]:
    """
    가능한 한 JSON을 파싱. 실패하면 빈 dict 반환.
    """
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        # JSON 객체만 추출 시도
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def now_kst_str() -> str:
    # 서버 타임존은 다를 수 있지만, 표시용으로만 사용
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def render_risk_badges(risks: List[Dict[str, Any]]):
    if not risks:
        st.success("현재 입력 기준으로 큰 위험 신호는 두드러지지 않습니다. (단, 실차 점검/성능기록부 확인은 필수)")
        return
    for r in risks:
        level = (r.get("level") or "주의").strip()
        title = r.get("title") or "리스크"
        desc = r.get("description") or ""
        check = r.get("check") or ""
        if level in ["높음", "고위험", "High"]:
            st.error(f"⚠️ [{level}] {title}\n\n- {desc}\n- 확인: {check}")
        elif level in ["중간", "보통", "Medium"]:
            st.warning(f"🟠 [{level}] {title}\n\n- {desc}\n- 확인: {check}")
        else:
            st.info(f"🔵 [{level}] {title}\n\n- {desc}\n- 확인: {check}")


def clamp_recommendations(items: List[Dict[str, Any]], max_n: int = 5) -> List[Dict[str, Any]]:
    if not items:
        return []
    return items[:max_n]


# =========================
# LLM Core
# =========================
SYSTEM_PROMPT = """당신은 대한민국 중고차 구매를 돕는 전문가(정비/거래/보험/감가 관점 포함)입니다.
사용자 입력(예산, 용도, 선호, 지역, 주행거리/연식, 옵션, 사진 등)을 바탕으로
'모델 추천'과 '구매 체크리스트', '리스크 경고', '협상 포인트', '다음 액션'을 한국어로 작성합니다.

중요:
- 확정적 진단/단정 금지(사진만으로 사고/침수/누유를 확정하지 말 것). 가능성/추정으로 표현.
- 불확실하면 '추가로 확인할 항목'을 제시.
- 결과는 반드시 JSON 단일 객체로만 출력.
- 가격은 '만원' 단위로 가정 가능(명확히 표시).
- 특정 업체/딜러 실명 추천은 하지 말고, 일반적 기준으로 안내.
"""

JSON_SCHEMA_HINT = """반드시 아래 형태의 JSON 객체로만 답하세요(키 이름 유지, 누락 최소화):

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
      "model": "예: 아반떼 (CN7) 1.6",
      "why_fit": ["", ""],
      "target_year_range": "예: 2020~2022",
      "target_mileage_km": "예: 3만~8만",
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

# 여기에 나머지 코드...
