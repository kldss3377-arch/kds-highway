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


def build_user_text_payload(
    budget_manwon: int,
    purpose: str,
    preferred_body: List[str],
    fuel: List[str],
    region: str,
    must_have_options: List[str],
    annual_mileage_km: Optional[int],
    family_size: Optional[int],
    extra_notes: str,
) -> str:
    return f"""
[사용자 조건]
- 예산(만원): {budget_manwon}
- 용도: {purpose}
- 선호 차종(바디타입): {', '.join(preferred_body) if preferred_body else '상관없음'}
- 연료: {', '.join(fuel) if fuel else '상관없음'}
- 지역: {region}
- 필수 옵션: {', '.join(must_have_options) if must_have_options else '없음/상관없음'}
- 연간 주행거리(추정, km): {annual_mileage_km if annual_mileage_km else '미입력'}
- 가족 인원: {family_size if family_size else '미입력'}
- 추가 메모: {extra_notes.strip() if extra_notes.strip() else '없음'}

[요청]
1) 위 조건에 맞는 중고차 후보를 3~5개 추천하고, 각 후보별 추천 이유/권장 연식/권장 주행거리/예상 가격 범위를 제시해 주세요.
2) 사진이 있다면 사진에서 보이는 특징(차종 추정, 외관 손상 가능성 등)을 '추정'으로 설명하고, 반드시 확인해야 할 항목을 정리해 주세요.
3) 구매 체크리스트, 리스크 경고, 협상 포인트, 다음 액션을 포함해 주세요.

{JSON_SCHEMA_HINT}
""".strip()


def call_openai_with_optional_image(
    client: OpenAI,
    model: str,
    user_text: str,
    image_data_url: Optional[str] = None,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """
    Uses OpenAI Responses API (v1+).
    If image is provided, use multimodal input.
    """
    if image_data_url:
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
    else:
        input_payload = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

    resp = client.responses.create(
        model=model,
        input=input_payload,
        instructions=SYSTEM_PROMPT,
        temperature=temperature,
    )

    # Responses API: resp.output_text contains the assistant text (if any)
    raw_text = getattr(resp, "output_text", "") or ""
    data = parse_json_safely(raw_text)
    if not data:
        # 안전장치: 파싱 실패 시 사용자에게 원문 일부 표시
        return {"_raw_text": raw_text, "_parse_failed": True}
    return data


# =========================
# UI
# =========================
st.title("🚗 중고 자동차 추천 앱")
st.caption("이미지(선택) + 조건 입력 → AI 분석 → 추천 보고서 출력")

with st.sidebar:
    st.header("설정")
    # 모델은 사용자가 바꿀 수 있게 하되 기본값은 가벼운 멀티모달 모델
    model = st.selectbox(
        "모델 선택",
        options=[
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
        ],
        index=0,
        help="Streamlit Cloud 비용/속도를 고려해 기본은 mini 모델 권장",
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
        ["출퇴근", "가족용(패밀리카)", "장거리/여행", "업무/영업", "초보 운전", "세컨카", "기타"],
        index=0,
    )

    preferred_body = st.multiselect(
        "선호 차종(바디타입)",
        ["경차", "소형", "준중형", "중형", "대형", "SUV", "미니밴/MPV", "해치백", "왜건", "픽업"],
        default=["준중형", "SUV"] if purpose in ["출퇴근", "장거리/여행"] else [],
    )

    fuel = st.multiselect(
        "선호 연료",
        ["가솔린", "디젤", "하이브리드", "전기", "LPG"],
        default=["가솔린", "하이브리드"],
    )

    region = st.text_input("거주/구매 지역(예: 대전, 천안, 서울)", value="대전")

    must_have_options = st.multiselect(
        "필수 옵션(있으면 좋음 포함)",
        ["후방카메라", "어댑티브 크루즈(ACC)", "차선유지(LFA/LKAS)", "통풍시트", "열선시트", "썬루프", "내비게이션", "블루투스", "HUD", "360도 어라운드뷰"],
        default=["후방카메라", "블루투스"],
    )

    annual_mileage_km = st.number_input("연간 주행거리 추정(km, 선택)", min_value=0, max_value=60000, value=12000, step=1000)
    family_size = st.number_input("가족 인원(선택)", min_value=0, max_value=10, value=2, step=1)

    extra_notes = st.text_area(
        "추가 메모(예: 유지비 중요, 고장 적은 차, 주차가 어려움, 아이 카시트 등)",
        height=120,
        placeholder="예) 유지비와 고장 적은 모델 우선. 주차가 어려워서 차 길이가 짧으면 좋겠습니다.",
    )

with col2:
    st.subheader("2) 사진 업로드(선택)")
    st.write("예: 마음에 드는 매물 사진(외관/실내/계기판/타이어/엔진룸/성능기록부 캡처 등)")
    uploaded = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png", "webp"])
    image_data_url = None

    if uploaded is not None:
        image_bytes = uploaded.read()
        mime = uploaded.type if uploaded.type else "image/jpeg"
        image_data_url = bytes_to_data_url(image_bytes, mime=mime)

        # 미리보기
        try:
            img = Image.open(uploaded)
            st.image(img, caption="업로드한 이미지 미리보기", use_container_width=True)
        except Exception:
            st.info("이미지 미리보기를 표시하지 못했지만, 분석은 계속 진행할 수 있습니다.")

st.divider()

run = st.button("🔎 추천 보고서 생성", type="primary", use_container_width=True)

if run:
    client = get_client()

    user_text = build_user_text_payload(
        budget_manwon=int(budget_manwon),
        purpose=purpose,
        preferred_body=preferred_body,
        fuel=fuel,
        region=region,
        must_have_options=must_have_options,
        annual_mileage_km=int(annual_mileage_km) if annual_mileage_km else None,
        family_size=int(family_size) if family_size else None,
        extra_notes=extra_notes,
    )

    with st.spinner("AI가 중고차 후보를 분석 중입니다..."):
        result = call_openai_with_optional_image(
            client=client,
            model=model,
            user_text=user_text,
            image_data_url=image_data_url,
            temperature=float(temperature),
        )

    if result.get("_parse_failed"):
        st.error("응답을 JSON으로 해석하지 못했습니다. 아래 원문을 확인해 주세요.")
        st.code(result.get("_raw_text", ""), language="text")
        st.stop()

    # Normalize / safety
    summary = result.get("summary", "")
    user_profile = result.get("user_profile", {})
    image_obs = result.get("image_observations", []) or []
    recs = clamp_recommendations(result.get("recommendations", []) or [], max_n=5)
    deal_tips = result.get("deal_tips", []) or []
    risks = result.get("risk_alerts", []) or []
    next_actions = result.get("next_actions", []) or []
    disclaimer = result.get("disclaimer", "")

    # =========================
    # Render Report
    # =========================
    st.subheader("3) 추천 보고서")
    st.caption(f"생성 시각: {now_kst_str()} (표시용)")

    if summary:
        st.success(summary)

    with st.expander("사용자 조건 요약", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("예산(만원)", safe_int(user_profile.get("budget_manwon"), int(budget_manwon)) or int(budget_manwon))
        c2.metric("용도", user_profile.get("purpose") or purpose)
        c3.metric("지역", user_profile.get("region") or region)

        st.write("**선호 차종(바디타입)**:", ", ".join(user_profile.get("preferred_body", preferred_body) or []) or "상관없음")
        st.write("**연료**:", ", ".join(user_profile.get("fuel", fuel) or []) or "상관없음")
        st.write("**필수 옵션**:", ", ".join(user_profile.get("must_have_options", must_have_options) or []) or "없음/상관없음")

    if image_data_url:
        with st.expander("이미지 관찰(추정)", expanded=True):
            if image_obs:
                for o in image_obs:
                    item = o.get("item", "")
                    conf = o.get("confidence", "낮음")
                    note = o.get("note", "")
                    st.write(f"- **{item}** (신뢰도: {conf}) — {note}")
            else:
                st.info("이미지로부터 유의미한 관찰 결과가 충분하지 않습니다. 다른 각도의 사진을 추가해 보세요.")

    st.markdown("### ✅ 추천 차량 TOP 리스트")
    if not recs:
        st.warning("추천 결과가 비어 있습니다. 입력 조건을 조금 더 단순화해서 다시 시도해 보세요.")
    else:
        for r in recs:
            rank = r.get("rank", "")
            model_name = r.get("model", "추천 모델")
            year_range = r.get("target_year_range", "")
            mileage = r.get("target_mileage_km", "")
            price = r.get("expected_price_manwon", {}) or {}
            pmin = safe_int(price.get("min"), None)
            pmax = safe_int(price.get("max"), None)

            with st.container(border=True):
                st.markdown(f"#### #{rank} {model_name}")
                cols = st.columns(3)
                cols[0].write(f"**권장 연식**: {year_range or '—'}")
                cols[1].write(f"**권장 주행거리**: {mileage or '—'}")
                if pmin is not None and pmax is not None:
                    cols[2].write(f"**예상 가격(만원)**: {pmin:,} ~ {pmax:,}")
                else:
                    cols[2].write("**예상 가격(만원)**: —")

                why = r.get("why_fit", []) or []
                watch = r.get("watch_out", []) or []
                focus = r.get("inspection_focus", []) or []

                if why:
                    st.write("**추천 이유**")
                    for x in why:
                        st.write(f"- {x}")

                if watch:
                    st.write("**주의 포인트(감가/결함/유지비)**")
                    for x in watch:
                        st.write(f"- {x}")

                if focus:
                    st.write("**점검 집중 항목(실차/성능기록부)**")
                    for x in focus:
                        st.write(f"- {x}")

    st.markdown("### 🧾 거래/협상 팁")
    if deal_tips:
        for t in deal_tips:
            topic = t.get("topic", "팁")
            detail = t.get("detail", "")
            st.write(f"- **{topic}**: {detail}")
    else:
        st.info("협상 팁이 제공되지 않았습니다. (재시도 시 더 자세히 요청해 보세요)")

    st.markdown("### ⚠️ 리스크 경고")
    render_risk_badges(risks)

    st.markdown("### 🧭 다음 액션")
    if next_actions:
        for i, a in enumerate(next_actions, 1):
            st.write(f"{i}. {a}")
    else:
        st.info("다음 액션이 제공되지 않았습니다.")

    if disclaimer:
        st.caption(disclaimer)
    else:
        st.caption("본 결과는 입력 정보와 사진을 기반으로 한 참고용 안내이며, 실차 점검/성능·상태점검기록부/보험이력 확인 후 최종 판단이 필요합니다.")

    # 다운로드(보고서 JSON)
    st.divider()
    st.markdown("### 📥 결과 다운로드")
    st.download_button(
        "결과(JSON) 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="usedcar_report.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
with st.expander("배포 체크리스트(필수)", expanded=False):
    st.markdown(
        """
1) GitHub 저장소에 `main.py`와 `requirements.txt` 업로드  
2) Streamlit Community Cloud에서 앱 생성 후 **메인 파일을 `main.py`로 지정**  
3) 앱 설정(Secrets)에 아래처럼 등록:

```toml
OPENAI_API_KEY = "sk-..."
