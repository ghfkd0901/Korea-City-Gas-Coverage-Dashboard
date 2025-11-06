# main.py
from pathlib import Path
import streamlit as st

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="도시가스 보급률 대시보드 · 홈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data"

def _find_default_csv() -> Path | None:
    """우선순위: ./out → ./data"""
    for p in [OUT_DIR / "보급률_tidy_(2006-2024).csv", DATA_DIR / "보급률_tidy_(2006-2024).csv"]:
        if p.is_file():
            return p
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return None

# 다른 페이지들이 사용하는 기본 CSV 경로 세팅만 조용히 유지
DEFAULT_PATH = _find_default_csv()
if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = str(DEFAULT_PATH) if DEFAULT_PATH else ""

# ---------------------------
# 본문
# ---------------------------
st.markdown(
    """
    # 전국 도시가스 보급률 분석 대시보드

    **전국 도시가스 보급률 분석 대시보드에 오신 것을 환영합니다.**  
    아래 로고 이미지는 리포지토리 상대경로(`./logo/logo_kor.png`)에서 불러옵니다.
    """
)

# 로고 이미지 (상대경로)
logo_path = ROOT / "logo" / "logo_kor.png"
st.markdown("---")
st.subheader(" ")

if logo_path.is_file():
    st.image(str(logo_path), use_container_width=True)
else:
    st.warning("로고 이미지를 찾을 수 없습니다. `./logo/logo_kor.png` 경로에 파일을 배치해 주세요.")

st.markdown("---")

# 분석 페이지 링크
st.subheader("분석 페이지로 이동")
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/전국도시가스보급률_비교.py", label="📊 보급률 비교/추이 (시도·회사)")
with c2:
    st.page_link("pages/전국도시가스보급률_경주.py", label="🏁 보급률 바 차트 레이스")

# 현재 CSV 경로 간단 안내
csv_path_info = st.session_state.get("csv_path", "")
if csv_path_info:
    st.caption(f"현재 분석에 사용할 기본 CSV: `{Path(csv_path_info).as_posix()}`")
else:
    st.caption("기본 CSV를 찾지 못했습니다. `./out` 또는 `./data`에 `보급률_tidy_(2006-2024).csv`를 두면 자동 인식합니다.")

# ✅ 주의:
# - 데이터 업로드 및 경로 입력 UI(파일 업로더 등)는 모두 제거했습니다.
# - 메인 페이지는 정적 로고만 노출하고, 분석 페이지로 이동하는 단순 구성입니다.
