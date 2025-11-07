# app_onepage.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="도시가스 보급률 대시보드 - 시도×회사", layout="wide")

# 상대 경로
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "out").is_dir() else HERE.parent  # pages/면 한 단계 위가 루트
DEFAULT_CSV = (ROOT / "out" / "보급률_tidy_(2006-2024).csv").as_posix()

# ---------------------------
# 유틸/집계
# ---------------------------
def calc_agg_city_company(df: pd.DataFrame) -> pd.DataFrame:
    """연도×시도×회사 집계 후 보급률 계산 + 시도-회사 레이블 생성"""
    g = (
        df.groupby(["연도", "시도", "회사"], as_index=False)[["세대수", "수요가수"]]
          .sum(min_count=1)
    )
    g["보급률(%)"] = np.where(
        (g["세대수"] > 0) & (~g["세대수"].isna()),
        (g["수요가수"] / g["세대수"]) * 100.0,
        np.nan
    )
    g["시도-회사"] = g["시도"].astype(str) + " - " + g["회사"].astype(str)
    return g.sort_values(["시도", "회사", "연도"])

def transform_for_plot(df: pd.DataFrame, group_col: str, value_col: str, scale_mode: str):
    """절대값 / 전년대비(%) 변환"""
    out = df.copy()
    layout_kwargs = {}
    if scale_mode == "absolute":
        y_title = value_col
    elif scale_mode == "yoy_pct":
        out[value_col] = out.groupby(group_col)[value_col].pct_change() * 100.0
        y_title = f"{value_col} 전년대비(%)"
        layout_kwargs["yaxis"] = dict(tickformat=".1f")
    else:
        y_title = value_col
    return out, y_title, layout_kwargs

def drops_for_mode(df_abs: pd.DataFrame, df_trans: pd.DataFrame,
                   group_col: str, value_col: str, scale_mode: str) -> pd.DataFrame:
    """감소 지점(네모 마커용) 반환"""
    key = [group_col, "연도"]
    if scale_mode == "yoy_pct":
        cond = (df_trans[value_col] < 0) & df_trans[value_col].notna()
        return df_trans.loc[cond, key + [value_col]].copy()
    else:
        t = df_abs[[group_col, "연도", value_col]].copy()
        t["prev"] = t.groupby(group_col)[value_col].shift(1)
        dec = t[(t["prev"].notna()) & (t[value_col] < t["prev"])][key]
        return dec.merge(df_trans[key + [value_col]], on=key, how="left")

def non_decrease_groups(df_abs: pd.DataFrame, group_col: str, value_col: str) -> set:
    """'한번도 감소 없음' 그룹 집합 (절대값 기준)"""
    t = df_abs[[group_col, "연도", value_col]].copy()
    t["prev"] = t.groupby(group_col)[value_col].shift(1)
    dec = t[(t["prev"].notna()) & (t[value_col] < t["prev"])]
    has_dec = set(dec[group_col].unique().tolist())
    all_g  = set(t[group_col].unique().tolist())
    return all_g - has_dec

def add_group_markers(fig: go.Figure, drops_df: pd.DataFrame,
                      group_col: str, x_col: str, y_col: str):
    """감소 지점 네모 마커"""
    if drops_df.empty:
        return
    for g, sub in drops_df.groupby(group_col):
        fig.add_scatter(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            name=g, legendgroup=g, showlegend=False,
            marker_symbol="square-open", marker_size=14,
            marker_line_width=2, marker_color="red",
            hovertemplate=f"{group_col}=%{{name}}<br>연도=%{{x}}<br>{y_col}=%{{y:.2f}}<extra></extra>",
        )

def highlight_traces(fig: go.Figure, names: set):
    """하이라이트 라인 굵게"""
    for tr in fig.data:
        tr.update(line=dict(width=5 if tr.name in names else 2))

def apply_star_for_nondec(fig: go.Figure, nondec_set: set):
    """감소 없는 그룹은 별 마커"""
    if not nondec_set:
        return
    for tr in fig.data:
        if tr.name in nondec_set:
            tr.update(marker=dict(symbol="star", size=10, line=dict(width=1)))
        else:
            tr.update(marker=dict(size=6))

def add_deltas(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    out["세대수증감"]   = out.groupby(group_col)["세대수"].diff()
    out["수요가수증감"] = out.groupby(group_col)["수요가수"].diff()
    out["보급률증감"]   = out.groupby(group_col)["보급률(%)"].diff().round(2)
    return out

def dec_sets(df_all: pd.DataFrame, group_col: str, value_col: str):
    """부분집합 기준 감소/무감소 집합"""
    t = df_all[[group_col, "연도", value_col]].dropna().copy()
    t["prev"] = t.groupby(group_col)[value_col].shift(1)
    dec = set(t[(t["prev"].notna()) & (t[value_col] < t["prev"])][group_col].unique())
    allg = set(t[group_col].unique())
    nondec = allg - dec
    return dec, nondec

def fmt(items: set) -> str:
    return ", ".join(sorted(items)) if items else "없음"

def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """표 표시용 포맷(천단위/퍼센트)"""
    out = df.copy()
    int_cols = ["세대수", "세대수증감", "수요가수", "수요가수증감"]
    pct_cols = ["보급률(%)", "보급률증감"]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    return out

# ---------------------------
# 데이터 로드
# ---------------------------
st.sidebar.header("설정")
csv_default = st.session_state.get("csv_path", DEFAULT_CSV)
csv_path = st.sidebar.text_input("CSV 경로", value=csv_default)
st.session_state["csv_path"] = csv_path

if not os.path.isfile(csv_path):
    st.warning("CSV 파일 경로를 확인해 주세요.")
    st.stop()

df = pd.read_csv(csv_path, encoding="utf-8-sig")
for col in ["연도","세대수","수요가수","보급률"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df.rename(columns={"보급률":"보급률(%)"}, inplace=True)

# 기간 텍스트
years_all_full = sorted(df["연도"].dropna().unique().tolist())
period_text = f"{int(years_all_full[0])}년 ~ {int(years_all_full[-1])}년" if years_all_full else ""

# 시도×회사 집계
agg_pair = calc_agg_city_company(df)

# ---------------------------
# 헤더/설명
# ---------------------------
st.title("시도 × 회사 보급률 추이 대시보드")
st.markdown(
    f"""
**작성자** : 대성에너지 마케팅팀 배경호  
**출처** : 한국도시가스협회 → 연간 도시가스 통계 → *5. 보급률 실적*  
**출처 링크** : <http://www.citygas.or.kr/info/stats/index.jsp?sbranch_fk=2>  

본 화면은 **시도-회사 조합별 보급률**만을 집중적으로 보여줍니다.  
분석 기간: **{period_text}**

- 스케일: `절대값` / `전년대비(%)`
- 🔴 감소 연도 표시(네모), ⭐ 전체 기간 감소 없음 표시(별)
- 기본 하이라이트: **대구 - 대성**
---
"""
)

# ---------------------------
# 필터 UI
# ---------------------------
years_all = years_all_full.copy()

ALL_SIDOS = [
    "강원","경기","경남","경북","광주","대구","대전","부산","서울","울산",
    "인천","전남","전북","제주","충남","충북","세종"
]
sidos_in_data = [s for s in ALL_SIDOS if s in df["시도"].dropna().unique().tolist()]
companies_all = sorted([c for c in df["회사"].dropna().unique().tolist()])

# 기본 선택값: 대구/대성 우선 포함
DEFAULT_SIDOS = ["대구","서울","부산","대전","광주"]
default_sidos_in_data = [s for s in DEFAULT_SIDOS if s in sidos_in_data]
if "대구" not in default_sidos_in_data and "대구" in sidos_in_data:
    default_sidos_in_data = ["대구"] + default_sidos_in_data

# 회사 기본: 상위 사용량 + 대성 보장
top6 = (
    df.groupby("회사")["수요가수"].sum(min_count=1)
      .sort_values(ascending=False).head(6).index.tolist()
)
if "대성" not in top6 and "대성" in companies_all:
    top6 = ["대성"] + [c for c in top6 if c != "대성"]
default_comps = [c for c in top6 if c in companies_all]
if "대성" not in default_comps and "대성" in companies_all:
    default_comps = ["대성"] + default_comps

sel_years = st.sidebar.multiselect("연도", options=years_all, default=years_all)
sel_sidos = st.sidebar.multiselect("시도", options=sidos_in_data, default=default_sidos_in_data)
sel_comps = st.sidebar.multiselect("회사", options=companies_all, default=default_comps)

scale_mode = st.sidebar.radio(
    "스케일",
    ["absolute", "yoy_pct"],
    index=0,
    format_func=lambda x: {"absolute":"절대값", "yoy_pct":"전년대비(%)"}[x]
)

# 현재 필터에서 가능한 시도-회사 조합
pair_options = (
    agg_pair[
        agg_pair["시도"].isin(sel_sidos) &
        agg_pair["회사"].isin(sel_comps) &
        agg_pair["연도"].isin(sel_years)
    ]["시도-회사"].unique().tolist()
)

# ✅ 기본 하이라이트: 대구 - 대성만
default_highlight_pairs = ["대구 - 대성"] if "대구 - 대성" in pair_options else []

highlight_pairs = st.sidebar.multiselect(
    "강조할 시도-회사(복수 선택)",
    options=sorted(pair_options),
    default=default_highlight_pairs
)

# ---------------------------
# 필터 적용 데이터
# ---------------------------
f_pair = agg_pair[
    agg_pair["연도"].isin(sel_years) &
    agg_pair["시도"].isin(sel_sidos) &
    agg_pair["회사"].isin(sel_comps)
].copy()

# 요약(감소/무감소) – 보급률 기준
pair_rate_dec, pair_rate_nondec = dec_sets(f_pair, "시도-회사", "보급률(%)")

st.subheader("요약 (연도 필터 반영, 보급률 기준)")
st.markdown(f"- **감소한 시도-회사** ({len(pair_rate_dec)}): {fmt(pair_rate_dec)}")
st.markdown(f"- **감소 없는 시도-회사** ({len(pair_rate_nondec)}): {fmt(pair_rate_nondec)}")
st.markdown("---")

# ---------------------------
# 메인 그래프: 시도×회사 보급률
# ---------------------------
st.subheader("시도 × 회사 보급률 추이")
if f_pair.empty:
    st.info("선택된 조건에서 시도-회사 데이터가 없습니다.")
else:
    abs_df = f_pair[["연도","시도-회사","보급률(%)"]].copy()
    tr_df, y_label, y_layout = transform_for_plot(f_pair, "시도-회사", "보급률(%)", scale_mode)
    drops = drops_for_mode(abs_df, tr_df, "시도-회사", "보급률(%)", scale_mode)
    non_dec = non_decrease_groups(abs_df, "시도-회사", "보급률(%)")

    fig = px.line(tr_df, x="연도", y="보급률(%)", color="시도-회사", markers=True)
    highlight_traces(fig, set(highlight_pairs))
    apply_star_for_nondec(fig, non_dec)
    add_group_markers(fig, drops, "시도-회사", "연도", "보급률(%)")
    fig.update_layout(
        height=820, xaxis_title="연도", yaxis_title=y_label,
        legend_title="시도-회사", hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(groupclick="togglegroup"), **y_layout
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# ---------------------------
# 하단 표 (증감 포함)
# ---------------------------
st.subheader("시도 × 회사 집계 데이터 (전년대비 증감 포함)")
pair_table = add_deltas(
    f_pair.sort_values(["시도","회사","연도"]).copy(), "시도-회사"
)
pair_disp = format_for_display(
    pair_table[["연도","시도","회사","시도-회사","세대수","세대수증감","수요가수","수요가수증감","보급률(%)","보급률증감"]]
      .reset_index(drop=True)
)
st.caption(f"표 행수: {len(pair_disp)}")
st.dataframe(pair_disp, use_container_width=True, height=420)

# ---------------------------
# 엑셀 다운로드 (2시트)
# ---------------------------
with st.sidebar.expander("⬇ 엑셀 다운로드", expanded=True):
    export_mode = st.radio(
        "엑셀 내보내기 범위",
        ["전체 데이터", "현재 필터 적용"],
        index=0,
        help="엑셀에는 2개 시트(원본tidy / 시도-회사표)가 저장됩니다."
    )

    # 원본 tidy (전체 vs 필터)
    orig_df_all = (
        df[["연도","시도","회사","세대수","수요가수","보급률(%)"]]
          .sort_values(["연도","시도","회사"]).reset_index(drop=True)
    )
    orig_df_filtered = orig_df_all[
        orig_df_all["연도"].isin(sel_years) &
        orig_df_all["시도"].isin(sel_sidos) &
        orig_df_all["회사"].isin(sel_comps)
    ].reset_index(drop=True)

    if export_mode == "전체 데이터":
        xls_pair = add_deltas(
            agg_pair.sort_values(["시도","회사","연도"]).copy(), "시도-회사"
        ).reset_index(drop=True)
        xls_orig = orig_df_all
        export_name = "도시가스_보급률_시도회사_전체.xlsx"
        st.caption("엑셀에는 ‘전체 데이터’가 저장됩니다.")
    else:
        xls_pair = pair_table.sort_values(["시도","회사","연도"]).reset_index(drop=True)
        xls_orig = orig_df_filtered
        export_name = "도시가스_보급률_시도회사_필터.xlsx"
        st.caption("엑셀에는 ‘현재 필터 적용 데이터’가 저장됩니다.")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        xls_orig.to_excel(writer, sheet_name="원본(tidy)", index=False)
        xls_pair.to_excel(writer, sheet_name="시도-회사(표)", index=False)

        wb  = writer.book
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_pct = wb.add_format({"num_format": "0.00"})

        def style_sheet(ws, data: pd.DataFrame):
            headers = list(data.columns)
            for idx, name in enumerate(headers):
                if name in ["세대수","세대수증감","수요가수","수요가수증감"]:
                    ws.set_column(idx, idx, 14, fmt_int)
                elif name in ["보급률(%)","보급률증감"]:
                    ws.set_column(idx, idx, 12, fmt_pct)
                elif name in ["연도","시도","회사","시도-회사"]:
                    ws.set_column(idx, idx, 14)

        style_sheet(writer.sheets["원본(tidy)"], xls_orig)
        style_sheet(writer.sheets["시도-회사(표)"], xls_pair)

    st.download_button(
        "엑셀 파일 다운로드",
        data=buffer.getvalue(),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
