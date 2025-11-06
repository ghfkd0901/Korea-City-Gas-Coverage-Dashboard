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
st.set_page_config(page_title="도시가스 보급률 대시보드", layout="wide")

# 상대 경로로 변경
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "out").is_dir() else HERE.parent  # pages/면 한 단계 위가 루트
DEFAULT_CSV = (ROOT / "out" / "보급률_tidy_(2006-2024).csv").as_posix()

# ---------------------------
# 집계/유틸 함수
# ---------------------------
def calc_agg_city(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["연도", "시도"], as_index=False)[["세대수", "수요가수"]]
           .sum(min_count=1))
    g["보급률(%)"] = np.where(
        (g["세대수"] > 0) & (~g["세대수"].isna()),
        (g["수요가수"] / g["세대수"]) * 100.0,
        np.nan
    )
    return g.sort_values(["시도", "연도"])

def calc_agg_company(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["연도", "회사"], as_index=False)[["세대수", "수요가수"]]
           .sum(min_count=1))
    g["보급률(%)"] = np.where(
        (g["세대수"] > 0) & (~g["세대수"].isna()),
        (g["수요가수"] / g["세대수"]) * 100.0,
        np.nan
    )
    return g.sort_values(["회사", "연도"])

def transform_for_plot(df: pd.DataFrame, group_col: str, value_col: str,
                       scale_mode: str):
    """
    scale_mode: 'absolute' | 'yoy_pct'
    return: (df_transformed, y_title, layout_kwargs)
    """
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
    """
    네모박스 위치(감소 지점) 반환
    - yoy_pct : 변환값 < 0
    - absolute: 올해 < 전년 (절대값 기준), 표시는 변환된 y값으로
    """
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
    """감소 지점 네모 마커(legendgroup 연동)"""
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
    """names에 있는 라인은 굵게(5), 나머지는 보통(2)"""
    for tr in fig.data:
        tr.update(line=dict(width=5 if tr.name in names else 2))

def apply_star_for_nondec(fig: go.Figure, nondec_set: set):
    """감소 없는 그룹은 마커를 별(⭐)로 변경"""
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
    """전체 데이터(또는 요약 대상으로 들어온 부분집합) 기준의 감소/무감소 집합"""
    t = df_all[[group_col, "연도", value_col]].dropna().copy()
    t["prev"] = t.groupby(group_col)[value_col].shift(1)
    dec = set(t[(t["prev"].notna()) & (t[value_col] < t["prev"])][group_col].unique())
    allg = set(t[group_col].unique())
    nondec = allg - dec
    return dec, nondec

def fmt(items: set) -> str:
    return ", ".join(sorted(items)) if items else "없음"

# 표 표시용 포맷(천단위/퍼센트) – 화면 렌더링 전용
def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
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

# 전체 집계 (요약용/그래프용 원천)
agg_city = calc_agg_city(df)
agg_comp = calc_agg_company(df)

# 기간 자동 도출(설명 문구용)
years_all_full = sorted(df["연도"].dropna().unique().tolist())
period_text = f"{int(years_all_full[0])}년 ~ {int(years_all_full[-1])}년" if years_all_full else ""

# ---------------------------
# 최상단: 요약(연도 필터 반영)
# ---------------------------
st.title("전국 도시가스 보급률 현황")

# 상단 정보/설명 영역
st.markdown(
    f"""
**작성자** : 대성에너지 마케팅팀 배경호 사원  
**출처** : 한국도시가스협회 → 자료실 → 연간 도시가스 통계 → *5. 보급률 실적*  
**출처 링크** : <http://www.citygas.or.kr/info/stats/index.jsp?sbranch_fk=2>  

본 대시보드는 **전국 시도 및 도시가스 공급사별 보급률, 세대수, 수요가수(가구 수요처)의 추이 분석**을 위해 제작되었습니다.  
분석 기간은 **{period_text}**이며, 데이터는 도시가스협회 공식 통계 기반입니다.

### 🎮 사용 방법
- **사이드바**에서 연도, 시도, 회사를 선택하여 그래프를 필터링할 수 있습니다.  
- **스케일 옵션**  
  - `절대값` : 실제 수치(세대수·수요가수·보급률) 표시  
  - `전년대비(%)` : 전년 대비 증가·감소 비율 표시  
- **강조할 시도·회사 선택**  
  → 해당 항목의 선이 **굵게 표시**되어 비교가 용이합니다.

### 그래프 내 마커 설명  
- 🔴 **빨간색 네모 박스** : 해당 연도에 *감소가 발생한 경우*  
- ⭐ **별표 마커** : *분석 기간 동안 단 한 번도 감소가 없었던* 시도/회사  

---
"""
)

# ---------------------------
# 필터 & 기본 선택값 계산
# ---------------------------
years_all = years_all_full.copy()

ALL_SIDOS = [
    "강원","경기","경남","경북","광주","대구","대전","부산","서울","울산",
    "인천","전남","전북","제주","충남","충북","세종"
]
DEFAULT_SIDOS = ["서울","대구","부산","대전","광주"]
sidos_in_data = [s for s in ALL_SIDOS if s in df["시도"].dropna().unique().tolist()]
default_sidos_in_data = [s for s in DEFAULT_SIDOS if s in sidos_in_data]

companies_all = sorted([c for c in df["회사"].dropna().unique().tolist()])
top6 = (df.groupby("회사")["수요가수"].sum(min_count=1)
        .sort_values(ascending=False).head(6).index.tolist())
if "대성" not in top6 and "대성" in companies_all:
    top6 = ["대성"] + [c for c in top6 if c != "대성"]
default_comps = [c for c in top6 if c in companies_all]

sel_years = st.sidebar.multiselect("연도", options=years_all, default=years_all)
sel_sidos = st.sidebar.multiselect("시도", options=sidos_in_data, default=default_sidos_in_data)
sel_comps = st.sidebar.multiselect("회사", options=companies_all, default=default_comps)

# 스케일: 기본값을 절대값, 라벨 단순화
scale_mode = st.sidebar.radio(
    "스케일",
    ["absolute", "yoy_pct"],
    index=0,
    format_func=lambda x: {"absolute":"절대값", "yoy_pct":"전년대비(%)"}[x]
)

# 강조 대상 멀티셀렉트
highlight_cities = st.sidebar.multiselect(
    "강조할 시도(복수 선택)",
    options=sidos_in_data,
    default=(["대구"] if "대구" in sidos_in_data else [])
)
highlight_companies = st.sidebar.multiselect(
    "강조할 회사명(복수 선택)",
    options=companies_all,
    default=(["대성"] if "대성" in companies_all else [])
)

# ---------------------------
# 요약(연도 필터만 반영: 시도/회사 선택은 무시)
# ---------------------------
sum_city = agg_city[agg_city["연도"].isin(sel_years)].copy()
sum_comp = agg_comp[agg_comp["연도"].isin(sel_years)].copy()

city_rate_dec,  city_rate_nondec  = dec_sets(sum_city, "시도", "보급률(%)")
comp_rate_dec,  comp_rate_nondec  = dec_sets(sum_comp, "회사", "보급률(%)")

city_sd_dec,    city_sd_nondec    = dec_sets(sum_city, "시도", "세대수")
comp_sd_dec,    comp_sd_nondec    = dec_sets(sum_comp, "회사", "세대수")

city_cst_dec,   city_cst_nondec   = dec_sets(sum_city, "시도", "수요가수")
comp_cst_dec,   comp_cst_nondec   = dec_sets(sum_comp, "회사", "수요가수")

st.subheader("요약 (연도 필터 반영)")
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("### 보급률(%)")
    st.markdown(f"- **감소한 지역** ({len(city_rate_dec)}): {fmt(city_rate_dec)}")
    st.markdown(f"- **감소 없는 지역** ({len(city_rate_nondec)}): {fmt(city_rate_nondec)}")
    st.markdown(f"- **감소한 회사** ({len(comp_rate_dec)}): {fmt(comp_rate_dec)}")
    st.markdown(f"- **감소 없는 회사** ({len(comp_rate_nondec)}): {fmt(comp_rate_nondec)}")
with c2:
    st.markdown("### 세대수")
    st.markdown(f"- **감소한 지역** ({len(city_sd_dec)}): {fmt(city_sd_dec)}")
    st.markdown(f"- **감소 없는 지역** ({len(city_sd_nondec)}): {fmt(city_sd_nondec)}")
    st.markdown(f"- **감소한 회사** ({len(comp_sd_dec)}): {fmt(comp_sd_dec)}")
    st.markdown(f"- **감소 없는 회사** ({len(comp_sd_nondec)}): {fmt(comp_sd_nondec)}")
with c3:
    st.markdown("### 수요가수")
    st.markdown(f"- **감소한 지역** ({len(city_cst_dec)}): {fmt(city_cst_dec)}")
    st.markdown(f"- **감소 없는 지역** ({len(city_cst_nondec)}): {fmt(city_cst_nondec)}")
    st.markdown(f"- **감소한 회사** ({len(comp_cst_dec)}): {fmt(comp_cst_dec)}")
    st.markdown(f"- **감소 없는 회사** ({len(comp_cst_nondec)}): {fmt(comp_cst_nondec)}")

st.markdown("---")

# ---------------------------
# 집계 & 필터 적용 (그래프/테이블용)
# ---------------------------
agg_city_all = agg_city.copy()
agg_comp_all = agg_comp.copy()

f_city = agg_city_all[agg_city_all["연도"].isin(sel_years) & agg_city_all["시도"].isin(sel_sidos)].copy()
f_comp = agg_comp_all[agg_comp_all["연도"].isin(sel_years) & agg_comp_all["회사"].isin(sel_comps)].copy()

# 전체 집계표(엑셀용) – 필터 미적용(전체)
city_table_all = add_deltas(agg_city_all.sort_values(["시도","연도"]), "시도")
comp_table_all = add_deltas(agg_comp_all.sort_values(["회사","연도"]), "회사")

# ---------------------------
# 상단: 보급률 (시도/회사)
# ---------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("시도별 보급률 추이")
    if f_city.empty:
        st.info("시도 데이터가 없습니다.")
    else:
        abs_df = f_city[["연도","시도","보급률(%)"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_city, "시도", "보급률(%)", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "시도", "보급률(%)", scale_mode)
        non_dec = non_decrease_groups(abs_df, "시도", "보급률(%)")

        fig1 = px.line(tr_df, x="연도", y="보급률(%)", color="시도", markers=True)
        highlight_traces(fig1, set(highlight_cities))
        apply_star_for_nondec(fig1, non_dec)
        add_group_markers(fig1, drops, "시도", "연도", "보급률(%)")
        fig1.update_layout(height=800, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="시도", hovermode="x unified",
                           margin=dict(l=40, r=20, t=40, b=40),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

with col2:
    st.subheader("회사별 보급률 추이")
    if f_comp.empty:
        st.info("회사 데이터가 없습니다.")
    else:
        abs_df = f_comp[["연도","회사","보급률(%)"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_comp, "회사", "보급률(%)", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "회사", "보급률(%)", scale_mode)
        non_dec = non_decrease_groups(abs_df, "회사", "보급률(%)")

        fig2 = px.line(tr_df, x="연도", y="보급률(%)", color="회사", markers=True)
        highlight_traces(fig2, set(highlight_companies))
        apply_star_for_nondec(fig2, non_dec)
        add_group_markers(fig2, drops, "회사", "연도", "보급률(%)")
        fig2.update_layout(height=800, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="회사", hovermode="x unified",
                           margin=dict(l=20, r=40, t=40, b=40),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

# ---------------------------
# 세대수/수요가수 4개 그래프
# ---------------------------
# 1행: 세대수
r1c1, r1c2 = st.columns(2, gap="large")
with r1c1:
    st.subheader("시도별 세대수")
    if f_city.empty:
        st.info("시도 데이터가 없습니다.")
    else:
        abs_df = f_city[["연도","시도","세대수"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_city, "시도", "세대수", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "시도", "세대수", scale_mode)
        non_dec = non_decrease_groups(abs_df, "시도", "세대수")

        fig3 = px.line(tr_df, x="연도", y="세대수", color="시도", markers=True)
        highlight_traces(fig3, set(highlight_cities))
        apply_star_for_nondec(fig3, non_dec)
        add_group_markers(fig3, drops, "시도", "연도", "세대수")
        fig3.update_layout(height=520, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="시도", hovermode="x unified",
                           margin=dict(l=40, r=20, t=30, b=30),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig3, use_container_width=True, theme="streamlit")

with r1c2:
    st.subheader("회사별 세대수")
    if f_comp.empty:
        st.info("회사 데이터가 없습니다.")
    else:
        abs_df = f_comp[["연도","회사","세대수"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_comp, "회사", "세대수", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "회사", "세대수", scale_mode)
        non_dec = non_decrease_groups(abs_df, "회사", "세대수")

        fig4 = px.line(tr_df, x="연도", y="세대수", color="회사", markers=True)
        highlight_traces(fig4, set(highlight_companies))
        apply_star_for_nondec(fig4, non_dec)
        add_group_markers(fig4, drops, "회사", "연도", "세대수")
        fig4.update_layout(height=520, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="회사", hovermode="x unified",
                           margin=dict(l=20, r=40, t=30, b=30),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig4, use_container_width=True, theme="streamlit")

# 2행: 수요가수
r2c1, r2c2 = st.columns(2, gap="large")
with r2c1:
    st.subheader("시도별 수요가수")
    if f_city.empty:
        st.info("시도 데이터가 없습니다.")
    else:
        abs_df = f_city[["연도","시도","수요가수"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_city, "시도", "수요가수", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "시도", "수요가수", scale_mode)
        non_dec = non_decrease_groups(abs_df, "시도", "수요가수")

        fig5 = px.line(tr_df, x="연도", y="수요가수", color="시도", markers=True)
        highlight_traces(fig5, set(highlight_cities))
        apply_star_for_nondec(fig5, non_dec)
        add_group_markers(fig5, drops, "시도", "연도", "수요가수")
        fig5.update_layout(height=520, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="시도", hovermode="x unified",
                           margin=dict(l=40, r=20, t=30, b=30),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig5, use_container_width=True, theme="streamlit")

with r2c2:
    st.subheader("회사별 수요가수")
    if f_comp.empty:
        st.info("회사 데이터가 없습니다.")
    else:
        abs_df = f_comp[["연도","회사","수요가수"]].copy()
        tr_df, y_label, y_layout = transform_for_plot(f_comp, "회사", "수요가수", scale_mode)
        drops = drops_for_mode(abs_df, tr_df, "회사", "수요가수", scale_mode)
        non_dec = non_decrease_groups(abs_df, "회사", "수요가수")

        fig6 = px.line(tr_df, x="연도", y="수요가수", color="회사", markers=True)
        highlight_traces(fig6, set(highlight_companies))
        apply_star_for_nondec(fig6, non_dec)
        add_group_markers(fig6, drops, "회사", "연도", "수요가수")
        fig6.update_layout(height=520, xaxis_title="연도", yaxis_title=y_label,
                           legend_title="회사", hovermode="x unified",
                           margin=dict(l=20, r=40, t=30, b=30),
                           legend=dict(groupclick="togglegroup"), **y_layout)
        st.plotly_chart(fig6, use_container_width=True, theme="streamlit")

# ---------------------------
# 하단 표 (증감 포함)
# ---------------------------
st.subheader("집계 데이터 (전년대비 증감 포함)")
city_table = add_deltas(f_city.sort_values(["시도","연도"]), "시도")
comp_table = add_deltas(f_comp.sort_values(["회사","연도"]), "회사")

# 화면 표시용 포맷
city_disp = format_for_display(
    city_table[["연도","시도","세대수","세대수증감","수요가수","수요가수증감","보급률(%)","보급률증감"]]
    .reset_index(drop=True)
)
comp_disp = format_for_display(
    comp_table[["연도","회사","세대수","세대수증감","수요가수","수요가수증감","보급률(%)","보급률증감"]]
    .reset_index(drop=True)
)

st.caption(f"시도 표 행수: {len(city_disp)}  |  회사 표 행수: {len(comp_disp)}")
tcol1, tcol2 = st.columns(2, gap="large")
with tcol1:
    st.markdown(f"**시도별** (강조: {', '.join(highlight_cities) if highlight_cities else '없음'} · ⭐=무감소)")
    st.dataframe(city_disp, use_container_width=True, height=360)
with tcol2:
    st.markdown(f"**회사별** (강조: {', '.join(highlight_companies) if highlight_companies else '없음'} · ⭐=무감소)")
    st.dataframe(comp_disp, use_container_width=True, height=360)

with st.sidebar.expander("⬇ 엑셀 다운로드 (3시트)", expanded=True):
    export_mode = st.radio(
        "엑셀 내보내기 범위",
        ["전체 데이터", "현재 필터 적용"],
        index=0,
        help="엑셀에는 3개 시트(원본tidy / 시도별표 / 회사별표)가 저장됩니다."
    )

    # --- ① 원본 tidy (전체 vs 필터) ---
    orig_df_all = (
        df[["연도","시도","회사","세대수","수요가수","보급률(%)"]]
        .sort_values(["연도","시도","회사"]).reset_index(drop=True)
    )
    # 필터 적용 시: 연도 AND (시도 OR 회사)로 필터링
    orig_df_filtered = orig_df_all[
        orig_df_all["연도"].isin(sel_years) &
        (orig_df_all["시도"].isin(sel_sidos) | orig_df_all["회사"].isin(sel_comps))
    ].reset_index(drop=True)

    # --- ②③ 시도/회사 표 (전체 vs 필터) ---
    if export_mode == "전체 데이터":
        xls_city = city_table_all
        xls_comp = comp_table_all
        xls_orig = orig_df_all
        export_name = "도시가스_보급률_전체데이터.xlsx"
        st.caption("엑셀에는 ‘전체 데이터’가 저장됩니다.")
    else:
        # 이미 화면용으로 만든 표를 사용 (필터 반영)
        xls_city = city_table.sort_values(["시도","연도"]).reset_index(drop=True)
        xls_comp = comp_table.sort_values(["회사","연도"]).reset_index(drop=True)
        xls_orig = orig_df_filtered
        export_name = "도시가스_보급률_필터데이터.xlsx"
        st.caption("엑셀에는 ‘현재 필터 적용 데이터’가 저장됩니다.")

    # --- 엑셀로 내보내기 ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        xls_orig.to_excel(writer, sheet_name="원본(tidy)", index=False)
        xls_city.to_excel(writer, sheet_name="시도별(표)", index=False)
        xls_comp.to_excel(writer, sheet_name="회사별(표)", index=False)

        wb  = writer.book
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_pct = wb.add_format({"num_format": "0.00"})

        # 시트별 숫자 서식 (헤더명 기준 자동 적용)
        def style_sheet(ws, data: pd.DataFrame):
            headers = list(data.columns)
            for idx, name in enumerate(headers):
                if name in ["세대수","세대수증감","수요가수","수요가수증감"]:
                    ws.set_column(idx, idx, 14, fmt_int)
                elif name in ["보급률(%)","보급률증감"]:
                    ws.set_column(idx, idx, 12, fmt_pct)
                elif name in ["연도","시도","회사"]:
                    ws.set_column(idx, idx, 12)

        style_sheet(writer.sheets["원본(tidy)"], xls_orig)
        style_sheet(writer.sheets["시도별(표)"], xls_city)
        style_sheet(writer.sheets["회사별(표)"], xls_comp)

    st.download_button(
        "엑셀 파일 다운로드",
        data=buffer.getvalue(),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
