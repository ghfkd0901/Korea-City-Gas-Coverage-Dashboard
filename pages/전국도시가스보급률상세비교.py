# app_onepage.py
import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import colorsys

try:
    import matplotlib.colors as mcolors  # 네임드 컬러 처리용(없어도 동작)
except Exception:
    mcolors = None

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="도시가스 보급률 대시보드 - 방향 전환", layout="wide")

# 상대 경로
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "out").is_dir() else HERE.parent
DEFAULT_CSV = (ROOT / "out" / "보급률_tidy_(2006-2024).csv").as_posix()

# ---------------------------
# 유틸/집계
# ---------------------------
def calc_agg_city_company(df: pd.DataFrame) -> pd.DataFrame:
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
    g["회사-시도"] = g["회사"].astype(str) + " - " + g["시도"].astype(str)
    return g.sort_values(["시도", "회사", "연도"])

def calc_agg_company(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["연도", "회사"], as_index=False)[["세대수", "수요가수"]]
          .sum(min_count=1)
    )
    g["보급률(%)"] = np.where(
        (g["세대수"] > 0) & (~g["세대수"].isna()),
        (g["수요가수"] / g["세대수"]) * 100.0,
        np.nan
    )
    return g.sort_values(["회사", "연도"])

def calc_agg_city(df: pd.DataFrame) -> pd.DataFrame:
    """연도×시도 단위 집계(전국 합) 후 보급률 계산"""
    g = (
        df.groupby(["연도", "시도"], as_index=False)[["세대수", "수요가수"]]
          .sum(min_count=1)
    )
    g["보급률(%)"] = np.where(
        (g["세대수"] > 0) & (~g["세대수"].isna()),
        (g["수요가수"] / g["세대수"]) * 100.0,
        np.nan
    )
    return g.sort_values(["시도", "연도"])

def transform_for_plot(df: pd.DataFrame, group_col: str, value_col: str, scale_mode: str):
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
    t = df_abs[[group_col, "연도", value_col]].copy()
    t["prev"] = t.groupby(group_col)[value_col].shift(1)
    dec = t[(t["prev"].notna()) & (t[value_col] < t["prev"])]
    has_dec = set(dec[group_col].unique().tolist())
    all_g  = set(t[group_col].unique().tolist())
    return all_g - has_dec

def add_group_markers(fig: go.Figure, drops_df: pd.DataFrame,
                      group_col: str, x_col: str, y_col: str):
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
    for tr in fig.data:
        tr.update(line=dict(width=5 if tr.name in names else 2))

def apply_star_for_nondec(fig: go.Figure, nondec_set: set):
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
    t = df_all[[group_col, "연도", value_col]].dropna().copy()
    t["prev"] = t.groupby(group_col)[value_col].shift(1)
    dec = set(t[(t["prev"].notna()) & (t[value_col] < t["prev"])][group_col].unique())
    allg = set(t[group_col].unique())
    nondec = allg - dec
    return dec, nondec

def fmt(items: set) -> str:
    return ", ".join(sorted(items)) if items else "없음"

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
# 색상 유틸(안전 파서: hex/rgb/rgba/네임드 지원)
# ---------------------------
def _parse_color_to_rgb01(color_str: str):
    """
    다양한 포맷(#rgb, #rrggbb, rgb(), rgba(), 네임드 컬러)을 0~1 범위 (r,g,b)로 변환.
    """
    if not isinstance(color_str, str):
        raise ValueError(f"Unsupported color type: {type(color_str)}")

    s = color_str.strip().lower()

    # 1) #rgb / #rrggbb
    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 3:  # #abc
            r = int(s[0]*2, 16); g = int(s[1]*2, 16); b = int(s[2]*2, 16)
        elif len(s) == 6:  # #aabbcc
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        else:
            raise ValueError(f"Invalid hex color: #{s}")
        return (r/255.0, g/255.0, b/255.0)

    # 2) rgb()/rgba()
    if s.startswith("rgb"):
        m = re.match(r"rgba?\(([^)]+)\)", s)
        if not m:
            raise ValueError(f"Invalid rgb/rgba color: {color_str}")
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) < 3:
            raise ValueError(f"Invalid rgb/rgba components: {color_str}")

        def _to_255(x):
            x = x.strip()
            if x.endswith("%"):
                return max(0, min(255, round(float(x[:-1]) * 2.55)))
            return max(0, min(255, int(float(x))))

        r = _to_255(parts[0]); g = _to_255(parts[1]); b = _to_255(parts[2])
        return (r/255.0, g/255.0, b/255.0)

    # 3) 네임드 컬러 (matplotlib)
    if mcolors is not None:
        try:
            r, g, b = mcolors.to_rgb(s)  # 0~1
            return (r, g, b)
        except Exception:
            pass

    # 4) 실패
    raise ValueError(f"Unsupported color format: {color_str}")

def _rgb01_to_hex(rgb01):
    r, g, b = rgb01
    r = max(0, min(255, int(round(r*255))))
    g = max(0, min(255, int(round(g*255))))
    b = max(0, min(255, int(round(b*255))))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def adjust_lightness_any(color_in: str, factor: float) -> str:
    """
    color_in(임의 포맷)을 받아 명도(HLS의 L)를 factor만큼 조정한 HEX 반환.
    factor: 음수=어둡게, 양수=밝게 (권장 -0.3 ~ +0.3)
    """
    r, g, b = _parse_color_to_rgb01(color_in)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex((r2, g2, b2))

def build_color_map_family(df_pairs: pd.DataFrame, base_key: str, var_key: str, legend_key: str, var_list: list):
    """
    동일 base_key(기준 차원)는 같은 계열색 유지, var_key(다른 차원)에 따라 명도만 변화.
    legend_key: 범례 레이블 컬럼 (예: '회사-시도' 또는 '시도-회사')
    """
    base_palette = (
        px.colors.qualitative.D3
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Safe
    )

    bases = sorted(df_pairs[base_key].dropna().unique().tolist())

    # base 팔레트 → HEX로 정규화
    base_color = {}
    for i, b in enumerate(bases):
        raw_c = base_palette[i % len(base_palette)]
        base_color[b] = adjust_lightness_any(raw_c, 0.0)  # 정규화 겸 HEX화

    # 명도 스텝
    light_steps = [0.0, -0.12, +0.12, -0.24, +0.24, -0.32, +0.32]
    var_light = {v: light_steps[i % len(light_steps)] for i, v in enumerate(var_list)}

    color_map = {}
    for _, row in df_pairs[[base_key, var_key, legend_key]].drop_duplicates().iterrows():
        base_hex = base_color.get(row[base_key], "#888888")
        factor  = var_light.get(row[var_key], 0.0)
        color_map[row[legend_key]] = adjust_lightness_any(base_hex, factor)
    return color_map

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

# 기간/집계
years_all_full = sorted(df["연도"].dropna().unique().tolist())
period_text = f"{int(years_all_full[0])}년 ~ {int(years_all_full[-1])}년" if years_all_full else ""

agg_pair = calc_agg_city_company(df)
agg_comp  = calc_agg_company(df)
agg_city  = calc_agg_city(df)

# ---------------------------
# 헤더
# ---------------------------
st.title("보급률 추이 대시보드 (방향 전환)")
st.markdown(
    f"""
라디오 버튼으로 **분류 방향**을 전환할 수 있습니다.  
- **회사 → 시도**: 좌=회사별(전국), 우=회사-시도  
- **시도 → 회사**: 좌=시도별(전국), 우=시도-회사  
분석 기간: **{period_text}**  
---
"""
)

# ---------------------------
# 방향 선택 + 통합 필터
# ---------------------------
direction = st.sidebar.radio(
    "분류 방향",
    ["회사 → 시도", "시도 → 회사"],
    index=0,
)

years_all = years_all_full.copy()
sel_years = st.sidebar.multiselect("연도", options=years_all, default=years_all)

companies_all = sorted(df["회사"].dropna().unique().tolist())
sidos_all     = sorted(df["시도"].dropna().unique().tolist())

if direction == "회사 → 시도":
    # 회사 멀티
    top6 = (
        df.groupby("회사")["수요가수"].sum(min_count=1)
          .sort_values(ascending=False).head(6).index.tolist()
    )
    if "대성" not in top6 and "대성" in companies_all:
        top6 = ["대성"] + [c for c in top6 if c != "대성"]
    default_primary = [c for c in top6 if c in companies_all] or companies_all[:6]
    sel_primary = st.sidebar.multiselect("회사 (공통)", options=companies_all, default=default_primary)

    # 좌 데이터(회사), 우 데이터(회사-시도)
    left_df = agg_comp[(agg_comp["연도"].isin(sel_years)) & (agg_comp["회사"].isin(sel_primary))].copy()
    right_df = agg_pair[(agg_pair["연도"].isin(sel_years)) & (agg_pair["회사"].isin(sel_primary))].copy()
    right_df["회사-시도"] = right_df["회사"].astype(str) + " - " + right_df["시도"].astype(str)

    # 색상: 같은 시도 계열, 회사 명도
    color_map = build_color_map_family(
        df_pairs=right_df, base_key="시도", var_key="회사",
        legend_key="회사-시도", var_list=sel_primary
    )
    legend_title_right = "회사 - 시도"
    group_key_left = "회사"
    legend_title_left = "회사"
    hi_left  = {"대성"} if "대성" in left_df.get("회사", pd.Series([], dtype=str)).unique().tolist() else set()
    hi_right = {name for name in right_df["회사-시도"].unique().tolist() if name.startswith("대성 - ")}

else:  # "시도 → 회사"
    # 시도 멀티
    default_primary = [s for s in ["대구","경북","서울","부산"] if s in sidos_all] or sidos_all[:6]
    sel_primary = st.sidebar.multiselect("시도 (공통)", options=sidos_all, default=default_primary)

    # 좌 데이터(시도), 우 데이터(시도-회사)
    left_df = agg_city[(agg_city["연도"].isin(sel_years)) & (agg_city["시도"].isin(sel_primary))].copy()
    right_df = agg_pair[(agg_pair["연도"].isin(sel_years)) & (agg_pair["시도"].isin(sel_primary))].copy()
    right_df["시도-회사"] = right_df["시도"].astype(str) + " - " + right_df["회사"].astype(str)

    # 색상: 같은 회사 계열, 시도 명도
    color_map = build_color_map_family(
        df_pairs=right_df, base_key="회사", var_key="시도",
        legend_key="시도-회사", var_list=sel_primary
    )
    legend_title_right = "시도 - 회사"
    group_key_left = "시도"
    legend_title_left = "시도"
    hi_left  = {s for s in ["대구","경북"] if s in left_df.get("시도", pd.Series([], dtype=str)).unique().tolist()}
    hi_right = {name for name in right_df.get("시도-회사", pd.Series([], dtype=str)).unique().tolist()
                if name.startswith("대구 - ") or name.startswith("경북 - ")}

# 공통 스케일
scale_mode = st.sidebar.radio(
    "스케일",
    ["absolute", "yoy_pct"],
    index=0,
    format_func=lambda x: {"absolute":"절대값", "yoy_pct":"전년대비(%)"}[x]
)

# ---------------------------
# 요약 (오른쪽 그래프 기준)
# ---------------------------
if direction == "회사 → 시도":
    right_for_summary = right_df.rename(columns={"회사-시도":"그룹"})
else:
    right_for_summary = right_df.rename(columns={"시도-회사":"그룹"})

pair_rate_dec, pair_rate_nondec = dec_sets(right_for_summary, "그룹", "보급률(%)")
st.subheader("요약 (오른쪽 그래프의 그룹 기준, 공통 필터 반영)")
st.markdown(f"- **감소한 그룹** ({len(pair_rate_dec)}): {fmt(pair_rate_dec)}")
st.markdown(f"- **감소 없는 그룹** ({len(pair_rate_nondec)}): {fmt(pair_rate_nondec)}")
st.markdown("---")

col_left, col_right = st.columns(2, gap="large")

# ---------------------------
# 좌측 그래프
# ---------------------------
with col_left:
    st.subheader(f"{legend_title_left}별 보급률 추이 (좌)")
    if left_df.empty:
        st.info("선택된 조건에서 데이터가 없습니다.")
    else:
        abs_df_l = left_df[["연도", group_key_left, "보급률(%)"]].copy()
        tr_df_l, y_label_l, y_layout_l = transform_for_plot(left_df, group_key_left, "보급률(%)", scale_mode)
        drops_l = drops_for_mode(abs_df_l, tr_df_l, group_key_left, "보급률(%)", scale_mode)
        non_dec_l = non_decrease_groups(abs_df_l, group_key_left, "보급률(%)")

        fig_l = px.line(tr_df_l, x="연도", y="보급률(%)", color=group_key_left, markers=True)
        highlight_traces(fig_l, hi_left)
        apply_star_for_nondec(fig_l, non_dec_l)
        add_group_markers(fig_l, drops_l, group_key_left, "연도", "보급률(%)")
        fig_l.update_layout(
            height=820, xaxis_title="연도", yaxis_title=y_label_l,
            legend_title=legend_title_left, hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(groupclick="togglegroup"), **y_layout_l
        )
        st.plotly_chart(fig_l, use_container_width=True, theme="streamlit")

# ---------------------------
# 우측 그래프
# ---------------------------
with col_right:
    st.subheader(f"{legend_title_right} 보급률 추이 (우)")
    if right_df.empty:
        st.info("선택된 조건에서 데이터가 없습니다.")
    else:
        if direction == "회사 → 시도":
            legend_col = "회사-시도"
        else:
            legend_col = "시도-회사"

        abs_df_r = right_df[["연도", legend_col, "보급률(%)"]].copy()
        tr_df_r, y_label_r, y_layout_r = transform_for_plot(right_df, legend_col, "보급률(%)", scale_mode)
        drops_r = drops_for_mode(abs_df_r, tr_df_r, legend_col, "보급률(%)", scale_mode)
        non_dec_r = non_decrease_groups(abs_df_r, legend_col, "보급률(%)")

        fig_r = px.line(
            tr_df_r, x="연도", y="보급률(%)", color=legend_col, markers=True,
            color_discrete_map=color_map
        )
        highlight_traces(fig_r, hi_right)
        apply_star_for_nondec(fig_r, non_dec_r)
        add_group_markers(fig_r, drops_r, legend_col, "연도", "보급률(%)")
        fig_r.update_layout(
            height=820, xaxis_title="연도", yaxis_title=y_label_r,
            legend_title=legend_title_right, hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(groupclick="togglegroup"), **y_layout_r
        )
        st.plotly_chart(fig_r, use_container_width=True, theme="streamlit")

# ---------------------------
# 하단 표 / 엑셀
# ---------------------------
st.subheader("집계 데이터 (전년대비 증감 포함)")
if direction == "회사 → 시도":
    f_pair_table = agg_pair[(agg_pair["연도"].isin(sel_years)) & (agg_pair["회사"].isin(sel_primary))].copy()
else:
    f_pair_table = agg_pair[(agg_pair["연도"].isin(sel_years)) & (agg_pair["시도"].isin(sel_primary))].copy()

pair_table = add_deltas(
    f_pair_table.sort_values(["시도","회사","연도"]).copy(), "시도-회사"
)
pair_disp = format_for_display(
    pair_table[["연도","시도","회사","시도-회사","세대수","세대수증감","수요가수","수요가수증감","보급률(%)","보급률증감"]]
      .reset_index(drop=True)
)
st.caption(f"표 행수: {len(pair_disp)}")
st.dataframe(pair_disp, use_container_width=True, height=420)

with st.sidebar.expander("⬇ 엑셀 다운로드", expanded=True):
    export_mode = st.radio(
        "엑셀 내보내기 범위",
        ["전체 데이터", "현재 필터 적용"],
        index=1,
        help="엑셀에는 2개 시트(원본tidy / 시도-회사표)가 저장됩니다."
    )

    orig_df_all = (
        df[["연도","시도","회사","세대수","수요가수","보급률(%)"]]
          .sort_values(["연도","시도","회사"]).reset_index(drop=True)
    )

    if export_mode == "전체 데이터":
        xls_pair = add_deltas(
            agg_pair.sort_values(["시도","회사","연도"]).copy(), "시도-회사"
        ).reset_index(drop=True)
        xls_orig = orig_df_all
        export_name = "도시가스_보급률_시도회사_전체.xlsx"
        st.caption("엑셀에는 ‘전체 데이터’가 저장됩니다.")
    else:
        xls_pair = add_deltas(
            f_pair_table.sort_values(["시도","회사","연도"]).copy(), "시도-회사"
        ).reset_index(drop=True)
        if direction == "회사 → 시도":
            mask = (orig_df_all["연도"].isin(sel_years)) & (orig_df_all["회사"].isin(sel_primary))
        else:
            mask = (orig_df_all["연도"].isin(sel_years)) & (orig_df_all["시도"].isin(sel_primary))
        xls_orig = orig_df_all[mask].reset_index(drop=True)
        export_name = "도시가스_보급률_시도회사_현재표시.xlsx"
        st.caption("엑셀에는 ‘현재 연도·분류(회사/시도) 필터’가 반영됩니다.")

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
