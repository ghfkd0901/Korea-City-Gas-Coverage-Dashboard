# app_race_like_youtube.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from itertools import cycle
from pathlib import Path

st.set_page_config(page_title="보급률 레이스 (YouTube 스타일)", layout="wide")

HERE = Path(__file__).resolve().parent
# pages/파일이면 루트는 한 단계 위, 루트에 out/가 있으면 그대로 HERE
ROOT = HERE if (HERE / "out").is_dir() else HERE.parent

CANDIDATES = [
    ROOT / "out"  / "보급률_tidy_(2006-2024).csv",
    ROOT / "data" / "보급률_tidy_(2006-2024).csv",
]
DEFAULT_CSV_PATH = next((p for p in CANDIDATES if p.is_file()), CANDIDATES[0])
DEFAULT_CSV = DEFAULT_CSV_PATH.as_posix()

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("CSV 인코딩을 열 수 없습니다. (utf-8-sig / cp949 / euc-kr 시도)")

def calc_agg(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(["연도", group_col], as_index=False)[["세대수", "수요가수"]].sum(min_count=1)
    sd = pd.to_numeric(g["세대수"], errors="coerce")
    cs = pd.to_numeric(g["수요가수"], errors="coerce")
    g["보급률(%)"] = np.where(sd > 0, (cs / sd) * 100.0, np.nan)
    return g.sort_values([group_col, "연도"]).reset_index(drop=True)

def build_color_map(names):
    palette = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab",
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    ]
    cyc = cycle(palette)
    return {n: next(cyc) for n in names}

def build_race_figure(
    dfin: pd.DataFrame,
    group_col: str,            # "시도" 또는 "회사"
    top_n: int,
    years: list[int],
    highlight: set[str],
    frame_ms: int,
    interp_steps: int,         # 연도 사이 중간 프레임 수
) -> go.Figure:

    all_names = dfin[group_col].dropna().unique().tolist()
    color_map = build_color_map(all_names)

    yearly = {}
    for y in years:
        sub = dfin[dfin["연도"] == y][[group_col, "보급률(%)"]].dropna().copy()
        sub = sub.sort_values("보급률(%)", ascending=False)
        if highlight:
            missing = [h for h in highlight if h in sub[group_col].values and h not in sub.head(top_n)[group_col].values]
            if missing:
                sub = pd.concat([sub[sub[group_col].isin(missing)], sub[~sub[group_col].isin(missing)]], axis=0)
                sub = sub.drop_duplicates(subset=[group_col], keep="first").sort_values("보급률(%)", ascending=False)
        yearly[y] = sub.head(top_n).copy()

    xmax = max([0.0] + [yearly[y]["보급률(%)"].max() for y in years if not yearly[y].empty])
    xmax = float(max(100.0, xmax * 1.10))

    first_year = years[0]
    base = yearly[first_year].sort_values("보급률(%)", ascending=False)
    y_labels = base[group_col].tolist()
    x_vals  = base["보급률(%)"].tolist()
    textlab = [f"{v:.1f}%" if pd.notna(v) else "" for v in x_vals]
    bar_colors = [color_map[n] for n in y_labels]

    fig = go.Figure(
        data=[go.Bar(
            x=x_vals[::-1],
            y=y_labels[::-1],
            orientation="h",
            text=textlab[::-1],
            textposition="outside",
            marker=dict(
                color=bar_colors[::-1],
                line=dict(width=[3 if (lbl in highlight) else 0 for lbl in y_labels[::-1]], color="#333")
            ),
            hovertemplate=f"{group_col}=%{{y}}<br>보급률=%{{x:.2f}}%<extra></extra>",
        )],
        layout=go.Layout(
            height=820,
            margin=dict(l=120, r=60, t=80, b=80),
            xaxis=dict(title="보급률(%)", range=[0, xmax], tickformat=".1f", gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title=group_col, categoryorder="array", categoryarray=y_labels[::-1]),
            bargap=0.15,
            plot_bgcolor="white",
            paper_bgcolor="white",
            title=dict(text=f"{group_col}별 보급률 — {first_year}", x=0.02, xanchor="left"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.02, x=1.0, xanchor="right",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                      "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ]
            )],
            annotations=[dict(
                text=str(first_year), x=0.95, y=0.08, xref="paper", yref="paper",
                showarrow=False, font=dict(size=96, color="rgba(0,0,0,0.12)"),
                xanchor="right", yanchor="bottom"
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[str(y)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}],
                            label=str(y)) for y in years],
                transition=dict(duration=0),
                x=0.02, len=0.96
            )],
        )
    )

    frames = []

    def _interp_frame(d0, d1, t):
        names = list(set(d0.index) | set(d1.index))
        v0 = d0.reindex(names).fillna(0.0).values
        v1 = d1.reindex(names).fillna(0.0).values
        v  = (1 - t) * v0 + t * v1
        out = pd.Series(v, index=names).sort_values(ascending=False).head(top_n)
        return out

    for i in range(len(years)):
        y_cur = years[i]
        d0 = yearly[y_cur].set_index(group_col)["보급률(%)"]

        if i == len(years) - 1:
            out = d0.sort_values(ascending=False)
            names = out.index.tolist()
            vals  = out.values.tolist()
            frames.append(go.Frame(
                name=str(y_cur),
                data=[go.Bar(
                    x=vals[::-1],
                    y=names[::-1],
                    orientation="h",
                    text=[f"{v:.1f}%" for v in vals[::-1]],
                    textposition="outside",
                    marker=dict(
                        color=[color_map[n] for n in names[::-1]],
                        line=dict(width=[3 if (lbl in highlight) else 0 for lbl in names[::-1]], color="#333")
                    ),
                    hovertemplate=f"{group_col}=%{{y}}<br>보급률=%{{x:.2f}}%<extra></extra>",
                )],
                layout=go.Layout(
                    yaxis=dict(categoryorder="array", categoryarray=names[::-1]),
                    xaxis=dict(range=[0, xmax]),
                    title=dict(text=f"{group_col}별 보급률 — {y_cur}", x=0.02, xanchor="left"),
                    annotations=[dict(
                        text=str(y_cur), x=0.95, y=0.08, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=96, color="rgba(0,0,0,0.12)"),
                        xanchor="right", yanchor="bottom"
                    )],
                )
            ))
            break

        y_nxt = years[i + 1]
        d1 = yearly[y_nxt].set_index(group_col)["보급률(%)"]

        out0 = d0.sort_values(ascending=False)
        names0 = out0.index.tolist()
        vals0  = out0.values.tolist()
        frames.append(go.Frame(
            name=str(y_cur),
            data=[go.Bar(
                x=vals0[::-1],
                y=names0[::-1],
                orientation="h",
                text=[f"{v:.1f}%" for v in vals0[::-1]],
                textposition="outside",
                marker=dict(
                    color=[color_map[n] for n in names0[::-1]],
                    line=dict(width=[3 if (lbl in highlight) else 0 for lbl in names0[::-1]], color="#333")
                ),
            )],
            layout=go.Layout(
                yaxis=dict(categoryorder="array", categoryarray=names0[::-1]),
                xaxis=dict(range=[0, xmax]),
                title=dict(text=f"{group_col}별 보급률 — {y_cur}", x=0.02, xanchor="left"),
                annotations=[dict(
                    text=str(y_cur), x=0.95, y=0.08, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=96, color="rgba(0,0,0,0.12)"),
                    xanchor="right", yanchor="bottom"
                )],
            )
        ))

        for k in range(1, interp_steps + 1):
            t = k / (interp_steps + 1)
            inter = _interp_frame(d0, d1, t)
            namesI = inter.index.tolist()
            valsI  = inter.values.tolist()
            frames.append(go.Frame(
                name=f"{y_cur}->{y_nxt}:{k}",
                data=[go.Bar(
                    x=valsI[::-1],
                    y=namesI[::-1],
                    orientation="h",
                    text=[f"{v:.1f}%" for v in valsI[::-1]],
                    textposition="outside",
                    marker=dict(
                        color=[color_map[n] for n in namesI[::-1]],
                        line=dict(width=[3 if (lbl in highlight) else 0 for lbl in namesI[::-1]], color="#333")
                    ),
                )],
                layout=go.Layout(
                    yaxis=dict(categoryorder="array", categoryarray=namesI[::-1]),
                    xaxis=dict(range=[0, xmax]),
                    title=dict(text=f"{group_col}별 보급률 — {y_cur}→{y_nxt}", x=0.02, xanchor="left"),
                    annotations=[dict(
                        text=f"{y_cur + t*(y_nxt - y_cur):.2f}".rstrip('0').rstrip('.'),
                        x=0.95, y=0.08, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=96, color="rgba(0,0,0,0.12)"),
                        xanchor="right", yanchor="bottom"
                    )],
                )
            ))

    fig.update(frames=frames)
    return fig

# ---------------------------
# 사이드바
# ---------------------------
st.sidebar.header("설정")
csv_path = st.sidebar.text_input("CSV 경로", value=DEFAULT_CSV)
if not os.path.isfile(csv_path):
    st.error("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    st.stop()

df = load_csv(csv_path)
for c in ["연도","세대수","수요가수","보급률"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df.rename(columns={"보급률":"보급률(%)"}, inplace=True)

tab_dim = st.sidebar.radio("대상", ["시도", "회사"], index=0)
years_all = sorted(df["연도"].dropna().astype(int).unique().tolist())

if tab_dim == "시도":
    agg = calc_agg(df, "시도")
    all_groups = sorted(agg["시도"].dropna().unique().tolist())
else:
    agg = calc_agg(df, "회사")
    all_groups = sorted(agg["회사"].dropna().unique().tolist())

sel_years  = st.sidebar.multiselect("연도", options=years_all, default=years_all)
sel_groups = st.sidebar.multiselect(f"{tab_dim}(레이스 포함)", options=all_groups, default=all_groups)

top_n     = st.sidebar.slider("연도별 상위 N", min_value=5, max_value=max(5, len(sel_groups)), value=min(12, max(5, len(sel_groups))))
frame_ms  = st.sidebar.slider("프레임 속도(ms)", min_value=200, max_value=2000, value=700, step=100)
interp    = st.sidebar.slider("부드러움(중간프레임 수)", min_value=0, max_value=6, value=2)
hi_default = ["대구"] if (tab_dim=="시도" and "대구" in sel_groups) else (["대성"] if (tab_dim=="회사" and "대성" in sel_groups) else [])
highlight = set(st.sidebar.multiselect("하이라이트", options=sel_groups, default=hi_default))

# ---------------------------
# 본문
# ---------------------------
st.title("보급률 바 차트 레이스")
st.caption("보급률 = (수요가수 합 / 세대수 합) × 100  ·  연도별 합계 기준으로 재계산")

sub = agg[(agg["연도"].isin(sel_years)) & (agg[tab_dim].isin(sel_groups))][["연도", tab_dim, "보급률(%)"]].copy()
if sub.empty:
    st.info("표시할 데이터가 없습니다. 사이드바 필터를 확인하세요.")
else:
    fig = build_race_figure(
        dfin=sub,
        group_col=tab_dim,
        top_n=min(top_n, len(sel_groups)),
        years=sorted(set(sel_years)),
        highlight=highlight,
        frame_ms=frame_ms,
        interp_steps=int(interp),
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
