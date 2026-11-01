"""
Uber Ireland - Customer Experience Dashboard
============================================
Streamlit + Plotly dashboard built on the Google Forms survey
"Uber Customer Service Survey - Ireland".

Data is read LIVE from the Google Sheet that collects the form responses
(CSV export endpoint), so every new response appears after a refresh.
If the sheet can't be reached, the app falls back to the bundled snapshot
(data/survey_snapshot.csv) or to a CSV you upload in the sidebar.

Run:  pip install -r requirements.txt
      streamlit run app.py
"""

from __future__ import annotations

import io
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import insights as ai

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SHEET_ID = "1y_O6u7R6xHhzdP1tkvs4hl0qliVaOu20WS4jB7Otjrk"
SHEET_GID = "1727233898"  # tab "Respuestas de formulario 1"
SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
)
SNAPSHOT = Path(__file__).parent / "data" / "survey_snapshot.csv"

# Corporate palette (black/white core + functional accents)
BLACK, WHITE = "#000000", "#FFFFFF"
GREY_900, GREY_600, GREY_300, GREY_100 = "#1F1F1F", "#545454", "#CBCBCB", "#EEEEEE"
BLUE, GREEN, RED, YELLOW = "#276EF1", "#05944F", "#E11900", "#FFC043"
PURPLE, ORANGE, BROWN = "#7356BF", "#FF6937", "#99644C"
CATEGORICAL = [BLACK, BLUE, GREEN, YELLOW, PURPLE, ORANGE, GREY_600, BROWN]
# 5-step diverging scale: most negative -> neutral -> most positive
DIVERGING_5 = ["#B3140A", "#F29C91", "#D6D6D6", "#7FCBA4", "#05944F"]

# ---------------------------------------------------------------------------
# SURVEY SCHEMA
# ---------------------------------------------------------------------------
ASPECTS = {
    "r_driver": "Driver friendliness & professionalism",
    "r_punctuality": "Driver punctuality",
    "r_app": "Ease of use of the app",
    "r_support": "Customer support response",
    "r_overall": "Overall experience",
}
RATING_SCALE = ["Very poor", "Poor", "Average", "Good", "Excellent"]
FREQ_SCALE = ["Never", "Rarely", "Sometimes", "Often", "Always"]
LIKELY_SCALE = ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"]
SATIS_SCALE = [
    "Very dissatisfied",
    "Dissatisfied",
    "Neither dissatisfied nor satisfied",
    "Satisfied",
    "Very satisfied",
]
AGE_ORDER = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
USAGE_ORDER = ["Daily", "Weekly", "Monthly", "Less than once a month", "Rarely", "Never"]
PROBLEM_OPTIONS = [
    "Driver behaviour or professionalism",
    "Poor communication with the driver",
    "Unsafe or uncomfortable driving",
    "Driver punctuality or route knowledge",
    "Difficulty contacting Uber customer support",
    "Slow or ineffective problem resolution",
    "Payment, fare or refund issues",
]

QUESTIONS = {
    "age": "Q1 · Age group",
    "residency": "Q2 · Residency status in Ireland",
    "usage": "Q3 · Uber usage frequency",
    "had_problem": "Q4 · Experienced a problem with a driver or support",
    "problem_types": "Q4b · Aspects with problems",
    "support_prompt": "Q6 · Support responds promptly & helpfully",
    "cx_loyalty": "Q7 · Positive CX → more likely to reuse / recommend",
    "poor_driver_sat": "Q8 · Satisfaction when driver behaves poorly",
    "trust_drop": "Q9 · Trust decrease when problem unresolved (1–10)",
    "competitor_better": "Q10 · Others provide better service",
    "competitor_name": "Q10b · Which service",
    "improvement": "Q11 · Suggested improvements",
}

SEGMENTS = {
    "Age group": "age",
    "Residency status": "residency",
    "Usage frequency": "usage",
    "Experienced a problem": "had_problem",
    "Thinks competitors are better": "competitor_better",
}

THEMES = {
    "Pricing & fares": r"pric|expens|fare|cheap|cost",
    "Driver courtesy": r"polite|friendl|gentle|courte|rude|manner",
    "Faster problem resolution": r"resolv|quick|fast|support|respon|refund",
    "App reliability": r"\bapp\b|crash|bug|maintenance|glitch",
    "Availability & coverage": r"availab|rural|coverage|wait",
    "Already satisfied": r"nothing|already good|fantastic|great|excellent|doing well",
    "General overhaul": r"^\s*everything\s*$",
}

# ---------------------------------------------------------------------------
# PAGE SETUP & STYLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Uber Ireland · CX Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stMetric, button, input, select, textarea {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
    .cx-hero {
        background: #000; color: #fff; border-radius: 14px;
        padding: 26px 30px; margin-bottom: 18px;
        display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:16px;
    }
    .cx-hero h1 {color:#fff; font-size: 1.9rem; font-weight: 800; margin:0; letter-spacing:-0.02em;}
    .cx-hero p {color:#CBCBCB; margin:6px 0 0 0; font-size:0.95rem;}
    .cx-pill {background:#1F1F1F; border:1px solid #333; color:#fff; border-radius:999px;
        padding:6px 14px; font-size:0.8rem; font-weight:600; white-space:nowrap;}
    .cx-pill.live {border-color:#05944F;}
    .cx-pill.live::before {content:'● '; color:#05944F;}
    .cx-pill.offline::before {content:'● '; color:#FFC043;}
    [data-testid="stMetric"] {
        background:#fff; border:1px solid #E2E2E2; border-radius:12px; padding:14px 16px;
    }
    [data-testid="stMetricLabel"] p {font-size:0.78rem !important; color:#545454; font-weight:600;
        text-transform:uppercase; letter-spacing:0.03em;}
    [data-testid="stMetricValue"] {font-weight:800; font-size:1.75rem;}
    .cx-card {background:#fff; border:1px solid #E2E2E2; border-radius:12px; padding:16px 18px; height:100%;}
    .cx-card h4 {margin:0 0 6px 0; font-size:0.95rem;}
    .cx-insight {background:#fff; border-left:4px solid #276EF1; border-radius:8px;
        padding:12px 16px; margin-bottom:10px; font-size:0.92rem;}
    .cx-insight.neg {border-left-color:#E11900;}
    .cx-insight.pos {border-left-color:#05944F;}
    .cx-insight.warn {border-left-color:#FFC043;}
    .cx-section {font-size:1.05rem; font-weight:700; margin:8px 0 2px 0;}
    .cx-sub {color:#545454; font-size:0.85rem; margin-bottom:6px;}
    [data-testid="stPlotlyChart"] {background:#fff; border:1px solid #E2E2E2; border-radius:14px;
        padding:6px 8px; box-shadow:0 1px 2px rgba(0,0,0,0.04);}
    .kpi-band {font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:#545454;
        font-weight:700; margin:4px 0 6px 2px;}
    .stTabs button[role="tab"] {background:#fff; border:1px solid #E2E2E2 !important; border-radius:999px;
        padding:6px 16px !important; margin-right:6px;}
    .stTabs button[role="tab"][aria-selected="true"] {background:#000 !important; border-color:#000 !important;}
    .stTabs button[role="tab"][aria-selected="true"] p {color:#fff !important;}
    .ai-card {display:flex; gap:14px; align-items:flex-start;}
    .ai-card .ring {flex:0 0 74px;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {background:#fff; border:1px solid #E2E2E2; border-radius:999px;
        padding:6px 16px; height:auto;}
    .stTabs [aria-selected="true"] {background:#000 !important; color:#fff !important; border-color:#000;}
    .stTabs [aria-selected="true"] p {color:#fff !important;}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
    .ai-head {background:linear-gradient(135deg,#000 0%,#1F1F1F 100%); color:#fff; border-radius:14px;
        padding:20px 24px; margin-bottom:14px;}
    .ai-head .lbl {font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:#7FB2FF; font-weight:700;}
    .ai-head .txt {font-size:1.12rem; font-weight:600; line-height:1.5; margin-top:6px;}
    .ai-head .conf {font-size:0.8rem; color:#CBCBCB; margin-top:8px;}
    .ai-col-h {font-weight:800; font-size:1rem; margin:4px 0 8px 0; display:flex; align-items:center; gap:8px;}
    .ai-card {background:#fff; border:1px solid #E2E2E2; border-radius:10px; padding:12px 14px; margin-bottom:10px;
        border-top:4px solid #CBCBCB;}
    .ai-card.hi {border-top-color:#05944F;} .ai-card.lo {border-top-color:#E11900;} .ai-card.wa {border-top-color:#FFC043;}
    .ai-card .t {font-weight:700; font-size:0.93rem; margin-bottom:4px;}
    .ai-card .b {font-size:0.87rem; color:#1F1F1F; line-height:1.45;}
    .ai-card .e {font-size:0.72rem; color:#757575; margin-top:6px;}
    .chip {display:inline-block; font-size:0.66rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;
        padding:2px 8px; border-radius:999px; background:#EEEEEE; color:#000; margin-bottom:6px;}
    .rec {background:#fff; border:1px solid #E2E2E2; border-radius:10px; padding:12px 14px; margin-bottom:8px;
        display:flex; gap:12px; align-items:flex-start;}
    .rec .p {font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:6px; white-space:nowrap; color:#fff;}
    .rec .p.High {background:#E11900;} .rec .p.Medium {background:#000;} .rec .p.Quick {background:#05944F;}
    .rec .a {font-weight:600; font-size:0.9rem;} .rec .w {font-size:0.8rem; color:#545454; margin-top:3px;}
    section[data-testid="stSidebar"] {background:#fff; border-right:1px solid #E2E2E2;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Shared Plotly look
PLOT_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", color=GREY_900, size=13),
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(bgcolor=BLACK, font_color=WHITE, font_family="Inter, sans-serif"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
    colorway=CATEGORICAL,
)


def style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=15), x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GREY_300)
    fig.update_yaxes(gridcolor=GREY_100, zeroline=False, linecolor=GREY_300)
    return fig


def show(fig: go.Figure, key: str, **kwargs):
    return st.plotly_chart(
        fig, width="stretch", key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------------------------------
def _match_column(cols: list[str], *needles: str) -> str | None:
    for c in cols:
        lc = c.lower()
        if all(n.lower() in lc for n in needles):
            return c
    return None


def _ordered(series: pd.Series, order: list[str]) -> pd.Categorical:
    """Ordered categorical that keeps any unexpected answer at the end."""
    extra = sorted(set(series.dropna()) - set(order))
    cats = [o for o in order if o in set(series.dropna())] + extra
    return pd.Categorical(series, categories=cats, ordered=True)


def parse_problems(text) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    found, rest = [], text
    for opt in sorted(PROBLEM_OPTIONS, key=len, reverse=True):
        if opt in rest:
            found.append(opt)
            rest = rest.replace(opt, "")
    leftovers = [t.strip() for t in rest.split(",") if t.strip()]
    if leftovers:
        found.append("Other")
    return found


def normalise_competitor(text) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = re.sub(r"\s+", "", text.lower())
    if t in {"n/a", "na", "none", "-"}:
        return None
    if "freenow" in t:
        return "FREENOW"
    if "bolt" in t:
        return "Bolt"
    if "didi" in t:
        return "DiDi"
    if "bus" in t or "train" in t or "luas" in t or "dart" in t:
        return "Public transport"
    if "uber" in t:
        return "Comment (no competitor named)"
    return text.strip().title()


def tag_themes(text) -> list[str]:
    if not isinstance(text, str) or not text.strip() or text.strip().lower() in {"n/a", "na", "-"}:
        return []
    tags = [name for name, rx in THEMES.items() if re.search(rx, text, flags=re.I)]
    if "Already satisfied" in tags and len(tags) > 1:
        tags.remove("Already satisfied")  # "already good" only if nothing else asked for
    return tags or ["Other"]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(raw.columns)
    m = {
        "timestamp": _match_column(cols, "marca temporal") or _match_column(cols, "timestamp"),
        "age": _match_column(cols, "age group"),
        "residency": _match_column(cols, "residency"),
        "usage": _match_column(cols, "how often do you use uber"),
        "had_problem": _match_column(cols, "have you ever experienced a problem"),
        "problem_types": _match_column(cols, "which aspects"),
        "r_driver": _match_column(cols, "[driver friendliness"),
        "r_punctuality": _match_column(cols, "[driver punctuality"),
        "r_app": _match_column(cols, "[ease of use"),
        "r_support": _match_column(cols, "[customer support response"),
        "r_overall": _match_column(cols, "[overall experience"),
        "support_prompt": _match_column(cols, "respond promptly"),
        "cx_loyalty": _match_column(cols, "positive customer service experience"),
        "poor_driver_sat": _match_column(cols, "behaves poorly"),
        "trust_drop": _match_column(cols, "trust in uber decrease"),
        "competitor_better": _match_column(cols, "other taxi"),
        "competitor_name": _match_column(cols, "which one"),
        "improvement": _match_column(cols, "what improvements"),
    }
    missing = [k for k, v in m.items() if v is None]
    df = pd.DataFrame({k: raw[v] if v else np.nan for k, v in m.items()})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string").str.strip().replace({"": pd.NA}).astype(object)
            df.loc[df[c].isna(), c] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "respondent", [f"R{i + 1:03d}" for i in range(len(df))])
    df["date"] = df["timestamp"].dt.date

    # Ordered categoricals
    df["age"] = _ordered(df["age"], AGE_ORDER)
    df["usage"] = _ordered(df["usage"], USAGE_ORDER)
    df["residency"] = _ordered(df["residency"], sorted(df["residency"].dropna().unique()))
    for c in ASPECTS:
        df[c] = _ordered(df[c], RATING_SCALE)
    df["support_prompt"] = _ordered(df["support_prompt"], FREQ_SCALE)
    df["cx_loyalty"] = _ordered(df["cx_loyalty"], LIKELY_SCALE)
    df["poor_driver_sat"] = _ordered(df["poor_driver_sat"], SATIS_SCALE)

    # Numeric scores (1-5) for averages and correlations
    for c in ASPECTS:
        df[c + "_n"] = df[c].astype(object).map({v: i + 1 for i, v in enumerate(RATING_SCALE)})
    df["support_prompt_n"] = df["support_prompt"].astype(object).map({v: i + 1 for i, v in enumerate(FREQ_SCALE)})
    df["cx_loyalty_n"] = df["cx_loyalty"].astype(object).map({v: i + 1 for i, v in enumerate(LIKELY_SCALE)})
    df["poor_driver_sat_n"] = df["poor_driver_sat"].astype(object).map({v: i + 1 for i, v in enumerate(SATIS_SCALE)})
    df["trust_drop"] = pd.to_numeric(df["trust_drop"], errors="coerce")

    # Composite CX Index: mean of the five Q5 ratings rescaled 0-100
    aspect_n = [c + "_n" for c in ASPECTS]
    df["cx_index"] = ((df[aspect_n].astype(float).mean(axis=1) - 1) / 4 * 100).round(1)
    df["trust_band"] = pd.cut(
        df["trust_drop"], bins=[0, 4, 7, 10],
        labels=["Low (1–4)", "Moderate (5–7)", "High (8–10)"],
    )

    # Multi-select & open text
    df["problem_list"] = df["problem_types"].apply(parse_problems)
    df["problem_count"] = df["problem_list"].str.len()
    df["competitor_clean"] = df["competitor_name"].apply(normalise_competitor)
    df["themes"] = df["improvement"].apply(tag_themes)
    df.attrs["missing_columns"] = missing
    return df


@st.cache_data(ttl=300, show_spinner="Loading survey responses…")
def load_live(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def load_data() -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.markdown("### Data source")
        source = st.radio(
            "Source", ["Live Google Sheet", "Upload CSV", "Bundled snapshot"],
            label_visibility="collapsed",
        )
        if source == "Live Google Sheet":
            if st.button("↻ Refresh responses", width="stretch"):
                load_live.clear()
            try:
                return clean(load_live(SHEET_CSV_URL)), "live"
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    "Couldn't reach the Google Sheet, showing the bundled snapshot. "
                    "Make sure the sheet is shared as *Anyone with the link → Viewer*.\n\n"
                    f"`{type(exc).__name__}`"
                )
                return clean(pd.read_csv(SNAPSHOT)), "offline"
        if source == "Upload CSV":
            up = st.file_uploader("Google Forms CSV export", type="csv")
            if up is not None:
                return clean(pd.read_csv(up)), "upload"
            st.info("Upload a CSV to continue. Showing the snapshot meanwhile.")
        return clean(pd.read_csv(SNAPSHOT)), "offline"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def share(series: pd.Series, values) -> float:
    s = series.dropna().astype(object)
    return float(s.isin(values).mean() * 100) if len(s) else float("nan")


def fmt_pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0f}%"


def counts(series: pd.Series, order: list[str] | None = None) -> pd.DataFrame:
    vc = series.astype(object).value_counts(dropna=True)
    if order:
        vc = vc.reindex([o for o in order if o in vc.index] + [i for i in vc.index if i not in order])
    out = vc.rename_axis("answer").reset_index(name="count")
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(1)
    return out


def explode_list(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    return df[["respondent", col]].explode(col).dropna().rename(columns={col: name})


def likert_diverging(df: pd.DataFrame, items: dict[str, str], scale, title: str,
                     height: int | None = None) -> go.Figure:
    """Centered (diverging) stacked bar for 5-point scales.
    `scale` is one list for all items, or a dict {column: scale} when items use different labels."""
    rows, item_scales = [], []
    for col, label in items.items():
        sc_ = scale[col] if isinstance(scale, dict) else scale
        item_scales.append(sc_)
        s = df[col].dropna().astype(object)
        n = len(s)
        cnt = s.value_counts().reindex(sc_, fill_value=0)
        pct = cnt / n * 100 if n else cnt * 0
        rows.append((f"{label}  <span style='color:{GREY_600}'>(n={n})</span>", cnt, pct))
    fig = go.Figure()
    labels = [r[0] for r in rows]
    starts = np.array([-(r[2].iloc[0] + r[2].iloc[1] + r[2].iloc[2] / 2) for r in rows])
    base = starts.copy()
    legend_scale = scale if isinstance(scale, list) else ["Most negative", "Negative", "Neutral", "Positive", "Most positive"]
    for i, level in enumerate(legend_scale):
        widths = np.array([r[2].iloc[i] for r in rows])
        cnts = [int(r[1].iloc[i]) for r in rows]
        lvl_names = [sc_[i] for sc_ in item_scales]
        fig.add_bar(
            y=labels, x=widths, base=base, orientation="h", name=level,
            marker=dict(color=DIVERGING_5[i], line=dict(color=WHITE, width=1)),
            customdata=[list(t) for t in zip(cnts, widths.tolist(), lvl_names)],
            text=[f"{w:.0f}%" if w >= 11 else "" for w in widths], textposition="inside",
            insidetextanchor="middle", textfont=dict(color=BLACK if i in (1, 2, 3) else WHITE, size=11),
            hovertemplate="<b>%{y}</b><br>%{customdata[2]}: %{customdata[0]} resp. (%{customdata[1]:.0f}%)<extra></extra>",
        )
        base = base + widths
    fig.add_vline(x=0, line_color=BLACK, line_width=1)
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                                                     yanchor="top", font=dict(size=11)),
                      margin=dict(l=10, r=10, t=50, b=50))
    fig.update_traces(textangle=0)
    fig.update_xaxes(ticksuffix="%", range=[-100, 100], tickvals=[-100, -50, 0, 50, 100],
                     ticktext=["100%", "50%", "0", "50%", "100%"])
    fig.update_yaxes(autorange="reversed", gridcolor=WHITE)
    return style(fig, title, height or 120 + 55 * len(items))


def bar_counts(c: pd.DataFrame, title: str, color=BLACK, horizontal=False, height=340,
               color_map: dict | None = None) -> go.Figure:
    colors = [color_map.get(a, color) for a in c["answer"]] if color_map else color
    if horizontal:
        fig = go.Figure(go.Bar(
            y=c["answer"], x=c["count"], orientation="h", marker_color=colors,
            text=[f"{n} · {p:.0f}%" for n, p in zip(c["count"], c["pct"])], textposition="outside",
            cliponaxis=False, hovertemplate="<b>%{y}</b><br>%{x} respondents<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", gridcolor=WHITE)
        fig.update_xaxes(showticklabels=False, range=[0, c["count"].max() * 1.35 if len(c) else 1])
    else:
        fig = go.Figure(go.Bar(
            x=c["answer"], y=c["count"], marker_color=colors,
            text=[f"{n}<br>{p:.0f}%" for n, p in zip(c["count"], c["pct"])], textposition="outside",
            cliponaxis=False, hovertemplate="<b>%{x}</b><br>%{y} respondents<extra></extra>",
        ))
        fig.update_yaxes(showticklabels=False, range=[0, c["count"].max() * 1.3 if len(c) else 1])
    return style(fig, title, height)


def donut(c: pd.DataFrame, title: str, colors=None, height=340) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=c["answer"], values=c["count"], hole=0.62, sort=False,
        marker=dict(colors=colors or CATEGORICAL, line=dict(color=WHITE, width=2)),
        texttemplate=[f"{p:.0f}%" if p >= 6 else "" for p in (c["count"] / max(c["count"].sum(), 1) * 100)],
        textposition="inside", insidetextorientation="horizontal", textfont=dict(size=12, color=WHITE),
        hovertemplate="<b>%{label}</b><br>%{value} respondents (%{percent})<extra></extra>",
        domain=dict(y=[0.0, 1.0]),
    ))
    fig.add_annotation(text=f"<b>{int(c['count'].sum())}</b><br><span style='font-size:11px;color:{GREY_600}'>responses</span>",
                       showarrow=False, font=dict(size=22), x=0.5, y=0.5, xref="paper", yref="paper")
    fig = style(fig, title, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top",
                                  font=dict(size=11)),
                      margin=dict(l=10, r=10, t=50, b=60))
    return fig


def insight(text: str, kind: str = ""):
    st.markdown(f"<div class='cx-insight {kind}'>{text}</div>", unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f"<div class='cx-section'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='cx-sub'>{sub}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD + FILTERS
# ---------------------------------------------------------------------------
df_all, status = load_data()

if df_all.attrs.get("missing_columns"):
    st.warning("Some survey columns weren't found and will show as empty: "
               + ", ".join(df_all.attrs["missing_columns"]))

with st.sidebar:
    st.markdown("### Filters")
    dmin, dmax = df_all["date"].min(), df_all["date"].max()
    date_range = st.date_input("Response date", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    def multiselect(label, col):
        opts = [o for o in df_all[col].cat.categories] if hasattr(df_all[col], "cat") \
            else sorted(df_all[col].dropna().unique())
        return st.multiselect(label, opts, default=[], placeholder="All")

    f_age = multiselect("Age group", "age")
    f_res = multiselect("Residency status", "residency")
    f_use = multiselect("Usage frequency", "usage")
    f_prob = st.multiselect("Experienced a problem (Q4)", ["Yes", "No"], placeholder="All")
    f_comp = st.multiselect("Thinks competitors are better (Q10)", ["Yes", "No"], placeholder="All")
    f_cx = st.slider("CX Index range (0–100)", 0, 100, (0, 100), step=5,
                     help="Composite of the five Q5 ratings, rescaled to 0–100.")
    st.divider()
    st.markdown("### Compare segments by")
    seg_label = st.selectbox("Segment", list(SEGMENTS), index=1, label_visibility="collapsed")
    seg = SEGMENTS[seg_label]

df = df_all.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
for col, vals in [("age", f_age), ("residency", f_res), ("usage", f_use),
                  ("had_problem", f_prob), ("competitor_better", f_comp)]:
    if vals:
        df = df[df[col].astype(object).isin(vals)]
df = df[df["cx_index"].between(*f_cx) | df["cx_index"].isna()]
N, N_ALL = len(df), len(df_all)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
pill = {"live": ("live", "Live · Google Sheet"), "offline": ("offline", "Snapshot data"),
        "upload": ("live", "Uploaded CSV")}[status]
last = df_all["timestamp"].max()
st.markdown(
    f"""
    <div class="cx-hero">
      <div>
        <h1>Uber Ireland · Customer Experience Dashboard</h1>
        <p>How customer service shapes rider satisfaction, loyalty and trust — survey of riders in Ireland</p>
      </div>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span class="cx-pill {pill[0]}">{pill[1]}</span>
        <span class="cx-pill">{N} of {N_ALL} responses in view</span>
        <span class="cx-pill">Last response {last:%d %b %Y, %H:%M}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if N == 0:
    st.error("No responses match the current filters. Widen the filters in the sidebar.")
    st.stop()
if N < 10:
    st.info(f"Only {N} responses match these filters — read percentages as directional.")

# ---------------------------------------------------------------------------
# KPIs (filtered vs all-responses delta)
# ---------------------------------------------------------------------------
def kpis(d: pd.DataFrame) -> dict:
    return {
        "cx": d["cx_index"].mean(),
        "overall_t2b": share(d["r_overall"], ["Good", "Excellent"]),
        "support_t2b": share(d["r_support"], ["Good", "Excellent"]),
        "problem": share(d["had_problem"], ["Yes"]),
        "prompt": share(d["support_prompt"], ["Often", "Always"]),
        "loyalty": share(d["cx_loyalty"], ["Likely", "Very likely"]),
        "trust": d["trust_drop"].mean(),
        "competitor": share(d["competitor_better"], ["Yes"]),
    }


k, k_all = kpis(df), kpis(df_all)
filtered = N != N_ALL


def delta(key, unit="pp", invert=False):
    if not filtered or pd.isna(k[key]) or pd.isna(k_all[key]):
        return None
    d = k[key] - k_all[key]
    return f"{d:+.1f} {unit} vs all"


def _delta_txt(key, unit="pp", invert=False) -> str:
    d = delta(key, unit)
    if not d:
        return "All responses"
    val = k[key] - k_all[key]
    good = (val < 0) if invert else (val > 0)
    colr = GREEN if good else (RED if val != 0 else GREY_600)
    return f"<span style='color:{colr}'>{'▲' if val > 0 else '▼' if val < 0 else '■'} {d}</span>"


def gauge_tile(value, title, maxv, bands, fmt, key, sub):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=0 if pd.isna(value) else value,
        number=dict(valueformat=fmt, font=dict(size=34, color=BLACK), suffix=f"<span style='font-size:14px;color:{GREY_600}'>/{maxv}</span>"),
        gauge=dict(
            axis=dict(range=[0, maxv], tickwidth=0, tickcolor=WHITE, tickfont=dict(size=10, color=GREY_600)),
            bar=dict(color=BLACK, thickness=0.28), bgcolor=WHITE, borderwidth=0,
            steps=[dict(range=r, color=c) for r, c in bands],
        ),
        domain=dict(x=[0, 1], y=[0.12, 0.95]),
    ))
    fig.add_annotation(x=0.5, y=0.02, text=sub, showarrow=False, font=dict(size=11, color=GREY_600))
    fig.update_layout(**PLOT_LAYOUT, height=210,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13), x=0.03, y=0.95))
    fig.update_layout(margin=dict(l=18, r=18, t=40, b=8))
    show(fig, key)


def ring_tile(value, title, color, key, sub):
    v = 0 if pd.isna(value) else float(value)
    fig = go.Figure(go.Pie(
        values=[v, max(100 - v, 0)], hole=0.74, sort=False, direction="clockwise", rotation=0,
        marker=dict(colors=[color, GREY_100], line=dict(width=0)), textinfo="none", hoverinfo="skip",
        domain=dict(x=[0.05, 0.95], y=[0.14, 0.9]),
    ))
    fig.add_annotation(x=0.5, y=0.52, text=f"<b>{v:.0f}%</b>", showarrow=False, font=dict(size=26, color=BLACK))
    fig.add_annotation(x=0.5, y=0.02, text=sub, showarrow=False, font=dict(size=11, color=GREY_600))
    fig.update_layout(**PLOT_LAYOUT, height=210, showlegend=False,
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13), x=0.03, y=0.95))
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    show(fig, key)


st.markdown("<div class='kpi-band'>Headline indicators</div>", unsafe_allow_html=True)
g = st.columns([1.25, 1, 1, 1, 1.25])
with g[0]:
    gauge_tile(k["cx"], "CX Index", 100, [([0, 45], "#FBE3E0"), ([45, 60], "#FFF1D1"), ([60, 100], "#DDF2E7")],
               ".0f", "kpi_cx", _delta_txt("cx", "pts"))
with g[1]:
    ring_tile(k["overall_t2b"], "Overall · Good+", GREEN, "kpi_overall", _delta_txt("overall_t2b"))
with g[2]:
    ring_tile(k["support_t2b"], "Support · Good+", BLUE, "kpi_support", _delta_txt("support_t2b"))
with g[3]:
    ring_tile(k["loyalty"], "CX drives loyalty", GREEN, "kpi_loyal", _delta_txt("loyalty"))
with g[4]:
    gauge_tile(k["trust"], "Trust erosion", 10, [([0, 4], "#DDF2E7"), ([4, 7], "#FFF1D1"), ([7, 10], "#FBE3E0")],
               ".1f", "kpi_trust", _delta_txt("trust", "pts", invert=True))
g2 = st.columns(4)
with g2[0]:
    ring_tile(k["problem"], "Problem incidence", RED, "kpi_prob", _delta_txt("problem", invert=True))
with g2[1]:
    ring_tile(k["prompt"], "Support prompt & helpful", BLUE, "kpi_prompt", _delta_txt("prompt"))
with g2[2]:
    ring_tile(k["competitor"], "See rivals as better", RED, "kpi_comp", _delta_txt("competitor", invert=True))
with g2[3]:
    pdf = share(df["poor_driver_sat"], ["Very dissatisfied", "Dissatisfied"])
    ring_tile(pdf, "Upset by poor driver", ORANGE, "kpi_poor", f"Q8 · n={df['poor_driver_sat'].notna().sum()}")

st.write("")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
report = ai.generate(df, ASPECTS, SEGMENTS)


def ring_svg(value, unit: str, color: str) -> str:
    if value is None or pd.isna(value):
        return ""
    frac = {"%": value / 100, "/5": value / 5, "/10": value / 10, "/100": value / 100,
            "pts": abs(value) / 50, "ρ": abs(value)}.get(unit, value / 100)
    frac = float(min(max(frac, 0), 1))
    label = {"%": f"{value:.0f}%", "/5": f"{value:.1f}", "/10": f"{value:.1f}", "/100": f"{value:.0f}",
             "pts": f"{value:+.0f}", "ρ": f"{value:.2f}"}.get(unit, f"{value:.0f}")
    small = {"/5": "of 5", "/10": "of 10", "/100": "of 100", "pts": "pts", "ρ": "ρ"}.get(unit, "")
    r, c = 30, 2 * 3.14159 * 30
    return (f"<svg width='74' height='74' viewBox='0 0 74 74'>"
            f"<circle cx='37' cy='37' r='{r}' fill='none' stroke='#EEEEEE' stroke-width='7'/>"
            f"<circle cx='37' cy='37' r='{r}' fill='none' stroke='{color}' stroke-width='7' stroke-linecap='round'"
            f" stroke-dasharray='{frac * c:.1f} {c:.1f}' transform='rotate(-90 37 37)'/>"
            f"<text x='37' y='{40 if not small else 37}' text-anchor='middle' font-size='15' font-weight='800' "
            f"font-family='Inter, sans-serif' fill='#000'>{label}</text>"
            + (f"<text x='37' y='50' text-anchor='middle' font-size='9' fill='#757575' "
               f"font-family='Inter, sans-serif'>{small}</text>" if small else "")
            + "</svg>")


def finding_card(f: "ai.Finding", cls: str):
    color = {"hi": GREEN, "lo": RED, "wa": "#E0A100"}[cls]
    st.markdown(
        f"<div class='ai-card {cls}'><div class='ring'>{ring_svg(f.value, f.unit, color)}</div><div>"
        f"<span class='chip'>{f.tag}</span><div class='t'>{f.title}</div>"
        f"<div class='b'>{f.text}</div><div class='e'>Evidence: {f.evidence}</div></div></div>",
        unsafe_allow_html=True,
    )


tabs = st.tabs([
    "Executive summary", "✦ AI insights", "Respondent profile", "Service ratings", "Problems & support",
    "Loyalty & trust", "Competition & voice of customer", "Response explorer",
])

# ============================ 1. EXECUTIVE SUMMARY ==========================
with tabs[0]:
    c1, c2 = st.columns([1.35, 1])
    with c1:
        sc = pd.DataFrame({
            "aspect": list(ASPECTS.values()),
            "avg": [df[c + "_n"].mean() for c in ASPECTS],
            "t2b": [share(df[c], ["Good", "Excellent"]) for c in ASPECTS],
            "b2b": [share(df[c], ["Very poor", "Poor"]) for c in ASPECTS],
        }).sort_values("avg")
        bar_colors = [RED if a < 3 else (YELLOW if a < 3.5 else GREEN) for a in sc["avg"]]
        fig = go.Figure(go.Bar(
            y=sc["aspect"], x=sc["avg"], orientation="h", marker_color=bar_colors,
            customdata=np.column_stack([sc["t2b"], sc["b2b"]]),
            text=[f"<b>{a:.2f}</b>  ·  {t:.0f}% good+" for a, t in zip(sc["avg"], sc["t2b"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average %{x:.2f} / 5<br>Good+Excellent %{customdata[0]:.0f}%"
                          "<br>Poor+Very poor %{customdata[1]:.0f}%<extra></extra>",
        ))
        fig.add_vline(x=3, line_dash="dot", line_color=GREY_600,
                      annotation_text="Average (3)", annotation_position="top")
        fig.update_xaxes(range=[1, 5.9], tickvals=[1, 2, 3, 4, 5],
                         ticktext=["Very poor", "Poor", "Average", "Good", "Excellent"])
        show(style(fig, "Service scorecard · average rating per aspect (Q5)", 360), "exec_score")
        # Funnel-style path: had problem -> support good -> trust not eroded
        stages = {
            "All respondents": N,
            "Experienced a problem": int((df["had_problem"] == "Yes").sum()),
            "…support rated Good+": int(((df["had_problem"] == "Yes")
                                         & df["r_support"].astype(object).isin(["Good", "Excellent"])).sum()),
            "…still Likely to reuse": int(((df["had_problem"] == "Yes")
                                           & df["r_support"].astype(object).isin(["Good", "Excellent"])
                                           & df["cx_loyalty"].astype(object).isin(["Likely", "Very likely"])).sum()),
        }
        fig = go.Figure(go.Funnel(
            y=list(stages), x=list(stages.values()), textinfo="value+percent initial",
            marker=dict(color=[BLACK, RED, BLUE, GREEN]),
            connector=dict(line=dict(color=GREY_300)),
        ))
        show(style(fig, "Service-recovery funnel", 300), "exec_funnel")
        st.caption("Of riders who hit a problem, how many rated support well — and stayed loyal.")

    with c2:
        section("What the data says", "Auto-interpreted from the responses in view · full analysis in ✦ AI insights")
        for f in report.lowlights[:2]:
            finding_card(f, "lo")
        for f in report.highlights[:2]:
            finding_card(f, "hi")

    st.write("")
    e1, e2, e3 = st.columns(3)
    with e1:
        show(donut(counts(df["had_problem"], ["Yes", "No"]), "Experienced a problem (Q4)",
                   colors=[RED, GREY_300], height=320), "exec_q4")
    with e2:
        c_ = counts(df["r_overall"], RATING_SCALE)
        show(donut(c_, "Overall experience rating (Q5)",
                   colors=[dict(zip(RATING_SCALE, DIVERGING_5)).get(x, GREY_300) for x in c_["answer"]],
                   height=320), "exec_overall")
    with e3:
        show(donut(counts(df["competitor_better"], ["Yes", "No"]), "Rivals offer better service? (Q10)",
                   colors=[RED, GREY_300], height=320), "exec_q10")
    st.write("")
    show(likert_diverging(df, {"cx_loyalty": "Positive CX → reuse / recommend (Q7)",
                               "poor_driver_sat": "Satisfaction despite poor driver (Q8)",
                               "support_prompt": "Support prompt & helpful (Q6)"},
                          {"cx_loyalty": LIKELY_SCALE, "poor_driver_sat": SATIS_SCALE,
                           "support_prompt": FREQ_SCALE},
                          "Loyalty, tolerance & support at a glance", 300), "exec_likert")
    st.caption("Each bar uses its own 5-point scale (negative ← neutral → positive). "
               "Hover for the exact answer labels.")

# ============================ AI INSIGHTS ===================================
def _secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:  # noqa: BLE001  (no secrets.toml)
        return os.getenv(name, "")


with tabs[1]:
    st.markdown(
        f"<div class='ai-head'><div class='lbl'>✦ AI interpretation · {N} responses in view</div>"
        f"<div class='txt'>{report.headline}</div>"
        f"<div class='conf'>Confidence: {report.confidence} Updates automatically with every filter.</div></div>",
        unsafe_allow_html=True,
    )

    h, l, w = st.columns(3)
    with h:
        st.markdown(f"<div class='ai-col-h'>▲ Highlights <span class='chip'>{len(report.highlights)}</span></div>",
                    unsafe_allow_html=True)
        for f in report.highlights:
            finding_card(f, "hi")
    with l:
        st.markdown(f"<div class='ai-col-h'>▼ Lowlights <span class='chip'>{len(report.lowlights)}</span></div>",
                    unsafe_allow_html=True)
        for f in report.lowlights:
            finding_card(f, "lo")
    with w:
        st.markdown(f"<div class='ai-col-h'>◆ Watch-outs & levers <span class='chip'>{len(report.watch)}</span></div>",
                    unsafe_allow_html=True)
        for f in report.watch:
            finding_card(f, "wa")

    st.write("")
    k1, k2 = st.columns([1.25, 1])
    with k1:
        d = report.drivers
        if d is not None and not d.empty:
            qcol = {"Fix first": RED, "Protect": GREEN, "Maintain": BLUE, "Monitor": GREY_600}
            fig = go.Figure()
            for qd, colr in qcol.items():
                sub = d[d["quadrant"] == qd]
                if len(sub):
                    fig.add_scatter(
                        x=sub["performance"], y=sub["importance"], mode="markers+text", name=qd,
                        text=sub["aspect"], textposition="top center",
                        marker=dict(size=22, color=colr, line=dict(color=WHITE, width=2)),
                        customdata=np.column_stack([sub["t2b"], sub["b2b"], sub["loyalty_link"]]),
                        hovertemplate="<b>%{text}</b><br>Avg rating %{x:.2f}/5<br>Link to overall ρ=%{y:.2f}"
                                      "<br>Link to loyalty ρ=%{customdata[2]:.2f}<br>Good+ %{customdata[0]:.0f}% · "
                                      "Poor- %{customdata[1]:.0f}%<extra></extra>",
                    )
            xm, ym = d["performance"].median(), d["importance"].median()
            fig.add_vline(x=xm, line_dash="dot", line_color=GREY_300)
            fig.add_hline(y=ym, line_dash="dot", line_color=GREY_300)
            pad_x = max(0.4, (d["performance"].max() - d["performance"].min()) * 0.35)
            pad_y = max(0.15, (d["importance"].max() - d["importance"].min()) * 0.35)
            fig.update_xaxes(title="Performance · average rating (1–5)",
                             range=[d["performance"].min() - pad_x, d["performance"].max() + pad_x])
            fig.update_yaxes(title="Importance · correlation with Overall experience",
                             range=[d["importance"].min() - pad_y, d["importance"].max() + pad_y])
            show(style(fig, "Key-driver matrix · where to act first", 470), "ai_kda")
            st.caption("Top-left = **Fix first** (matters a lot, performs below par). "
                       "Top-right = **Protect**. Quadrant lines are the medians.")
    with k2:
        st.markdown("<div class='ai-col-h'>➜ Recommended actions</div>", unsafe_allow_html=True)
        for r in report.recommendations:
            pcls = "Quick" if r.priority == "Quick win" else r.priority
            st.markdown(
                f"<div class='rec'><span class='p {pcls}'>{r.priority.upper()}</span><div>"
                f"<div class='a'>{r.action}</div><div class='w'>Why: {r.why}"
                f"{' · Owner: ' + r.owner if r.owner else ''}</div></div></div>",
                unsafe_allow_html=True,
            )

    # ---- Export ---------------------------------------------------------------
    md = [f"# Uber Ireland CX — AI insights\n\n_{N} responses · {report.confidence}_\n\n**{report.headline}**\n",
          "## Highlights"] + [f"- **{f.title}** — {f.text} _({f.evidence})_" for f in report.highlights] + \
         ["\n## Lowlights"] + [f"- **{f.title}** — {f.text} _({f.evidence})_" for f in report.lowlights] + \
         ["\n## Watch-outs & levers"] + [f"- **{f.title}** — {f.text} _({f.evidence})_" for f in report.watch] + \
         ["\n## Recommended actions"] + [f"- [{r.priority}] {r.action} — {r.why}" for r in report.recommendations]
    st.download_button("⬇ Download insights (Markdown)", "\n".join(md).encode("utf-8"),
                       "uber_cx_ai_insights.md", "text/markdown")

# ============================ 2. RESPONDENT PROFILE =========================
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        show(bar_counts(counts(df["age"], list(df["age"].cat.categories)), QUESTIONS["age"]), "p_age")
    with c2:
        show(donut(counts(df["residency"]), QUESTIONS["residency"]), "p_res")
    with c3:
        show(donut(counts(df["usage"], list(df["usage"].cat.categories)), QUESTIONS["usage"],
                   colors=[BLACK, BLUE, "#7FA8F5", GREY_300, GREY_600, YELLOW]), "p_use")

    c4, c5 = st.columns([1, 1.2])
    with c4:
        sb = df.assign(age=df["age"].astype(object), residency=df["residency"].astype(object),
                       usage=df["usage"].astype(object))
        sb = sb.groupby(["residency", "age", "usage"], dropna=True).size().reset_index(name="n")
        fig = px.sunburst(sb, path=["residency", "age", "usage"], values="n",
                          color="residency", color_discrete_sequence=CATEGORICAL)
        fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} respondents<br>"
                                        "%{percentParent:.0%} of parent<extra></extra>",
                          insidetextorientation="radial")
        show(style(fig, "Drill-down: residency → age → usage (click to zoom)", 460), "p_sun")
    with c5:
        ct = pd.crosstab(df["age"], df["residency"])
        fig = px.imshow(ct, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, WHITE], [1, BLUE]])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="", side="top")
        fig.update_yaxes(title="")
        show(style(fig, "Age × residency (respondent counts)", 280), "p_heat")

        daily = df.groupby("date").size().reset_index(name="responses")
        daily["cumulative"] = daily["responses"].cumsum()
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["responses"], name="Per day", marker_color=GREY_300)
        fig.add_scatter(x=daily["date"], y=daily["cumulative"], name="Cumulative", mode="lines+markers",
                        line=dict(color=BLACK, width=3), marker=dict(size=8))
        show(style(fig, "Response collection over time", 260), "p_time")

# ============================ 3. SERVICE RATINGS ============================
with tabs[3]:
    show(likert_diverging(df, ASPECTS, RATING_SCALE,
                          "Q5 · Rating distribution per service aspect (negative ← Average → positive)"),
         "r_likert")

    c1, c2 = st.columns([1, 1.15])
    with c1:
        segs = [s for s in df[seg].astype(object).dropna().unique()]
        fig = go.Figure()
        theta = list(ASPECTS.values()) + [list(ASPECTS.values())[0]]
        fig.add_scatterpolar(r=[df[c + "_n"].mean() for c in ASPECTS] + [df["r_driver_n"].mean()],
                             theta=theta, name="All in view", line=dict(color=BLACK, width=3, dash="dot"))
        for i, s in enumerate(sorted(segs, key=str)):
            sub = df[df[seg].astype(object) == s]
            vals = [sub[c + "_n"].mean() for c in ASPECTS]
            fig.add_scatterpolar(r=vals + [vals[0]], theta=theta, name=f"{s} (n={len(sub)})",
                                 line=dict(color=CATEGORICAL[(i + 1) % len(CATEGORICAL)], width=2),
                                 opacity=0.9)
        fig.update_polars(radialaxis=dict(range=[1, 5], tickvals=[1, 2, 3, 4, 5], gridcolor=GREY_100),
                          angularaxis=dict(gridcolor=GREY_100), bgcolor=WHITE)
        show(style(fig, f"Aspect profile by {seg_label.lower()} (avg 1–5)", 480), "r_radar")
        st.caption("Click legend items to isolate a segment.")
    with c2:
        hm = df.assign(_seg=df[seg].astype(object)).groupby("_seg")[[c + "_n" for c in ASPECTS]].mean()
        ns = df[seg].astype(object).value_counts()
        hm.index = [f"{i} (n={ns.get(i, 0)})" for i in hm.index]
        hm.columns = list(ASPECTS.values())
        fig = px.imshow(hm.round(2), text_auto=".2f", aspect="auto", zmin=1, zmax=5,
                        color_continuous_scale=[[0, RED], [0.5, "#F2F2F2"], [1, GREEN]])
        fig.update_xaxes(title="", side="top", tickangle=0)
        fig.update_yaxes(title="")
        fig.update_layout(coloraxis_colorbar=dict(title="Avg", thickness=10))
        show(style(fig, f"Heatmap · average rating by {seg_label.lower()}", 480), "r_heat")

    section("Drill into one aspect")
    a1, a2 = st.columns([1, 3])
    with a1:
        aspect_key = st.radio("Aspect", list(ASPECTS), format_func=ASPECTS.get, key="aspect_pick")
    with a2:
        ct = pd.crosstab(df[seg].astype(object), df[aspect_key], normalize="index") * 100
        ct = ct.reindex(columns=[c for c in RATING_SCALE if c in ct.columns])
        fig = go.Figure()
        for i, lvl in enumerate(RATING_SCALE):
            if lvl in ct.columns:
                fig.add_bar(y=ct.index, x=ct[lvl], name=lvl, orientation="h",
                            marker_color=DIVERGING_5[i],
                            text=[f"{v:.0f}%" if v >= 8 else "" for v in ct[lvl]], textposition="inside",
                            hovertemplate=f"<b>%{{y}}</b><br>{lvl}: %{{x:.0f}}%<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        show(style(fig, f"{ASPECTS[aspect_key]} · rating mix by {seg_label.lower()}",
                   140 + 45 * len(ct)), "r_drill")

# ============================ 4. PROBLEMS & SUPPORT =========================
with tabs[4]:
    c1, c2, c3 = st.columns([0.8, 1.4, 1])
    with c1:
        show(donut(counts(df["had_problem"], ["Yes", "No"]), QUESTIONS["had_problem"],
                   colors=[RED, GREY_300]), "pr_donut")
    with c2:
        probs = explode_list(df, "problem_list", "problem")
        pc = counts(probs["problem"]) if len(probs) else pd.DataFrame(columns=["answer", "count", "pct"])
        if len(pc):
            pc["pct"] = (pc["count"] / max(int((df["problem_count"] > 0).sum()), 1) * 100).round(1)
            driver_side = {o: BLACK for o in PROBLEM_OPTIONS[:4]}
            support_side = {o: RED for o in PROBLEM_OPTIONS[4:]}
            show(bar_counts(pc, "Q4b · Problem types (% of riders reporting problems)", horizontal=True,
                            color_map={**driver_side, **support_side, "Other": GREY_600}, height=360),
                 "pr_types")
            st.caption("■ Black = driver-side issues · ■ Red = support / payment issues. Multi-select question.")
        else:
            st.info("No problem types reported in this selection.")
    with c3:
        pcount = df.loc[df["problem_count"] > 0, "problem_count"].value_counts().sort_index()
        pc_df = pd.DataFrame({"answer": [f"{i} issue{'s' if i > 1 else ''}" for i in pcount.index],
                              "count": pcount.values})
        if len(pc_df):
            show(donut(pc_df, "Issues per affected rider", colors=[YELLOW, ORANGE, RED, "#8A0D05"], height=360),
                 "pr_count")

    c4, c5 = st.columns(2)
    with c4:
        # Co-occurrence matrix of problem types
        affected = df[df["problem_count"] > 1]["problem_list"]
        opts = sorted({p for lst in df["problem_list"] for p in lst})
        if len(opts) > 1:
            mat = pd.DataFrame(0, index=opts, columns=opts)
            for lst in df["problem_list"]:
                for p in lst:
                    mat.loc[p, p] += 1
                for a, b in combinations(sorted(set(lst)), 2):
                    mat.loc[a, b] += 1
                    mat.loc[b, a] += 1
            short = {o: (o[:28] + "…") if len(o) > 29 else o for o in opts}
            mat = mat.rename(index=short, columns=short)
            fig = px.imshow(mat, text_auto=True, color_continuous_scale=[[0, WHITE], [1, BLACK]], aspect="auto")
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="", tickangle=-35)
            fig.update_yaxes(title="")
            show(style(fig, "Problem co-occurrence (diagonal = total mentions)", 440), "pr_cooc")
    with c5:
        c_ = counts(df["support_prompt"], FREQ_SCALE)
        show(donut(c_, QUESTIONS["support_prompt"],
                   colors=[dict(zip(FREQ_SCALE, DIVERGING_5)).get(a, GREY_300) for a in c_["answer"]],
                   height=440), "pr_prompt")

    section("Did a problem change how riders rate Uber?",
            "Average Q5 rating: riders who reported a problem vs those who didn't")
    comp = df.groupby(df["had_problem"].astype(object))[[c + "_n" for c in ASPECTS]].mean().T
    comp.index = list(ASPECTS.values())
    fig = go.Figure()
    for grp, colr in [("No", GREY_600), ("Yes", RED)]:
        if grp in comp.columns:
            n_g = int((df["had_problem"] == grp).sum())
            fig.add_bar(x=comp.index, y=comp[grp], name=f"{'Had a problem' if grp == 'Yes' else 'No problem'} (n={n_g})",
                        marker_color=colr, text=comp[grp].round(2), textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 5.5], title="Avg rating (1–5)")
    show(style(fig, None, 360), "pr_compare")

    c6, c7 = st.columns(2)
    with c6:
        ct = pd.crosstab(df["support_prompt"], df["r_support"])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=[[0, WHITE], [1, BLUE]])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Q5 · Customer support rating")
        fig.update_yaxes(title="Q6 · Prompt & helpful")
        show(style(fig, "Q6 frequency × Q5 support rating (counts)", 380), "pr_q6q5")
    with c7:
        ct = pd.crosstab(df[seg].astype(object), df["had_problem"], normalize="index") * 100
        n_seg = df[seg].astype(object).value_counts()
        fig = go.Figure()
        if "Yes" in ct.columns:
            fig.add_bar(y=[f"{i} (n={n_seg.get(i, 0)})" for i in ct.index], x=ct["Yes"], orientation="h",
                        marker_color=RED, text=[f"{v:.0f}%" for v in ct["Yes"]], textposition="outside",
                        cliponaxis=False, name="Had a problem")
        fig.update_xaxes(range=[0, 110], ticksuffix="%")
        fig.update_layout(showlegend=False)
        show(style(fig, f"Problem incidence by {seg_label.lower()}", 380), "pr_seg")

# ============================ 5. LOYALTY & TRUST ============================
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        c_ = counts(df["cx_loyalty"], LIKELY_SCALE)
        show(donut(c_, QUESTIONS["cx_loyalty"],
                   colors=[dict(zip(LIKELY_SCALE, DIVERGING_5)).get(a, GREY_300) for a in c_["answer"]]), "l_q7")
    with c2:
        c_ = counts(df["poor_driver_sat"], SATIS_SCALE)
        show(donut(c_, QUESTIONS["poor_driver_sat"],
                   colors=[dict(zip(SATIS_SCALE, DIVERGING_5)).get(a, GREY_300) for a in c_["answer"]]), "l_q8")

    # Sankey: problem -> support rating -> loyalty
    def _band(v, good, bad):
        v = str(v)
        return "good" if v in good else ("bad" if v in bad else "mid")
    sk = df.dropna(subset=["had_problem", "r_support", "cx_loyalty"]).copy()
    if len(sk):
        sk["a"] = sk["had_problem"].map({"Yes": "Had a problem", "No": "No problem"})
        sk["b"] = sk["r_support"].astype(object).map(
            lambda v: {"good": "Support Good+", "mid": "Support Average", "bad": "Support Poor"}[
                _band(v, ["Good", "Excellent"], ["Poor", "Very poor"])])
        sk["c"] = sk["cx_loyalty"].astype(object).map(
            lambda v: {"good": "Would reuse / recommend", "mid": "Undecided", "bad": "Unlikely to reuse"}[
                _band(v, ["Likely", "Very likely"], ["Unlikely", "Very unlikely"])])
        nodes = ["Had a problem", "No problem", "Support Good+", "Support Average", "Support Poor",
                 "Would reuse / recommend", "Undecided", "Unlikely to reuse"]
        ncol = [RED, GREY_600, GREEN, GREY_300, RED, GREEN, GREY_300, RED]
        idx = {n: i for i, n in enumerate(nodes)}
        links = pd.concat([sk.groupby(["a", "b"]).size().reset_index(name="v").rename(columns={"a": "s", "b": "t"}),
                           sk.groupby(["b", "c"]).size().reset_index(name="v").rename(columns={"b": "s", "c": "t"})])
        lcol = {"Support Good+": "rgba(5,148,79,0.25)", "Support Poor": "rgba(225,25,0,0.25)",
                "Would reuse / recommend": "rgba(5,148,79,0.25)", "Unlikely to reuse": "rgba(225,25,0,0.25)"}
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(label=[f"{n} ({int(links.loc[links['t'] == n, 'v'].sum() or links.loc[links['s'] == n, 'v'].sum())})"
                             for n in nodes],
                      color=ncol, pad=22, thickness=18, line=dict(color=WHITE, width=1)),
            link=dict(source=[idx[x] for x in links["s"]], target=[idx[x] for x in links["t"]],
                      value=links["v"].tolist(), color=[lcol.get(t, "rgba(0,0,0,0.12)") for t in links["t"]],
                      hovertemplate="%{source.label} → %{target.label}: %{value} riders<extra></extra>"),
        ))
        show(style(fig, "Customer journey flow · problem → support rating → loyalty", 420), "l_sankey")
        st.caption("Follow the bands: how riders who had a problem rated support, and whether they would come back.")

    c3, c4 = st.columns([1.2, 1])
    with c3:
        tc = df["trust_drop"].value_counts().reindex(range(1, 11), fill_value=0)
        colors = [GREEN if i <= 4 else (YELLOW if i <= 7 else RED) for i in tc.index]
        fig = go.Figure(go.Bar(x=tc.index, y=tc.values, marker_color=colors, text=tc.values,
                               textposition="outside", cliponaxis=False,
                               hovertemplate="Score %{x}: %{y} respondents<extra></extra>"))
        m = df["trust_drop"].mean()
        fig.add_vline(x=m, line_dash="dash", line_color=BLACK,
                      annotation_text=f"Mean {m:.1f}", annotation_position="top left")
        fig.update_xaxes(tickvals=list(range(1, 11)), title="1 = no decrease · 10 = trust collapses")
        fig.update_yaxes(showticklabels=False, range=[0, tc.max() * 1.3 + 0.5])
        show(style(fig, QUESTIONS["trust_drop"], 380), "l_q9")
    with c4:
        bands = counts(df["trust_band"], ["Low (1–4)", "Moderate (5–7)", "High (8–10)"])
        show(donut(bands, "Trust-erosion bands", colors=[GREEN, YELLOW, RED], height=380), "l_band")

    c5, c6 = st.columns(2)
    with c5:
        bx = df.assign(_seg=df[seg].astype(object)).dropna(subset=["_seg", "trust_drop"])
        fig = px.box(bx, x="_seg", y="trust_drop", points="all", color="_seg",
                     color_discrete_sequence=CATEGORICAL, hover_data={"respondent": True, "_seg": False})
        fig.update_traces(marker=dict(size=8, opacity=0.75), jitter=0.4)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Trust decrease (1–10)", range=[0, 11])
        show(style(fig, f"Trust erosion by {seg_label.lower()}", 420), "l_box")
    with c6:
        num = {
            "Driver": "r_driver_n", "Punctuality": "r_punctuality_n", "App": "r_app_n",
            "Support": "r_support_n", "Overall": "r_overall_n", "Q6 Prompt": "support_prompt_n",
            "Q7 Loyalty": "cx_loyalty_n", "Q8 Tolerance": "poor_driver_sat_n", "Q9 Trust drop": "trust_drop",
        }
        corr = df[list(num.values())].astype(float).corr(method="spearman")
        corr.index = corr.columns = list(num)
        fig = px.imshow(corr.round(2), text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=[[0, RED], [0.5, WHITE], [1, BLUE]])
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(coloraxis_colorbar=dict(thickness=10))
        show(style(fig, "Relationships between measures (Spearman ρ)", 420), "l_corr")
        st.caption(f"Based on n={N}. With small samples treat |ρ| < 0.4 as weak / not reliable.")

    section("Does service quality move loyalty?",
            "Share Likely / Very likely to reuse or recommend (Q7), by Q5 overall-experience rating")
    lo = df.groupby(df["r_overall"].astype(object)).apply(
        lambda g: pd.Series({"loyal": share(g["cx_loyalty"], ["Likely", "Very likely"]), "n": len(g)}),
        include_groups=False,
    ).reindex([r for r in RATING_SCALE if r in df["r_overall"].astype(object).unique()])
    fig = go.Figure(go.Bar(x=lo.index, y=lo["loyal"], marker_color=DIVERGING_5[:len(lo)] if len(lo) == 5 else BLUE,
                           text=[f"{v:.0f}%<br><span style='font-size:10px'>n={int(n)}</span>"
                                 for v, n in zip(lo["loyal"], lo["n"])],
                           textposition="outside", cliponaxis=False))
    fig.update_yaxes(range=[0, 120], ticksuffix="%")
    show(style(fig, None, 320), "l_loyal")

# ===================== 6. COMPETITION & VOICE OF CUSTOMER ===================
with tabs[6]:
    c1, c2, c3 = st.columns([0.8, 1.1, 1.1])
    with c1:
        show(donut(counts(df["competitor_better"], ["Yes", "No"]), QUESTIONS["competitor_better"],
                   colors=[RED, GREY_300]), "c_donut")
    with c2:
        cc = df["competitor_clean"].dropna()
        if len(cc):
            cmap_ = {"FREENOW": RED, "Bolt": GREEN, "DiDi": ORANGE, "Public transport": BLUE}
            c_ = counts(cc)
            show(donut(c_, "Q10b · Services named as better",
                       colors=[cmap_.get(a, GREY_300) for a in c_["answer"]]), "c_names")
        else:
            st.info("No competitors named in this selection.")
    with c3:
        ct = pd.crosstab(df[seg].astype(object), df["competitor_better"], normalize="index") * 100
        n_seg = df[seg].astype(object).value_counts()
        fig = go.Figure()
        if "Yes" in ct.columns:
            fig.add_bar(y=[f"{i} (n={n_seg.get(i, 0)})" for i in ct.index], x=ct["Yes"], orientation="h",
                        marker_color=RED, text=[f"{v:.0f}%" for v in ct["Yes"]], textposition="outside",
                        cliponaxis=False)
        fig.update_xaxes(range=[0, 115], ticksuffix="%")
        show(style(fig, f"Think competitors are better · by {seg_label.lower()}", 340), "c_seg")

    with st.expander("How competitor names were normalised"):
        raw_map = (df[["competitor_name", "competitor_clean"]].dropna(subset=["competitor_name"])
                   .value_counts().reset_index(name="responses")
                   .rename(columns={"competitor_name": "Raw answer", "competitor_clean": "Grouped as"}))
        st.dataframe(raw_map, hide_index=True)

    st.write("")
    section("Voice of the customer · Q11 suggested improvements",
            "Open answers auto-tagged into themes (one answer can carry several themes)")
    t1, t2 = st.columns([1, 1.6])
    vo = df[df["improvement"].notna() & ~df["improvement"].astype(str).str.lower().isin(["n/a", "na", "-"])]
    with t1:
        th = explode_list(vo, "themes", "theme")
        if len(th):
            tcount = counts(th["theme"])
            theme_colors = {"Pricing & fares": BLACK, "Driver courtesy": BLUE,
                            "Faster problem resolution": RED, "App reliability": PURPLE,
                            "Availability & coverage": ORANGE, "Already satisfied": GREEN,
                            "General overhaul": GREY_600, "Other": GREY_300}
            fig = px.treemap(tcount, path=["answer"], values="count", color="answer",
                             color_discrete_map=theme_colors)
            fig.update_traces(texttemplate="<b>%{label}</b><br>%{value} mentions", textfont_size=14,
                              hovertemplate="<b>%{label}</b><br>%{value} mentions<extra></extra>",
                              marker=dict(line=dict(color=WHITE, width=3)), root_color=WHITE)
            show(style(fig, f"Improvement themes (n={len(vo)} comments)", 380), "c_themes")
        st.caption(f"{len(vo)} of {N} respondents left a suggestion.")
    with t2:
        q = st.text_input("Search comments", placeholder="e.g. price, polite, app…")
        theme_pick = st.multiselect("Filter by theme", list(THEMES) + ["Other"], placeholder="All themes")
        view = vo.copy()
        if q:
            view = view[view["improvement"].str.contains(q, case=False, na=False)]
        if theme_pick:
            view = view[view["themes"].apply(lambda t: bool(set(t) & set(theme_pick)))]
        st.dataframe(
            view.assign(themes=view["themes"].str.join(", "),
                        age=view["age"].astype(object), residency=view["residency"].astype(object),
                        overall=view["r_overall"].astype(object))
            [["respondent", "improvement", "themes", "age", "residency", "overall", "had_problem"]],
            hide_index=True, height=330,
            column_config={
                "respondent": st.column_config.TextColumn("ID", width="small"),
                "improvement": st.column_config.TextColumn("Comment", width="large"),
                "themes": "Themes", "age": "Age", "residency": "Residency",
                "overall": "Overall rating", "had_problem": "Had problem",
            },
        )

# ============================ 7. RESPONSE EXPLORER ==========================
with tabs[7]:
    section("Find individual riders",
            "Each dot is one respondent. Drag a box or click points to inspect them below.")
    rng = np.random.default_rng(7)
    sc = df.dropna(subset=["cx_index", "trust_drop"]).copy()
    sc["x"] = sc["cx_index"] + rng.uniform(-1.8, 1.8, len(sc))
    sc["y"] = sc["trust_drop"] + rng.uniform(-0.18, 0.18, len(sc))
    sc["colour_by"] = sc[seg].astype(object).fillna("n/a")
    fig = px.scatter(
        sc, x="x", y="y", color="colour_by", color_discrete_sequence=CATEGORICAL,
        custom_data=["respondent", "age", "residency", "usage", "r_overall", "cx_loyalty", "had_problem"],
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(color=WHITE, width=1.5), opacity=0.9),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[2]}<br>"
                      "Usage: %{customdata[3]}<br>Overall: %{customdata[4]}<br>"
                      "Q7: %{customdata[5]}<br>Had problem: %{customdata[6]}<extra></extra>",
    )
    fig.add_hline(y=7.5, line_dash="dot", line_color=GREY_300)
    fig.add_vline(x=50, line_dash="dot", line_color=GREY_300)
    for txt, x, y in [("At risk: low CX, fragile trust", 2, 10.6), ("Satisfied but fragile", 98, 10.6),
                      ("Low CX, resilient", 2, 0.4), ("Healthy", 98, 0.4)]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=GREY_600, size=11),
                           xanchor="left" if x < 50 else "right")
    fig.update_xaxes(title="CX Index (0–100, from Q5)", range=[-5, 105])
    fig.update_yaxes(title="Trust decrease if unresolved (Q9)", range=[0, 11])
    fig.update_layout(legend_title_text=seg_label, dragmode="select")
    event = show(style(fig, None, 480), "x_scatter", on_select="rerun", selection_mode=("points", "box"))

    picked = []
    try:
        picked = [p["customdata"][0] for p in event.selection.points]
    except Exception:  # noqa: BLE001
        picked = []

    detail_cols = {
        "respondent": "ID", "timestamp": "Submitted", "age": "Age", "residency": "Residency",
        "usage": "Usage", "had_problem": "Problem?", "problem_types": "Problem types",
        "r_driver": "Driver", "r_punctuality": "Punctuality", "r_app": "App", "r_support": "Support",
        "r_overall": "Overall", "cx_index": "CX Index", "support_prompt": "Q6 Prompt",
        "cx_loyalty": "Q7 Loyalty", "poor_driver_sat": "Q8 Poor-driver sat.", "trust_drop": "Q9 Trust drop",
        "competitor_better": "Q10 Competitor better", "competitor_name": "Which", "improvement": "Q11 Improvement",
    }
    table = df[list(detail_cols)].copy()
    for c in table.columns:
        if hasattr(table[c], "cat"):
            table[c] = table[c].astype(object)
    if picked:
        st.success(f"{len(picked)} respondent(s) selected: {', '.join(picked)}")
        table = table[table["respondent"].isin(picked)]

    st.dataframe(
        table.rename(columns=detail_cols), hide_index=True, height=420,
        column_config={
            "CX Index": st.column_config.ProgressColumn("CX Index", min_value=0, max_value=100, format="%.0f"),
            "Q9 Trust drop": st.column_config.ProgressColumn("Q9 Trust drop", min_value=0, max_value=10, format="%d"),
            "Submitted": st.column_config.DatetimeColumn("Submitted", format="DD MMM, HH:mm"),
        },
    )

    st.write("")
    d1, d2 = st.columns([1, 1])
    with d1:
        rid = st.selectbox("Respondent card", df["respondent"].tolist(),
                           index=df["respondent"].tolist().index(picked[0]) if picked else 0)
        r = df[df["respondent"] == rid].iloc[0]
        problems = ", ".join(r["problem_list"]) or "—"
        st.markdown(
            f"""<div class="cx-card">
            <h4>{rid} · {r['age']} · {r['residency']}</h4>
            <div class="cx-sub">Submitted {r['timestamp']:%d %b %Y %H:%M} · uses Uber: {r['usage']}</div>
            <b>CX Index:</b> {r['cx_index']:.0f}/100 &nbsp;|&nbsp; <b>Trust drop:</b> {r['trust_drop']:.0f}/10<br>
            <b>Ratings:</b> Driver {r['r_driver']} · Punctuality {r['r_punctuality']} · App {r['r_app']} ·
            Support {r['r_support']} · Overall {r['r_overall']}<br>
            <b>Problem:</b> {r['had_problem']} — {problems}<br>
            <b>Support prompt (Q6):</b> {r['support_prompt']} &nbsp;|&nbsp; <b>Loyalty (Q7):</b> {r['cx_loyalty']}<br>
            <b>Poor-driver satisfaction (Q8):</b> {r['poor_driver_sat']}<br>
            <b>Competitor better:</b> {r['competitor_better']} {('— ' + str(r['competitor_name'])) if pd.notna(r['competitor_name']) else ''}<br>
            <b>Suggestion:</b> <i>{r['improvement'] if pd.notna(r['improvement']) else '—'}</i>
            </div>""",
            unsafe_allow_html=True,
        )
    with d2:
        vals = [r[c + "_n"] for c in ASPECTS]
        avg = [df[c + "_n"].mean() for c in ASPECTS]
        theta = list(ASPECTS.values()) + [list(ASPECTS.values())[0]]
        fig = go.Figure()
        fig.add_scatterpolar(r=avg + [avg[0]], theta=theta, name="Average (in view)",
                             line=dict(color=GREY_600, dash="dot"))
        fig.add_scatterpolar(r=vals + [vals[0]], theta=theta, name=rid, fill="toself",
                             line=dict(color=BLUE, width=3), fillcolor="rgba(39,110,241,0.15)")
        fig.update_polars(radialaxis=dict(range=[1, 5], tickvals=[1, 2, 3, 4, 5], gridcolor=GREY_100),
                          angularaxis=dict(gridcolor=GREY_100), bgcolor=WHITE)
        show(style(fig, f"{rid} vs average", 360), "x_radar")

    # Downloads
    st.divider()
    export = df.drop(columns=["problem_list", "themes"]).assign(
        problem_list=df["problem_list"].str.join("; "), themes=df["themes"].str.join("; "))
    b1, b2 = st.columns(2)
    b1.download_button("⬇ Download filtered data (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "uber_cx_filtered.csv", "text/csv", width="stretch")
    buf = io.StringIO()
    summary = pd.DataFrame({
        "Metric": ["Responses", "CX Index", "Overall Good+ %", "Support Good+ %", "Problem incidence %",
                   "Support prompt (Often/Always) %", "CX drives loyalty %", "Trust erosion (avg /10)",
                   "Competitors seen as better %"],
        "Filtered": [N, k["cx"], k["overall_t2b"], k["support_t2b"], k["problem"], k["prompt"],
                     k["loyalty"], k["trust"], k["competitor"]],
        "All responses": [N_ALL, k_all["cx"], k_all["overall_t2b"], k_all["support_t2b"], k_all["problem"],
                          k_all["prompt"], k_all["loyalty"], k_all["trust"], k_all["competitor"]],
    }).round(1)
    summary.to_csv(buf, index=False)
    b2.download_button("⬇ Download KPI summary (CSV)", buf.getvalue().encode("utf-8"),
                       "uber_cx_kpis.csv", "text/csv", width="stretch")

st.markdown(
    f"<div style='color:{GREY_600}; font-size:0.78rem; margin-top:28px;'>"
    "Source: Google Forms · Uber Customer Service Survey – Ireland. CX Index = mean of the five Q5 ratings "
    "rescaled 0–100. Improvement themes are keyword-tagged and should be reviewed alongside the raw comments. "
    "Independent academic research; not affiliated with Uber.</div>",
    unsafe_allow_html=True,
)
