"""
Kirana Pasal Failure Risk — thesis dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import kirana_analysis as ka

warnings.filterwarnings("ignore")

DATA_PATH = "KiranaPasal.xlsx"

# ------------------------------------------------------------- palette
#
# Colour carries one meaning throughout: red is a shop that failed or was
# flagged, teal is a shop that survived or was cleared. Brass marks a result
# that survived multiple-testing correction. Nothing else uses these three.

INK = "#2A211A"       # body text, structural bars
INK_SOFT = "#6B5D4F"  # secondary text
MUTED = "#9C8E7D"     # inactive, unresolved, chance lines
LINE = "#E0D6C4"      # hairlines and gridlines
PAPER = "#EFE9DD"     # page ground
CARD = "#FBF6EC"      # raised surfaces
SHELL = "#FDF9F1"     # top bar
ESPRESSO = "#241E1A"  # sidebar
FAIL = "#A63A2E"
SURVIVE = "#3F6B5C"
BRASS = "#C08A2E"
FAIL_TINT = "#F6DFDA"
BRASS_TINT = "#FBF3E2"

DISPLAY = "'Source Serif 4', Georgia, serif"
BODY = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"

st.set_page_config(
    page_title="Kirana Pasal Failure Risk",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------- version-safe kwargs
#
# Streamlit 1.49 replaced use_container_width with width="stretch". Guessing
# from the version string is unreliable (release candidates, forks, hosted
# builds), so ask the function itself which keyword it actually declares.
# Anything it does not declare falls into **kwargs and raises the "keyword
# arguments have been deprecated" banner above every chart.

import inspect


def _stretch(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    var_kw = inspect.Parameter.VAR_KEYWORD
    if "width" in params and params["width"].kind is not var_kw:
        return {"width": "stretch"}
    if "use_container_width" in params and params["use_container_width"].kind is not var_kw:
        return {"use_container_width": True}
    return {}


STRETCH = _stretch(st.dataframe)
CHART_STRETCH = _stretch(st.plotly_chart)
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}


# ---------------------------------------------------------------- style

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      .stApp {{ background: {PAPER}; }}
      html, body, [class*="css"] {{ font-family: {BODY}; color: {INK}; }}

      /* ---------------------------------------------------------------
         Text colour is forced here rather than left to the active theme.
         If Streamlit falls back to its dark theme (browser dark mode, a
         missing config.toml, a host that overrides it), every default
         label renders white on this cream page and disappears. The sidebar
         rules further down carry !important and win it back for the dark
         panel, so this stays a main-pane rule in effect.
         --------------------------------------------------------------- */
      .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp small,
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
      .stApp [data-testid="stWidgetLabel"] p,
      .stApp [data-testid="stExpander"] summary,
      .stApp [data-testid="stExpander"] summary p,
      .stApp [data-baseweb="select"] div,
      .stApp [data-baseweb="select"] span,
      .stApp [data-testid="stDataFrame"] *,
      .stApp [role="alert"] p {{ color: {INK}; }}

      /* secondary text keeps its softer tone */
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stCaptionContainer"] p,
      .stApp .note, .stApp .note *,
      .stApp [data-testid="stTickBar"],
      .stApp [data-testid="stTickBar"] div {{ color: {INK_SOFT}; }}

      .stApp [data-baseweb="select"] > div {{ background: {CARD};
                                              border-color: {LINE}; }}

      /* the shell: a thin lighter band across the top of the page */
      [data-testid="stHeader"] {{ background: {SHELL};
                                  border-bottom: 1px solid {LINE}; height: 2.6rem; }}
      .block-container {{ padding-top: 2.4rem; max-width: 1480px; }}

      h1, h2, h3, h4 {{ font-family: {DISPLAY}; color: {INK};
                        letter-spacing: -0.015em; font-weight: 600; }}
      [data-testid="stHeadingWithActionElements"] h2 {{
          font-size: 1.85rem; margin-bottom: 0.1rem; }}
      [data-testid="stHeadingWithActionElements"] h3 {{
          font-size: 1.15rem; font-weight: 600; margin-top: 2.3rem;
          padding-top: 1.1rem; border-top: 1px solid {LINE}; }}

      /* topbar ----------------------------------------------------- */
      .topbar {{ display: flex; justify-content: space-between; align-items: center;
                 font-size: 0.78rem; color: {INK_SOFT}; margin: -1.1rem 0 1.6rem 0; }}
      .topbar b {{ color: {BRASS}; font-weight: 600; }}
      .topbar .live {{ color: {SURVIVE}; font-weight: 500; }}
      .topbar .live::before {{ content: "\25CF"; margin-right: 0.35rem; font-size: 0.7rem; }}

      /* masthead --------------------------------------------------- */
      .eyebrow {{ font-family: {BODY}; font-size: 0.7rem; font-weight: 600;
                  letter-spacing: 0.16em; text-transform: uppercase;
                  color: {INK_SOFT}; margin-bottom: 0.4rem; }}
      .masthead h1 {{ font-size: 2.75rem; line-height: 1.05; margin: 0 0 0.45rem 0;
                      font-weight: 700; }}
      .masthead p {{ color: {INK_SOFT}; font-size: 0.97rem; max-width: 68ch;
                     margin: 0 0 1.4rem 0; }}
      .masthead p b {{ color: {INK}; }}

      /* callouts --------------------------------------------------- */
      .caveat {{ background: {BRASS_TINT}; border: 1px solid {BRASS};
                 border-radius: 8px; padding: 1rem 1.25rem; font-size: 0.9rem;
                 line-height: 1.6; color: {INK_SOFT}; margin-bottom: 1.8rem; }}
      .caveat b {{ color: {INK}; }}
      .finding {{ background: {FAIL_TINT}; border-left: 4px solid {FAIL};
                  border-radius: 0 8px 8px 0; padding: 1rem 1.25rem;
                  font-size: 1rem; line-height: 1.6; color: {INK_SOFT};
                  margin: 0.9rem 0 0.4rem 0; }}
      .finding b {{ color: {INK}; }}
      .note {{ color: {INK_SOFT}; font-size: 0.85rem; line-height: 1.6;
               display: block; max-width: 92ch; margin-top: 0.4rem; }}
      .chart-title {{ font-family: {DISPLAY}; font-size: 1.02rem; font-weight: 600;
                      color: {INK}; margin: 1rem 0 0.1rem 0; }}

      /* metric cards ----------------------------------------------- */
      [data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {LINE};
                                  border-radius: 8px;
                                  padding: 0.95rem 1.1rem 0.85rem 1.1rem; }}
      .stApp [data-testid="stMetricLabel"] p, .stApp [data-testid="stMetricLabel"] p span {{ font-size: 0.72rem; font-weight: 600;
                                         letter-spacing: 0.09em; text-transform: uppercase;
                                         color: {INK_SOFT}; }}
      .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] div {{ font-family: {DISPLAY}; font-size: 2rem;
                                       font-weight: 600; color: {INK};
                                       font-variant-numeric: tabular-nums; }}
      .stApp [data-testid="stMetricDelta"], .stApp [data-testid="stMetricDelta"] div {{ font-size: 0.76rem; color: {INK_SOFT}; }}
      [data-testid="stMetricDelta"] svg {{ fill: {MUTED}; }}

      /* tabs -------------------------------------------------------- */
      .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {LINE}; }}
      .stTabs [data-baseweb="tab"] {{ font-size: 0.93rem; font-weight: 500;
                                      color: {INK_SOFT}; padding: 0.4rem 0; }}
      .stTabs [aria-selected="true"] {{ color: {INK} !important; font-weight: 600; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background: {BRASS}; height: 2px; }}

      /* sidebar ----------------------------------------------------- */
      [data-testid="stSidebar"] {{ background: {ESPRESSO}; }}
      [data-testid="stSidebar"] *,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
          color: #EDE4D6 !important; }}
      [data-testid="stSidebar"] h3 {{ color: {BRASS} !important; font-family: {DISPLAY};
                                      font-weight: 700; }}
      [data-testid="stSidebar"] .note,
      [data-testid="stSidebar"] .note * {{ color: #A2968A !important; font-size: 0.78rem; }}
      [data-testid="stSidebar"] hr {{ border-color: #3B322B; }}
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
      [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {{
          background: #2E2721; border: 1px solid #3B322B; border-radius: 8px; }}
      [data-testid="stSidebar"] label p {{ font-size: 0.82rem; font-weight: 500; }}

      /* tables ------------------------------------------------------ */
      [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}
      [data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {LINE};
                                    border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- caching


@st.cache_data(show_spinner="Reading the survey workbook…")
def load(path_or_buffer):
    df = ka.build_targets(ka.load_raw(path_or_buffer))
    X, single, derived = ka.build_predictors(df)
    return df, X, single, derived


@st.cache_data(show_spinner="Running the chi-square screen…")
def screen(X, y):
    return ka.screen_predictors(X, y)


@st.cache_data(show_spinner="Fitting and tuning the models…")
def models(X, y):
    return ka.run_models(X, y)


# ------------------------------------------------------- chart plumbing


def style(fig, height=None):
    """One visual grammar for every figure on the page."""
    fig.update_layout(
        # A neutral base template. Without this the figure inherits plotly's
        # default styling underneath ours, which is where the stray axis
        # colours come from.
        template="none",
        title=None,
        height=height or fig.layout.height or 360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY, size=12.5, color=INK_SOFT),
        margin=dict(l=0, r=24, t=8, b=0),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(family=BODY, size=12, color=INK)),
        legend=dict(font=dict(size=12, color=INK_SOFT), bgcolor="rgba(0,0,0,0)"),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=True, gridcolor=LINE, gridwidth=1, zeroline=False,
                     linecolor=LINE, ticks="outside", tickcolor=LINE,
                     tickfont=dict(color=INK_SOFT, size=11.5),
                     title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=LINE,
                     tickfont=dict(color=INK_SOFT, size=11.5),
                     title_font=dict(color=MUTED, size=12))
    # Annotations added with add_vline/add_hline carry their own font.
    for ann in fig.layout.annotations:
        if ann.font is None or ann.font.color is None:
            ann.font = dict(family=BODY, size=10, color=INK_SOFT)
    return fig


def plot(fig, title=None, height=None, note=None):
    if title:
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    # theme=None stops Streamlit overwriting the figure with its own template.
    # When Streamlit is running dark, that template paints every tick label
    # white and they vanish against this page.
    st.plotly_chart(style(fig, height), theme=None, config=PLOTLY_CONFIG, **CHART_STRETCH)
    if note:
        st.markdown(f"<span class='note'>{note}</span>", unsafe_allow_html=True)


def bar(fig_data, xtitle=""):
    fig = go.Figure(fig_data)
    fig.update_layout(xaxis_title=xtitle)
    return fig


def note(text):
    st.markdown(f"<span class='note'>{text}</span>", unsafe_allow_html=True)


# -------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### Kirana Pasal")
    st.markdown(
        "<span class='note' style='margin-top:-0.6rem'>Failure risk dashboard · "
        "small grocery shops, Kathmandu Valley</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    upload = st.file_uploader("Upload Excel data (.xlsx)", type=["xlsx"])
    source = upload if upload is not None else DATA_PATH

    st.divider()
    st.markdown("**Step 1 — Choose the prediction target**")
    target_label = st.radio(
        "What should we predict?",
        ["Observed closure", "Owner expectation"],
        help="Choose recorded closure or the owner's expectation.",
    )
    predictor_label = "All predictors"
    st.caption("All available risk factors are used automatically.")

try:
    df, X_all, single_vars, derived_vars = load(source)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Upload the workbook in the sidebar.")
    st.stop()

target_col = "target_observed" if target_label == "Observed closure" else "target_belief"
y = df[target_col]

scr = screen(X_all, y)
screened_vars = scr.loc[scr["significant_raw"], "variable"].tolist()
X = X_all[screened_vars] if predictor_label == "Screened predictors" and screened_vars else X_all
model_out = models(X, y)

# ------------------------------------------------------------- masthead

st.markdown(
    f"""
    <div class='topbar'>
      <div><b>Kirana Pasal</b> &nbsp;/&nbsp; {len(df)} records · {target_label.lower()}</div>
      <div class='live'>Live</div>
    </div>
    <div class='masthead'>
      <div class='eyebrow'>Method demonstration</div>
      <h1>Predicting failure risk in kirana pasals</h1>
      <p>Reading currently: <b>{target_label.lower()}</b>, <b>{predictor_label.lower()}</b>.
      Change either in the sidebar and every figure below refits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "**Quick start:** Choose the target on the left → open **Risk Factors** → "
    "check **Models** → use **Threshold** to explore the warning level."
)

tabs = st.tabs(
    ["🏠 Home", "🔎 Risk Factors", "↔️ Compare", "🤖 Models", "🎚️ Threshold"]
)

# ------------------------------------------------------------ overview

with tabs[0]:
    st.header("🏠 Home")
    st.caption("Start here. See the dataset, outcomes, and main findings.")

    resolved = int(df["target_observed"].notna().sum())
    failures = int(df["target_observed"].sum())

    c = st.columns(4)
    c[0].metric("Total shops", f"{len(df)}")
    c[1].metric("Known outcomes", f"{resolved}", f"{len(df) - resolved} unverified",
                delta_color="off")
    c[2].metric("Failed shops", f"{failures}", f"{failures / resolved:.1%} of resolved",
                delta_color="off")
    c[3].metric("Owner-flagged risk", f"{int(df['target_belief'].sum())}",
                f"{df['target_belief'].mean():.1%} of all", delta_color="off")

    st.subheader("The width problem")
    c = st.columns(4)
    c[0].metric("Predictor variables", f"{X.shape[1]}")
    c[1].metric("After one-hot encoding", f"{model_out['n_encoded_cols']}")
    c[2].metric("Training rows", f"{model_out['n_train']}")
    c[3].metric("Failures in test set", f"{model_out['test_events']} of {model_out['n_test']}")
    st.markdown(
        f"<div class='finding'><b>{model_out['n_encoded_cols']} encoded columns against "
        f"{model_out['n_train']} training rows.</b> This ratio, more than any modelling choice, "
        f"explains why the cross-validated scores later in this dashboard do not survive the "
        f"held-out test set.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Where the shops are, and how they closed")
    left, right = st.columns([1, 1])

    with left:
        counts = df["District"].value_counts()
        plot(
            bar([go.Bar(x=counts.values, y=counts.index, orientation="h",
                        marker_color=INK, marker_line_width=0)], "Shops"),
            title="Shops by district", height=320,
        )

    with right:
        status = df["Outcome Status"].value_counts()
        colours = [
            FAIL if s in ka.FAILURE_STATUSES else (MUTED if s == ka.UNRESOLVED_STATUS else SURVIVE)
            for s in status.index
        ]
        plot(
            bar([go.Bar(x=status.values, y=status.index, orientation="h",
                        marker_color=colours, marker_line_width=0)], "Shops"),
            title="Recorded outcome status", height=320,
        )

    note(
        "Permanent closure, temporary closure and conversion to another business are pooled "
        "into a single failure category. They are materially different fates for an owner, and "
        "separating them would be a change worth making with a larger sample."
    )

    with st.expander("Distribution of any variable"):
        pick = st.selectbox("Variable", sorted(single_vars), key="ov_var")
        vc = df[pick].astype(str).value_counts()
        plot(
            bar([go.Bar(x=vc.values, y=vc.index, orientation="h",
                        marker_color=INK, marker_line_width=0)], "Shops"),
            title=pick, height=320,
        )

# -------------------------------------------------------- risk factors

with tabs[1]:
    st.header("🔎 Risk Factors")
    st.caption("See which shop factors are most strongly associated with the selected outcome.")

    n_raw = int(scr["significant_raw"].sum())
    n_bh = int(scr["survives_bh"].sum())
    c = st.columns(3)
    c[0].metric("Variables tested", f"{len(scr)}")
    c[1].metric("Significant at p < .05", f"{n_raw}")
    c[2].metric("Survive BH correction", f"{n_bh}", f"{n_raw - n_bh} lost", delta_color="off")

    sig = scr[scr["significant_raw"]].sort_values("cramers_v")
    if len(sig):
        colours = [BRASS if s else MUTED for s in sig["survives_bh"]]
        fig = bar(
            [
                go.Bar(
                    x=sig["cramers_v"], y=sig["variable"], orientation="h",
                    marker_color=colours, marker_line_width=0,
                    text=[f"{v:.3f}" for v in sig["cramers_v"]],
                    textposition="outside", textfont=dict(family=MONO, size=11, color=INK_SOFT),
                    cliponaxis=False,
                    hovertext=[
                        f"χ² = {r.chi2:.2f}, df = {r.dof}<br>p = {r.p:.2e}<br>q = {r.q:.3f}"
                        for r in sig.itertuples()
                    ],
                    hoverinfo="text",
                )
            ],
            "Cramér's V",
        )
        fig.add_vline(x=0.10, line_dash="dot", line_color=MUTED, line_width=1,
                      annotation_text="small", annotation_font=dict(size=10, color=MUTED))
        fig.add_vline(x=0.30, line_dash="dot", line_color=MUTED, line_width=1,
                      annotation_text="moderate", annotation_font=dict(size=10, color=MUTED))
        plot(
            fig,
            title="Variables associated with failure, ranked by effect size",
            height=max(320, 34 * len(sig)),
            note="Brass survives correction, grey does not. Effect sizes are small throughout, "
                 "which fits the framing that risk builds from a combination of pressures rather "
                 "than one decisive weakness.",
        )

    st.subheader("What correction removed")
    fig = go.Figure(
        go.Scatter(
            x=scr["p"], y=scr["q"], mode="markers",
            marker=dict(
                color=[BRASS if s else (INK if r else MUTED)
                       for s, r in zip(scr["survives_bh"], scr["significant_raw"])],
                size=10, opacity=0.85, line=dict(width=0),
            ),
            text=scr["variable"],
            hovertemplate="%{text}<br>p = %{x:.4f}<br>q = %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color=FAIL, line_width=1)
    fig.add_vline(x=0.05, line_dash="dash", line_color=FAIL, line_width=1)
    fig.update_layout(xaxis_title="raw p-value", yaxis_title="BH-adjusted q-value",
                      xaxis_type="log", yaxis_type="log")
    fig.update_yaxes(showgrid=True, gridcolor=LINE)
    plot(
        fig, height=420,
        note="Points in the lower-left quadrant hold up after correction. Points left of the "
             "vertical line but above the horizontal one looked significant and did not survive "
             "being one test among many.",
    )

    st.subheader("Inspect a variable")
    choice = st.selectbox("Variable", scr["variable"].tolist(), key="rf_var")
    row = scr[scr["variable"] == choice].iloc[0]

    sub = df.loc[y.notna()]
    s = X_all.loc[y.notna(), choice].astype(str)
    tab = pd.crosstab(s, y.loc[y.notna()].astype(int))
    tab.columns = ["Survived", "Failed"]
    rate = (tab["Failed"] / tab.sum(axis=1)).sort_values()

    left, right = st.columns([2, 1])
    with left:
        fig = bar(
            [go.Bar(x=rate.values, y=rate.index, orientation="h",
                    marker_color=FAIL, marker_line_width=0,
                    text=[f"{v:.1%}" for v in rate.values], textposition="outside",
                    textfont=dict(family=MONO, size=11, color=INK_SOFT), cliponaxis=False)],
            "Share of shops that failed",
        )
        base = float(y.loc[y.notna()].mean())
        fig.add_vline(x=base, line_dash="dash", line_color=INK, line_width=1,
                      annotation_text=f"base rate {base:.1%}",
                      annotation_font=dict(size=10, color=INK_SOFT))
        plot(fig, title=f"Failure rate by {choice}", height=max(300, 40 * len(rate)))
    with right:
        st.metric("Cramér's V", f"{row.cramers_v:.3f}")
        st.metric("p-value", f"{row.p:.4g}")
        st.metric("BH q-value", f"{row.q:.4g}")
        st.caption(f"{row.test}, χ² = {row.chi2:.2f}, df = {row.dof}")
        if row.low_cell_warning:
            st.warning("Over 20% of expected cell counts fall below five. Read with caution.")
        st.dataframe(tab, **STRETCH)

    with st.expander("Full screening table"):
        st.dataframe(
            scr[["variable", "test", "chi2", "dof", "p", "q", "cramers_v",
                 "significant_raw", "survives_bh"]].round(4),
            height=400, **STRETCH,
        )

# --------------------------------------------------------- two targets

with tabs[2]:
    st.header("↔️ Compare")
    st.caption("Compare recorded closure with what shop owners expected.")

    ag = ka.agreement(df)
    c = st.columns(4)
    c[0].metric("Agreement", f"{ag['agreement_rate']:.1%}")
    c[1].metric("Correlation (φ)", f"{ag['phi']:.3f}")
    c[2].metric("Chi-square p", f"{ag['p']:.3f}")
    c[3].metric("Failures not anticipated",
                f"{ag['failed_unanticipated']} of {ag['n_failures']}")

    st.markdown(
        f"<div class='finding'>{ag['failed_unanticipated']} of the {ag['n_failures']} shops that "
        f"failed had owners who did not expect to fail.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        t = ag["table"]
        z = t.values
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["Owner expected to continue", "Owner expected trouble"],
                y=["Survived", "Failed"],
                colorscale=[[0, CARD], [1, INK]],
                text=z, texttemplate="%{text}",
                textfont=dict(family=MONO, size=18), showscale=False,
                xgap=3, ygap=3,
            )
        )
        fig.update_xaxes(showgrid=False, showline=False, ticks="")
        plot(fig, title="Observed outcome against owner expectation", height=340)

    with right:
        scr_obs = screen(X_all, df["target_observed"])
        scr_bel = screen(X_all, df["target_belief"])
        top_obs = scr_obs[scr_obs["significant_raw"]].nlargest(6, "cramers_v")
        top_bel = scr_bel[scr_bel["significant_raw"]].nlargest(6, "cramers_v")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_obs["cramers_v"], y=top_obs["variable"], orientation="h",
                             name="Observed closure", marker_color=FAIL, marker_line_width=0))
        fig.add_trace(go.Bar(x=top_bel["cramers_v"], y=top_bel["variable"], orientation="h",
                             name="Owner expectation", marker_color=SURVIVE, marker_line_width=0))
        fig.update_layout(xaxis_title="Cramér's V", barmode="group",
                          legend=dict(orientation="h", y=-0.18, x=0))
        plot(fig, title="Each target selects different variables", height=340)

    note(
        "The observed outcome favours stock and working-capital measures; expectation favours "
        "debt and supplier measures. Modelling recorded closure makes one population visible, "
        "modelling expectation makes another, and that choice is normally made early in a "
        "project without being recorded as a decision at all."
    )

# ---------------------------------------------------- model comparison

with tabs[3]:
    st.header("🤖 Models")
    st.caption("Compare Logistic Regression, Random Forest, and XGBoost on the test data.")

    res = model_out["results"]
    names = list(res)
    rows = []
    for n in names:
        m = res[n]["at_050"]
        rows.append({
            "Model": n, "CV AUC": res[n]["cv_auc"], "Test AUC": res[n]["test_auc"],
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "Specificity": m["specificity"], "F1": m["f1"],
            "TN/FP/FN/TP": f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}",
        })
    table = pd.DataFrame(rows)
    st.dataframe(table.round(3), hide_index=True, **STRETCH)

    left, right = st.columns([1, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=[res[n]["at_050"]["accuracy"] for n in names],
                             name="Accuracy", marker_color=MUTED, marker_line_width=0))
        fig.add_trace(go.Bar(x=names, y=[res[n]["at_050"]["recall"] for n in names],
                             name="Recall", marker_color=FAIL, marker_line_width=0))
        fig.add_hline(
            y=model_out["majority_accuracy"], line_dash="dash", line_color=INK, line_width=1,
            annotation_text=f"predict-everyone-survives = {model_out['majority_accuracy']:.3f}",
            annotation_font=dict(size=10, color=INK_SOFT),
        )
        fig.update_layout(barmode="group", yaxis_range=[0, 1],
                          legend=dict(orientation="h", y=-0.14, x=0))
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        plot(
            fig, title="Accuracy is not the story; recall is", height=400,
            note="A model that called every shop safe would score the dashed line and identify "
                 "nobody at risk. Any accuracy figure near it is reporting the base rate, not a "
                 "working system.",
        )

    with right:
        fig = go.Figure()
        palette = [FAIL, SURVIVE, INK, BRASS]
        for i, n in enumerate(names):
            fpr, tpr = res[n]["roc"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{n} ({res[n]['test_auc']:.3f})",
                                     line=dict(color=palette[i % len(palette)], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(color=MUTED, dash="dash", width=1)))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="h", y=-0.18, x=0))
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        plot(fig, title="ROC curves on the held-out test set", height=400)

    st.subheader("Cross-validation against the test set")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[res[n]["cv_auc"] for n in names],
                         name="CV AUC (training folds)", marker_color=SURVIVE,
                         marker_line_width=0))
    fig.add_trace(go.Bar(x=names, y=[res[n]["test_auc"] for n in names], name="Test AUC",
                         marker_color=FAIL, marker_line_width=0))
    fig.add_hline(y=0.5, line_dash="dash", line_color=INK, line_width=1,
                  annotation_text="chance", annotation_font=dict(size=10, color=INK_SOFT))
    fig.update_layout(barmode="group", yaxis_range=[0, 1],
                      legend=dict(orientation="h", y=-0.14, x=0))
    fig.update_yaxes(showgrid=True, gridcolor=LINE)
    plot(fig, height=360)

    if predictor_label == "Screened predictors":
        st.warning(
            "The screen that selected these variables was computed on the full sample, so the "
            "test set contributed to the choice of predictors. These figures are optimistic by "
            "an unknown margin. The all-predictor results are the defensible ones."
        )

    with st.expander("Selected hyperparameters"):
        for n in names:
            st.write(f"**{n}** — {res[n]['best_params']}")

# ------------------------------------------------- threshold explorer

with tabs[4]:
    st.header("🎚️ Threshold")
    st.caption("Choose how sensitive the warning should be. Lower thresholds catch more risky shops but may flag more healthy shops.")

    ctrl = st.columns([1, 2])
    model_name = ctrl[0].selectbox("Model", list(model_out["results"]), key="thr_model")
    threshold = ctrl[1].slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    proba = model_out["results"][model_name]["proba"]
    y_test = model_out["y_test"]
    m = ka.metrics_at(y_test, proba, threshold)

    c = st.columns(4)
    c[0].metric("Shops warned in time", f"{m['tp']} of {m['tp'] + m['fn']}")
    c[1].metric("Healthy shops wrongly flagged", f"{m['fp']} of {m['tn'] + m['fp']}")
    c[2].metric("Failing shops missed", f"{m['fn']}")
    c[3].metric("Recall / Precision", f"{m['recall']:.2f} / {m['precision']:.2f}")

    left, right = st.columns([1, 1])

    with left:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig = go.Figure(
            go.Heatmap(
                z=z, x=["Predicted safe", "Predicted at risk"],
                y=["Actually survived", "Actually failed"],
                colorscale=[[0, CARD], [1, INK]], text=z, texttemplate="%{text}",
                textfont=dict(family=MONO, size=18), showscale=False, xgap=3, ygap=3,
            )
        )
        fig.update_xaxes(showgrid=False, showline=False, ticks="")
        plot(fig, title=f"{model_name} at threshold {threshold:.2f}", height=340)

    with right:
        grid = np.arange(0.05, 0.96, 0.01)
        sweep = [ka.metrics_at(y_test, proba, t) for t in grid]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=[s["recall"] for s in sweep], name="Recall",
                                 line=dict(color=FAIL, width=2)))
        fig.add_trace(go.Scatter(x=grid, y=[s["precision"] for s in sweep], name="Precision",
                                 line=dict(color=SURVIVE, width=2)))
        fig.add_trace(go.Scatter(x=grid, y=[s["fp"] / max(s["fp"] + s["tn"], 1) for s in sweep],
                                 name="Share of healthy shops flagged",
                                 line=dict(color=MUTED, width=2, dash="dot")))
        fig.add_vline(x=threshold, line_color=INK, line_width=1)
        fig.update_layout(xaxis_title="Threshold",
                          legend=dict(orientation="h", y=-0.18, x=0))
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        plot(fig, title="What every cut-off buys and costs", height=340)

    st.markdown(
        f"<div class='finding'>At this setting the model warns {m['tp']} of the "
        f"{m['tp'] + m['fn']} shops that failed and puts {m['fp']} shops that survived under "
        f"suspicion. If a flag triggered an offer of support, a false positive costs little. If "
        f"it fed a credit decision, the framework would be denying finance mostly to shops that "
        f"were going to survive.</div>",
        unsafe_allow_html=True,
    )