"""
Kirana Pasal Failure Risk — thesis dashboard.

Run with:  streamlit run app.py

Requires the revised kirana_analysis.py (uncorrected chi-square, NaN-safe
belief target, training-only category pooling, repeated splits).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import kirana_analysis as ka

# Scoped rather than blanket. A convergence failure or a divide-by-zero is
# information; "ignore" everything hid exactly the warnings worth reading.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_PATH = "KiranaPasal.xlsx"

# Set False for the submitted version if the examiners want a plain register.
TAB_ICONS = True

SMALL_N = 10  # below this, a category percentage is not worth reading

# Set False for the public deployment. A hosted app inviting uploads of shop
# financials is the purpose-drift risk this study names in its own ethical
# evaluation; the control belongs in a local run, not on an open URL.
ALLOW_UPLOAD = False

# ---------------------------------------------------- reported specification
#
# The configuration the thesis tables were produced under. The dashboard's own
# defaults differ deliberately — pooling applied to the models, repeats at ten,
# no variable pre-selection — because each is the better analysis. This constant
# exists so the reported figures stay one click away and any divergence is named
# on screen rather than left for a reader to discover.

REPORTED_SPEC = {
    "target_label": "Observed closure",
    "unresolved_fail": False,
    "merge_rare": False,   # Table 25.1 is 266 unpooled encoded columns
    "yates": False,
    "n_repeats": 1,        # every reported figure is the seed-42 split
}

SPEC_LABELS = {
    "target_label": "prediction target",
    "unresolved_fail": "treatment of unverified shops",
    "merge_rare": "rare-category pooling",
    "yates": "Yates' correction",
    "n_repeats": "number of splits",
}


def _apply_reported_spec():
    """Restore the configuration the thesis tables were produced under."""
    for key, value in REPORTED_SPEC.items():
        st.session_state[key] = value


def _spec_divergences():
    """Settings currently differing from the reported analysis, named in prose."""
    return [
        SPEC_LABELS[k] for k, v in REPORTED_SPEC.items()
        if st.session_state.get(k, v) != v
    ]

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
PALETTE = [FAIL, SURVIVE, INK, BRASS]

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
# Streamlit 1.49 replaced use_container_width with width="stretch", and 1.6x
# dropped the old keyword. Pinning in requirements.txt is the real fix; this
# stays because a hosted rebuild can still move the version underneath you,
# and the failure mode is a deprecation banner above every single chart.


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
BUTTON_STRETCH = _stretch(st.button)
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
         label renders white on this cream page and disappears.
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

      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stCaptionContainer"] p,
      .stApp .note, .stApp .note *,
      .stApp [data-testid="stTickBar"],
      .stApp [data-testid="stTickBar"] div {{ color: {INK_SOFT}; }}

      .stApp [data-baseweb="select"] > div {{ background: {CARD};
                                              border-color: {LINE}; }}

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
                 font-size: 0.78rem; color: {INK_SOFT}; margin: -1.1rem 0 1.6rem 0;
                 gap: 1rem; flex-wrap: wrap; }}
      .topbar b {{ color: {BRASS}; font-weight: 600; }}
      .topbar .src {{ font-family: {MONO}; font-size: 0.72rem; color: {MUTED}; }}

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
      .stApp [data-testid="stMetricLabel"] p,
      .stApp [data-testid="stMetricLabel"] p span {{ font-size: 0.72rem; font-weight: 600;
                                         letter-spacing: 0.09em; text-transform: uppercase;
                                         color: {INK_SOFT}; }}
      .stApp [data-testid="stMetricValue"],
      .stApp [data-testid="stMetricValue"] div {{ font-family: {DISPLAY}; font-size: 2rem;
                                       font-weight: 600; color: {INK};
                                       font-variant-numeric: tabular-nums; }}
      .stApp [data-testid="stMetricDelta"],
      .stApp [data-testid="stMetricDelta"] div {{ font-size: 0.76rem; color: {INK_SOFT}; }}
      [data-testid="stMetricDelta"] svg {{ fill: {MUTED}; }}

      /* tabs -------------------------------------------------------- */
      .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {LINE};
                                           flex-wrap: wrap; }}
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
      [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
      [data-testid="stSidebar"] [data-testid="stExpander"] {{
          background: #2E2721; border: 1px solid #3B322B; border-radius: 8px; }}
      [data-testid="stSidebar"] label p {{ font-size: 0.82rem; font-weight: 500; }}

      /* buttons ----------------------------------------------------- */
      /* Streamlit's own button colours come from the active theme. When it
         falls back to dark, the label goes near-black on the espresso fill
         and the button reads as an empty box. Forced, like the text above. */
      .stApp .stButton button,
      .stApp .stDownloadButton button,
      .stApp [data-testid="stBaseButton-secondary"],
      .stApp [data-testid="stBaseButton-primary"] {{
          background: {CARD}; color: {INK} !important; border: 1px solid {LINE};
          border-radius: 8px; font-weight: 500; }}
      .stApp .stButton button p,
      .stApp .stDownloadButton button p,
      .stApp [data-testid="stBaseButton-secondary"] p,
      .stApp [data-testid="stBaseButton-primary"] p {{ color: {INK} !important; }}
      .stApp [data-testid="stBaseButton-primary"] {{
          background: {BRASS}; border-color: {BRASS}; }}
      .stApp [data-testid="stBaseButton-primary"],
      .stApp [data-testid="stBaseButton-primary"] p {{ color: {SHELL} !important; }}
      .stApp .stButton button:hover,
      .stApp .stDownloadButton button:hover {{ border-color: {BRASS}; }}

      /* tables ------------------------------------------------------ */
      [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}
      [data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {LINE};
                                    border-radius: 8px; }}

      /* four metric cards in a row become unreadable on a phone */
      @media (max-width: 640px) {{
        .masthead h1 {{ font-size: 1.9rem; }}
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] div {{ font-size: 1.5rem; }}
        [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
        [data-testid="stHorizontalBlock"] > div {{ min-width: 45% !important; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- caching


@st.cache_data(show_spinner="Reading the survey workbook…")
def load(path_or_buffer, sheet, unresolved_as_failure):
    df = ka.build_targets(
        ka.load_raw(path_or_buffer, sheet), unresolved_as_failure=unresolved_as_failure
    )
    X, single, derived = ka.build_predictors(df)
    return df, X, single, derived


@st.cache_data(show_spinner="Running the chi-square screen…")
def screen(X, y, yates=False):
    return ka.screen_predictors(X, y, yates=yates)


@st.cache_data(show_spinner="Fitting across repeated splits…")
def models(X, y, n_repeats, merge_rare):
    return ka.run_models(X, y, n_repeats=n_repeats, merge_rare=merge_rare)


# Fitted estimators are resources, not data. cache_data would pickle-copy
# every model on each rerun.
@st.cache_resource(show_spinner="Fitting the scoring models…")
def deployment(X, y):
    return ka.fit_deployment_models(X, y)


# ------------------------------------------------------- chart plumbing


def style(fig, height=None):
    """One visual grammar for every figure on the page."""
    fig.update_layout(
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
    for ann in fig.layout.annotations:
        if ann.font is None or ann.font.color is None:
            ann.font = dict(family=BODY, size=10, color=INK_SOFT)
    return fig


def plot(fig, title=None, height=None, note=None):
    if title:
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    # theme=None stops Streamlit overwriting the figure with its own template,
    # which paints every tick label white when Streamlit is running dark.
    st.plotly_chart(style(fig, height), theme=None, config=PLOTLY_CONFIG, **CHART_STRETCH)
    if note:
        st.markdown(f"<span class='note'>{note}</span>", unsafe_allow_html=True)


def bar(fig_data, xtitle=""):
    fig = go.Figure(fig_data)
    fig.update_layout(xaxis_title=xtitle)
    return fig


def note(text):
    st.markdown(f"<span class='note'>{text}</span>", unsafe_allow_html=True)


def finding(html):
    st.markdown(f"<div class='finding'>{html}</div>", unsafe_allow_html=True)


def heatmap(z, x, y, title, height=340):
    """
    Confusion-style heatmap with legible labels.

    The scale runs to near-black INK, so dark text on the largest cell — the
    one you most want read — disappears. The label colour flips once the cell
    is dark enough to swallow it.
    """
    z = np.asarray(z, dtype=float)
    hi = float(z.max()) if z.size else 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z, x=x, y=y, colorscale=[[0, CARD], [1, INK]],
            showscale=False, xgap=3, ygap=3,
            hovertemplate="%{y} · %{x}<br>%{z:.0f} shops<extra></extra>",
        )
    )
    # Written as annotations, not texttemplate: plotly rejects an array for
    # heatmap.textfont.color, so a per-cell colour is the only way to keep the
    # darkest cell's number legible.
    for r, row in enumerate(z):
        for c, v in enumerate(row):
            fig.add_annotation(
                x=x[c], y=y[r], text=f"{int(v)}", showarrow=False,
                font=dict(family=MONO, size=18,
                          color=INK if v < 0.55 * hi else PAPER),
            )
    fig.update_xaxes(showgrid=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, showline=False, ticks="")
    plot(fig, title=title, height=height)


def stat_line(row):
    """χ² and df are NaN on Fisher rows; print an em dash rather than nonsense."""
    if pd.isna(row.chi2):
        return f"{row.test} · n = {int(row.n)}"
    return f"{row.test}, χ² = {row.chi2:.2f}, df = {int(row.dof)} · n = {int(row.n)}"


def tab_label(icon, text):
    return f"{icon} {text}" if TAB_ICONS else text


# -------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### Kirana Pasal")
    st.markdown(
        "<span class='note' style='margin-top:-0.6rem'>Failure risk dashboard · "
        "small grocery shops, Kathmandu Valley</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Disabled on the public build. The ethical evaluation argues purpose
    # limitation has to be designed in rather than asserted, and an open URL
    # accepting workbooks of shop financials is exactly the drift it describes.
    upload = (
        st.file_uploader("Upload Excel data (.xlsx)", type=["xlsx"])
        if ALLOW_UPLOAD else None
    )
    if ALLOW_UPLOAD:
        st.caption(
            "Held in memory for this session only, never written to disk or retained. "
            "Do not upload records identifying a real shop or owner."
        )
    else:
        st.caption(
            "Upload is disabled on the deployed build. The bundled synthetic workbook "
            "is the only data this app reads."
        )
    source = upload if upload is not None else DATA_PATH

    sheet_names = ka.list_sheets(source)
    if len(sheet_names) > 1:
        default_ix = (
            sheet_names.index(ka.PREFERRED_SHEET)
            if ka.PREFERRED_SHEET in sheet_names else 0
        )
        sheet = st.selectbox("Sheet", sheet_names, index=default_ix)
    else:
        sheet = sheet_names[0] if sheet_names else None

    st.divider()
    st.markdown("**Step 1 — Choose the prediction target**")
    target_label = st.radio(
        "What should we predict?",
        ["Observed closure", "Owner expectation"],
        key="target_label",
        help="Choose recorded closure or the owner's expectation.",
    )
    st.caption("All available risk factors are used — no variable pre-selection.")

    st.divider()
    with st.expander("Step 2 — Robustness checks"):
        st.markdown(
            "<span class='note'>Defaults are the reported analysis. Change one at a "
            "time to see how far the results move.</span>",
            unsafe_allow_html=True,
        )
        unresolved_fail = st.checkbox(
            "Count unverified shops as failures",
            value=False,
            key="unresolved_fail",
            help="A shop that could not be traced is plausibly one that disappeared. "
            "Dropping those rows assumes it is not.",
        )
        merge_rare = st.checkbox(
            f"Pool categories with fewer than {ka.MIN_CATEGORY_N} shops",
            value=True,
            key="merge_rare",
            help="Applied to the significance tests either way. Applying it to the "
            "models too stops the encoded matrix carrying a column per two-shop category.",
        )
        yates = st.checkbox(
            "Yates' continuity correction",
            value=False,
            key="yates",
            help="Off by default: it deflates χ² on 2×2 tables and the same χ² is the "
            "numerator of the effect size.",
        )
        n_repeats = st.select_slider(
            "Repeated train/test splits", options=[1, 5, 10, 20], value=10,
            key="n_repeats",
            help="More repeats give a better read on how much of any score is the "
            "luck of one split. Slower.",
        )

    st.divider()
    st.button(
        "Restore reported specification",
        on_click=_apply_reported_spec,
        help="Sets every control back to the configuration the thesis tables were "
             "produced under, so the reported figures can be reproduced directly.",
        **BUTTON_STRETCH,
    )
    st.caption(
        "The defaults above are stricter than the reported analysis in places. "
        "This restores the reported one."
    )

try:
    df, X_all, single_vars, derived_vars = load(source, sheet, unresolved_fail)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Upload the workbook in the sidebar.")
    st.stop()
except ka.WorkbookError as exc:
    st.error(str(exc))
    st.stop()

target_col = "target_observed" if target_label == "Observed closure" else "target_belief"
y = df[target_col]

scr = screen(X_all, y, yates)
X = X_all  # every predictor, always: selecting on the full sample leaks

try:
    model_out = models(X, y, n_repeats, merge_rare)
except ka.WorkbookError as exc:
    st.error(str(exc))
    st.stop()

res = model_out["results"]
names = list(res)

# ------------------------------------------------------------- masthead

src_name = getattr(upload, "name", DATA_PATH)
divergences = _spec_divergences()

st.markdown(
    f"""
    <div class='topbar'>
      <div><b>Kirana Pasal</b> &nbsp;/&nbsp; {len(df)} records · {target_label.lower()}</div>
      <div class='src'>{src_name} · sheet “{sheet}” · synthetic survey extract</div>
    </div>
    <div class='masthead'>
      <div class='eyebrow'>Method demonstration · synthetic data · not for use on real shops</div>
      <h1>Predicting failure risk in kirana pasals</h1>
      <p>Currently predicting <b>{target_label.lower()}</b> from all
      {X.shape[1]} recorded factors. Change the target in the sidebar and every
      figure below refits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Which specification is on screen is the first thing a reader needs, because
# the defaults here are not the ones the reported tables were produced under.
if divergences:
    st.warning(
        "**Not the reported specification.** These settings differ from the "
        "configuration the thesis tables were produced under: "
        + ", ".join(divergences)
        + ". Figures below will not match the reported tables. Use **Restore "
        "reported specification** in the sidebar to reproduce them."
    )
else:
    st.success(
        "**Reported specification.** Every control is set to the configuration the "
        "thesis tables were produced under, so the figures below are the reported ones."
    )

st.info(
    "**Quick start:** choose the target on the left → open **Risk Factors** for what "
    "is associated with failure → **Models** for how well it can be predicted → "
    "**Threshold** for the cost of being wrong."
)

tabs = st.tabs([
    tab_label("🏠", "Home"),
    tab_label("🔎", "Risk Factors"),
    tab_label("↔️", "Compare"),
    tab_label("🤖", "Models"),
    tab_label("🎚️", "Threshold"),
    tab_label("🏪", "Score a Shop"),
    tab_label("📋", "Methods"),
])

# ------------------------------------------------------------ home

with tabs[0]:
    st.header(tab_label("🏠", "Home"))
    st.caption("Start here. What the dataset contains, and the constraint that shapes everything.")

    resolved = int(df["target_observed"].notna().sum())
    failures = int(df["target_observed"].sum())
    belief_answered = int(df["target_belief"].notna().sum())
    belief_flagged = int(df["target_belief"].sum())

    c = st.columns(4)
    c[0].metric("Total shops", f"{len(df)}")
    c[1].metric("Known outcomes", f"{resolved}", f"{len(df) - resolved} unverified",
                delta_color="off")
    c[2].metric("Failed shops", f"{failures}",
                f"{failures / resolved:.1%} of known" if resolved else "—",
                delta_color="off")
    # Denominator is answered, not all: a blank expectation is not a vote of
    # confidence, and counting it as one understated this share.
    c[3].metric("Owner-flagged risk", f"{belief_flagged}",
                f"{belief_flagged / belief_answered:.1%} of {belief_answered} answered"
                if belief_answered else "—",
                delta_color="off")

    if unresolved_fail:
        st.warning(
            "Robustness setting active: unverified shops are being counted as failures. "
            "Every figure on every tab reflects that recoding, not the reported analysis."
        )

    st.subheader("The width problem")
    c = st.columns(4)
    c[0].metric("Predictor variables", f"{X.shape[1]}")
    c[1].metric("Encoded columns", f"{model_out['n_encoded_cols']}",
                f"{model_out['n_encoded_cols_unmerged']} unpooled", delta_color="off")
    c[2].metric("Training rows", f"{model_out['n_train']}")
    c[3].metric("Failures in test set", f"{model_out['test_events']} of {model_out['n_test']}")
    finding(
        f"<b>{model_out['n_encoded_cols']} encoded columns against "
        f"{model_out['n_train']} training rows.</b> Pooling categories under "
        f"n = {ka.MIN_CATEGORY_N} brings this down from "
        f"{model_out['n_encoded_cols_unmerged']}; the rest is the survey instrument's own "
        f"width. This ratio, more than any modelling choice, explains why the "
        f"cross-validated scores later in this dashboard do not survive the held-out test set."
    )

    st.subheader("Where the shops are, and how they closed")
    left, right = st.columns([1, 1])

    with left:
        if "District" in df.columns:
            counts = df["District"].value_counts()
            plot(
                bar([go.Bar(x=counts.values, y=counts.index, orientation="h",
                            marker_color=INK, marker_line_width=0)], "Shops"),
                title="Shops by district", height=320,
            )
        else:
            st.info("No `District` column in this workbook.")

    with right:
        status = df["Outcome Status"].value_counts()
        colours = [
            FAIL if s in ka.FAILURE_STATUSES
            else (MUTED if s == ka.UNRESOLVED_STATUS else SURVIVE)
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
    st.header(tab_label("🔎", "Risk Factors"))
    st.caption(
        "Chi-square tests of independence against the selected target"
        + (" with Yates' correction" if yates else " without Yates' correction")
        + ", Cramér's V as the effect size, and Benjamini–Hochberg correction for running "
        "the tests in bulk. Where a 2×2 table has too many small expected counts, Fisher's "
        "exact test supplies the p-value and no χ² is reported."
    )

    if scr.empty:
        st.info("No variable produced a usable contingency table for this target.")
        st.stop()

    n_raw = int(scr["significant_raw"].sum())
    n_bh = int(scr["survives_bh"].sum())
    c = st.columns(3)
    c[0].metric("Variables tested", f"{len(scr)}")
    c[1].metric("Significant at p < .05", f"{n_raw}")
    c[2].metric("Survive BH correction", f"{n_bh}", f"{n_raw - n_bh} lost", delta_color="off")

    sig = scr[scr["significant_raw"]].sort_values("cramers_v")
    if len(sig):
        # Meaning is carried by a marker as well as by colour: brass-vs-grey is
        # the one pair here that fails for common colour vision deficiency.
        labels = [
            f"★ {v}" if s else f"   {v}"
            for v, s in zip(sig["variable"], sig["survives_bh"])
        ]
        colours = [BRASS if s else MUTED for s in sig["survives_bh"]]
        fig = bar(
            [
                go.Bar(
                    x=sig["cramers_v"], y=labels, orientation="h",
                    marker_color=colours, marker_line_width=0,
                    text=[f"{v:.3f}" for v in sig["cramers_v"]],
                    textposition="outside", textfont=dict(family=MONO, size=11, color=INK_SOFT),
                    cliponaxis=False,
                    hovertext=[
                        (f"{r.test}<br>"
                         + ("" if pd.isna(r.chi2) else f"χ² = {r.chi2:.2f}, df = {int(r.dof)}<br>")
                         + f"p = {r.p:.2e}<br>q = {r.q:.3f}<br>"
                         + f"V = {r.cramers_v:.3f} (corrected {r.cramers_v_corrected:.3f})")
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
            note="Starred and brass survives correction; grey does not. Cramér's V is biased "
                 "upward at this sample size, so hover for the Bergsma-corrected value. Effect "
                 "sizes are small throughout, which fits the framing that risk builds from a "
                 "combination of pressures rather than one decisive weakness.",
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

    obs_mask = y.notna()
    s = X_all.loc[obs_mask, choice].astype(str)
    tab = pd.crosstab(s, y.loc[obs_mask].astype(int))
    tab = tab.reindex(columns=[0, 1], fill_value=0)
    tab.columns = ["Survived", "Failed"]

    n_per = tab.sum(axis=1)
    rate = (tab["Failed"] / n_per).sort_values()
    n_sorted = n_per.loc[rate.index]

    left, right = st.columns([2, 1])
    with left:
        # A category resting on three shops used to render an identical bar to
        # one resting on eighty. The denominator now travels with the label.
        fig = bar(
            [go.Bar(
                x=rate.values,
                y=[f"{i}  (n={int(n)})" for i, n in zip(rate.index, n_sorted)],
                orientation="h",
                marker_color=[FAIL if n >= SMALL_N else MUTED for n in n_sorted],
                marker_line_width=0,
                text=[f"{v:.1%}" for v in rate.values], textposition="outside",
                textfont=dict(family=MONO, size=11, color=INK_SOFT), cliponaxis=False,
            )],
            "Share of shops that failed",
        )
        base = float(y.loc[obs_mask].mean())
        fig.add_vline(x=base, line_dash="dash", line_color=INK, line_width=1,
                      annotation_text=f"base rate {base:.1%}",
                      annotation_font=dict(size=10, color=INK_SOFT))
        plot(fig, title=f"Failure rate by {choice}", height=max(300, 40 * len(rate)),
             note=f"Grey bars rest on fewer than {SMALL_N} shops. Their percentages move by "
                  f"tens of points on a single shop and should not be read as rates.")
    with right:
        st.metric("Cramér's V", f"{row.cramers_v:.3f}")
        st.metric("Bias-corrected V", f"{row.cramers_v_corrected:.3f}")
        st.metric("p-value", f"{row.p:.4g}")
        st.metric("BH q-value", f"{row.q:.4g}")
        st.caption(stat_line(row))
        if row.low_cell_warning:
            st.warning("Over 20% of expected cell counts fall below five. Read with caution.")
        st.dataframe(tab, **STRETCH)

    with st.expander("Full screening table"):
        st.dataframe(
            scr[["variable", "test", "chi2", "dof", "p", "q", "cramers_v",
                 "cramers_v_corrected", "n", "n_categories",
                 "significant_raw", "survives_bh"]].round(4),
            height=400, **STRETCH,
        )
        st.download_button(
            "Download screening table (CSV)",
            scr.to_csv(index=False).encode(),
            file_name=f"screening_{target_col}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------- two targets

with tabs[2]:
    st.header(tab_label("↔️", "Compare"))
    st.caption(
        "The study carries two definitions of failure: what the shop did, and what its owner "
        "expected. Whoever picks the target decides whose distress the model can see."
    )

    ag = ka.agreement(df)
    c = st.columns(4)
    c[0].metric("Agreement", f"{ag['agreement_rate']:.1%}")
    c[1].metric("Correlation (φ)", f"{ag['phi']:.3f}")
    c[2].metric("Chi-square p", f"{ag['p']:.3f}")
    c[3].metric("Failures not anticipated",
                f"{ag['failed_unanticipated']} of {ag['n_failures']}")
    st.caption(
        f"Computed on the {ag['n']} shops with both an outcome and an answered expectation; "
        f"{ag['n_dropped']} records are missing one or the other."
    )

    finding(
        f"{ag['failed_unanticipated']} of the {ag['n_failures']} shops that failed had owners "
        f"who did not expect to fail."
    )

    left, right = st.columns([1, 1])
    with left:
        heatmap(
            ag["table"].values,
            ["Owner expected to continue", "Owner expected trouble"],
            ["Survived", "Failed"],
            "Observed outcome against owner expectation",
        )

    with right:
        scr_obs = screen(X_all, df["target_observed"], yates)
        scr_bel = screen(X_all, df["target_belief"], yates)
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
        "Modelling recorded closure makes one population visible, modelling expectation makes "
        "another, and that choice is normally made early in a project without being recorded "
        "as a decision at all."
    )

# ---------------------------------------------------- model comparison

with tabs[3]:
    st.header(tab_label("🤖", "Models"))
    st.caption(
        f"70:30 stratified split, tuning by grid search inside "
        f"{model_out['n_splits']}-fold cross-validation on the training portion, test set "
        f"scored once, repeated across {model_out['n_repeats']} random splits."
    )

    rows = []
    for n in names:
        m = res[n]["at_050"]
        rows.append({
            "Model": n, "CV AUC": res[n]["cv_auc"], "Test AUC": res[n]["test_auc"],
            "Test AP": res[n]["test_ap"],
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "Specificity": m["specificity"], "F1": m["f1"],
            "TN/FP/FN/TP": f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}",
        })
    table = pd.DataFrame(rows)
    st.dataframe(table.round(3), hide_index=True, **STRETCH)
    st.download_button("Download results table (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"models_{target_col}.csv", mime="text/csv")
    note(
        "Every row is the seed-42 split at a 0.50 cut-off. Because all three models reweight "
        "the classes, 0.50 is not a neutral threshold and not directly comparable between "
        "them — read the columns as one illustration, not as a ranking."
    )

    st.subheader("How much of this is the split?")
    rep = model_out["repeats"]
    if model_out["n_repeats"] > 1:
        fig = go.Figure()
        for i, n in enumerate(names):
            fig.add_trace(go.Box(
                y=rep.loc[rep["model"] == n, "test_auc"], name=n, boxpoints="all",
                jitter=0.5, pointpos=0, marker_color=PALETTE[i % len(PALETTE)],
                line=dict(width=1.5),
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=INK, line_width=1,
                      annotation_text="chance", annotation_font=dict(size=10, color=INK_SOFT))
        fig.update_layout(yaxis_title="Test AUC", showlegend=False, yaxis_range=[0, 1])
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        plot(
            fig, title=f"Test AUC across {model_out['n_repeats']} random splits", height=380,
            note="Each point is one 70:30 split. This spread is the honest uncertainty on every "
                 "single-split number in the table above. Where the boxes overlap, the models "
                 "are not distinguishable on this sample, and picking a winner is picking a seed.",
        )
        with st.expander("Repeat summary"):
            st.dataframe(model_out["repeat_summary"], **STRETCH)
    else:
        st.info("Raise 'Repeated train/test splits' in the sidebar to see the spread.")

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
        for i, n in enumerate(names):
            fpr, tpr = res[n]["roc"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{n} ({res[n]['test_auc']:.3f})",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(color=MUTED, dash="dash", width=1)))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="h", y=-0.18, x=0))
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        plot(fig, title="ROC curves on the held-out test set", height=400)

    st.subheader("Precision against recall")
    fig = go.Figure()
    for i, n in enumerate(names):
        rec, prec = res[n]["pr"]
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{n} (AP {res[n]['test_ap']:.3f})",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=model_out["base_rate"], line_dash="dash", line_color=MUTED, line_width=1,
                  annotation_text=f"base rate {model_out['base_rate']:.2f}",
                  annotation_font=dict(size=10, color=INK_SOFT))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1],
                      legend=dict(orientation="h", y=-0.18, x=0))
    fig.update_yaxes(showgrid=True, gridcolor=LINE)
    plot(
        fig, height=400,
        note="On an imbalanced test set ROC flatters everything, because most of its area comes "
             "from the easy negatives. The dashed line is what random ranking achieves here: an "
             "average precision near it means the ranking carries little usable signal, whatever "
             "the AUC says.",
    )

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

    st.subheader("What each model leans on")
    which = st.selectbox("Model", names, key="imp_model")
    imp = res[which]["importances"]
    if len(imp):
        top = imp.head(15).iloc[::-1]
        valcol = "odds_ratio" if "odds_ratio" in top.columns else "importance"
        fig = bar(
            [go.Bar(x=top["magnitude"], y=top["feature"], orientation="h",
                    marker_color=INK, marker_line_width=0,
                    text=[f"{v:.3f}" for v in top[valcol]], textposition="outside",
                    textfont=dict(family=MONO, size=11, color=INK_SOFT), cliponaxis=False)],
            "Coefficient magnitude" if valcol == "odds_ratio" else "Gini importance",
        )
        plot(
            fig, title=f"{which}: fifteen largest terms", height=max(320, 30 * len(top)),
            note="Logistic terms are labelled with odds ratios, tree models with Gini "
                 "importance. Neither is a causal claim, and at this encoded width both move "
                 "substantially between splits — read them as description, not mechanism.",
        )

    with st.expander("Selected hyperparameters"):
        for n in names:
            st.write(f"**{n}** — {res[n]['best_params']}")

# ------------------------------------------------- threshold explorer

with tabs[4]:
    st.header(tab_label("🎚️", "Threshold"))
    st.caption(
        "The cut-off is not a technical detail. It decides who carries the cost of being "
        "wrong, and no amount of tuning turns that into a statistical question."
    )

    ctrl = st.columns([1, 1, 2])
    model_name = ctrl[0].selectbox("Model", names, key="thr_model")
    entry = res[model_name]
    suggested = entry["suggested_threshold"]
    use_suggested = ctrl[1].checkbox("Use out-of-fold cut-off", value=True)

    if use_suggested:
        threshold = suggested
        ctrl[2].metric("Cut-off chosen on training folds", f"{suggested:.2f}")
    else:
        threshold = ctrl[2].slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    if use_suggested:
        st.caption(
            "Picked on out-of-fold training predictions as the highest-precision cut-off that "
            "still reaches 70% recall. The test set played no part in choosing it."
        )
    else:
        st.warning(
            "Sliding the cut-off while watching the test-set confusion matrix is the same leak "
            "as selecting variables on the full sample. Use the slider to show the trade-off, "
            "not to choose the operating point you then report."
        )

    proba = entry["proba"]
    y_test = model_out["y_test"]
    m = ka.metrics_at(y_test, proba, threshold)

    c = st.columns(4)
    c[0].metric("Shops warned in time", f"{m['tp']} of {m['tp'] + m['fn']}")
    c[1].metric("Healthy shops wrongly flagged", f"{m['fp']} of {m['tn'] + m['fp']}")
    c[2].metric("Failing shops missed", f"{m['fn']}")
    c[3].metric("Recall / Precision", f"{m['recall']:.2f} / {m['precision']:.2f}")

    left, right = st.columns([1, 1])

    with left:
        heatmap(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            ["Predicted safe", "Predicted at risk"],
            ["Actually survived", "Actually failed"],
            f"{model_name} at threshold {threshold:.2f}",
        )
        st.caption(
            f"{model_out['n_test']} test shops, {model_out['test_events']} of them failures. "
            "Every cell here can move by one or two shops on a different split."
        )

    with right:
        grid = np.arange(0.05, 0.96, 0.01)
        sweep = [ka.metrics_at(y_test, proba, float(t)) for t in grid]
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

    finding(
        f"At this setting the model warns {m['tp']} of the {m['tp'] + m['fn']} shops that "
        f"failed and puts {m['fp']} shops that survived under suspicion. If a flag triggered an "
        f"offer of support, a false positive costs little. If it fed a credit decision, the "
        f"framework would be denying finance mostly to shops that were going to survive."
    )

# ---------------------------------------------------------- score a shop

with tabs[5]:
    st.header(tab_label("🏪", "Score a Shop"))
    st.caption(
        "Fitted on every resolved observation. A number produced here inherits every "
        "limitation on the Methods tab: it demonstrates the pipeline, it is not a credit "
        "assessment."
    )

    # The study concludes that this framework cannot support a decision about an
    # individual shop. A scorer reachable in one click from a public URL
    # contradicts that in the build even while the prose repeats it, so the form
    # does not render until the reader has said out loud what it is.
    st.warning(
        "**This tab scores one shop, and the study concludes that it should not be "
        "used to.** Test ROC-AUC on the reported specification sits close to chance. "
        "The form exists to demonstrate the pipeline end to end, and to show what a "
        "system like this would ask an owner to disclose \u2014 not to assess a business."
    )
    ack = st.checkbox(
        "I understand this is a method demonstration on synthetic data, and not a credit "
        "assessment, screening tool, or basis for any decision about a real shop.",
        key="score_ack",
    )

    if not ack:
        st.info("Tick the box above to open the form.")
    else:
        cat_cols, bin_cols = ka.split_column_kinds(X)

        # Levels and defaults come from the data, not from a fitted bundle, so the
        # form renders instantly and the models are only fitted when a score is
        # actually asked for.
        levels = {c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols}
        defaults = {c: X[c].astype(str).mode().iat[0] for c in cat_cols}

        # Every variable gets an input. Leaving some of them to default silently
        # meant most of the shop's profile contributed nothing to the score while
        # the result still looked like an answer.
        record = {}
        PER_GROUP = 9
        groups = [cat_cols[i:i + PER_GROUP] for i in range(0, len(cat_cols), PER_GROUP)]
        for gi, group in enumerate(groups):
            label = (
                f"Shop details ({gi * PER_GROUP + 1}\u2013"
                f"{gi * PER_GROUP + len(group)} of {len(cat_cols)})"
            )
            with st.expander(label, expanded=(gi == 0)):
                cols = st.columns(3)
                for i, col in enumerate(group):
                    opts = levels[col]
                    record[col] = cols[i % 3].selectbox(
                        col, opts, index=opts.index(defaults[col]), key=f"nsp_{col}"
                    )

        if bin_cols:
            with st.expander("Multi-response options that apply", expanded=False):
                chosen = st.multiselect("Select all that apply", bin_cols, key="nsp_multi")
            for col in bin_cols:
                record[col] = int(col in chosen)

        note(
            f"Every field starts at the most common answer in the survey. The length of "
            f"this form is itself a finding: {len(cat_cols) + len(bin_cols)} disclosures, "
            "and the most informative of them \u2014 working capital, missed instalments, "
            "household shocks \u2014 are the ones an owner has most reason to protect. In a "
            "kirana pasal, household and business finance are rarely separate, so shop "
            "data is household data."
        )

        if st.button("Score this shop", type="primary"):
            bundle = deployment(X, y)
            out = ka.predict_new_shop(bundle, record)
            if out["unmatched"]:
                st.warning(
                    "These answers were pooled or unseen at fit time and contributed nothing: "
                    + "; ".join(out["unmatched"])
                )
            mcols = st.columns(len(out["probabilities"]))
            for (name, p), col in zip(out["probabilities"].items(), mcols):
                thr = out["thresholds"].get(name, 0.5)
                col.metric(name, f"{p:.1%}",
                           "flagged" if p >= thr else "not flagged", delta_color="off")
            finding(
                "These are ranking scores, not calibrated probabilities. A shop at 62% is "
                "ranked above one at 41%; neither figure means a six-in-ten chance of closing. "
                "Calibration would need a larger sample and a held-out calibration set. "
                "Three models are shown rather than one verdict on purpose: where they "
                "disagree, the disagreement is the result."
            )


# ---------------------------------------------------------------- methods

with tabs[6]:
    st.header(tab_label("📋", "Methods"))

    st.markdown(f"""
**Data.** {len(df)} shop records from `{src_name}`, sheet “{sheet}”. A static survey
extract; nothing on this page is live.

**Targets.** *Observed closure* codes {", ".join(ka.FAILURE_STATUSES).lower()} as failure,
verified-operating shops as survival, and “{ka.UNRESOLVED_STATUS}” as missing
{"— currently recoded as failure by the robustness setting" if unresolved_fail else ""}.
*Owner expectation* codes {" or ".join(ka.BELIEF_HIGH_RISK).lower()} as high risk; a blank
answer is missing, not a vote of confidence.

**Predictors.** The outcome-bearing fields ({", ".join(ka.OUTCOME_COLS)}) are dropped
before modelling so neither target can leak back in. Semicolon-delimited multi-response
items become one binary indicator per option rather than one category per observed
combination. Categories with fewer than {ka.MIN_CATEGORY_N} observations are pooled
{"before both testing and encoding" if merge_rare else "for the significance tests only"};
the pooling map is learned on training rows alone. All {X.shape[1]} variables are used —
there is no pre-selection step, because screening on the full sample and then testing on
part of it makes the test set complicit in choosing the predictors.

**Tests.** Chi-square {"with" if yates else "without"} Yates' continuity correction;
Fisher's exact where a 2×2 table has more than 20% of expected counts below five, in which
case no χ² is reported. Cramér's V given raw and Bergsma bias-corrected. Benjamini–Hochberg
applied across all {len(scr)} tests.

**Models.** 70:30 stratified split; grid search inside {model_out['n_splits']}-fold
stratified cross-validation on the training portion; test set scored once; repeated across
{model_out['n_repeats']} seeds. Class imbalance handled by reweighting rather than
resampling. Thresholds selected on out-of-fold training predictions as the
highest-precision cut-off still reaching 70% recall. This differs from the thesis, which
selected for F1 on the same folds; the rule here targets the early-warning use directly
rather than balancing the two errors equally, so its confusion matrices are not identical
to the reported ones. Seed {ka.SEED}.

**Known limitations.**
1. Complete-case analysis on an outcome whose missingness is plausibly informative — an
   untraceable shop is more likely to be a closed one. The sidebar toggle exists to show
   how far that assumption carries the results.
2. Predicted values are uncalibrated, so a threshold is a rank-based operating point, not
   a probability.
3. {model_out['n_encoded_cols']} encoded columns against {model_out['n_train']} training
   rows. No modelling choice repairs that ratio.
4. Failure pools three materially different fates for an owner.
5. Effect sizes are small throughout, and the reweighting makes 0.50 a non-neutral,
   cross-model-incomparable cut-off.
    """)

    st.divider()
    st.markdown(
        """
**Correspondence with the thesis.** The defaults here are stricter than the reported
analysis in three places, each deliberate.

1. **No variable pre-selection.** The thesis reports both the full and the
   chi-square-reduced predictor sets, and states that the reduced figures are optimistic
   because the screen saw the full sample. This dashboard implements the correction rather
   than the compromise: every predictor, always.
2. **Repeated splits.** The thesis reports a single seed-42 split. The dashboard shows the
   distribution that number was drawn from, which is the honest uncertainty on it.
3. **Rare-category pooling applied to the models**, not only to the significance tests,
   which lowers the encoded column count below the reported 266.

A fourth difference is the threshold rule, described above. Use **Restore reported
specification** in the sidebar to set the first three back and reproduce the reported
tables directly.
        """
    )

    with st.expander("Environment"):
        import sklearn
        import plotly
        st.code(
            f"streamlit {st.__version__}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
            f"scikit-learn {sklearn.__version__}\nplotly {plotly.__version__}\n"
            f"xgboost available: {ka.HAS_XGB}\nseed: {ka.SEED}",
            language="text",
        )