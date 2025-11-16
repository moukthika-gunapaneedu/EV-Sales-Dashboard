import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# ---------- CONFIG ----------

DATA_PATH = Path("EV Data Explorer 2025.xlsx")
SHEET_NAME = "EV sales countries"

BLUE = "#1f4e79"
LIGHT_BLUE = "#a7c4e0"
LIGHT_GREY = "#e5e5e5"
DARK_GREY = "#444444"


# ---------- DATA LOADING & CLEANING ----------

@st.cache_data
def load_ev_sales(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load EV sales from IEA 'EV sales countries' sheet and reshape to long format:
    columns: region_country, Year, EV_Sales
    """
    # Row 7 (0-based) is the header row with: region_country, 2010, 2011, ...
    df = pd.read_excel(path, sheet_name=sheet_name, header=7)

    # Keep region_country + year columns
    year_cols = []
    for c in df.columns:
        if isinstance(c, (int, float)):
            year_cols.append(c)
        elif isinstance(c, str) and c.strip().isdigit():
            year_cols.append(c)

    if "region_country" not in df.columns:
        raise ValueError("Expected a 'region_country' column in the sheet header row.")

    df_wide = df[["region_country"] + year_cols]

    # Melt to long format
    df_long = df_wide.melt(
        id_vars=["region_country"],
        value_vars=year_cols,
        var_name="Year",
        value_name="EV_Sales",
    )

    # Drop rows with no data
    df_long = df_long.dropna(subset=["EV_Sales"])
    df_long = df_long[df_long["region_country"].notna()]

    # Fix types
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["EV_Sales"] = df_long["EV_Sales"].astype(float)

    # Simple clean: trim whitespace
    df_long["region_country"] = df_long["region_country"].astype(str).str.strip()

    return df_long


# ---------- KPI HELPERS ----------

def compute_global_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EV sales by year (for the current selection)."""
    return (
        df_long.groupby("Year", as_index=False)["EV_Sales"]
        .sum()
        .sort_values("Year")
    )


def compute_kpis(df_long: pd.DataFrame, year_selected: int):
    global_by_year = compute_global_summary(df_long)

    # Total sales selected year
    total_selected = (
        global_by_year.loc[global_by_year["Year"] == year_selected, "EV_Sales"]
        .sum()
    )

    # YoY growth (vs previous year) – if previous year exists
    prev_year = year_selected - 1
    prev_sales = (
        global_by_year.loc[global_by_year["Year"] == prev_year, "EV_Sales"]
        .sum()
    )

    yoy_growth = None
    if prev_sales > 0:
        yoy_growth = (total_selected - prev_sales) / prev_sales * 100.0

    # Top country for selected year
    df_year = df_long[df_long["Year"] == year_selected]
    top_row = df_year.sort_values("EV_Sales", ascending=False).head(1)

    if not top_row.empty:
        top_country = top_row["region_country"].iloc[0]
        top_sales = top_row["EV_Sales"].iloc[0]
    else:
        top_country, top_sales = None, None

    num_countries = df_year["region_country"].nunique()

    return {
        "total_selected": total_selected,
        "yoy_growth": yoy_growth,
        "top_country": top_country,
        "top_sales": top_sales,
        "num_countries": num_countries,
    }


# ---------- PLOTTING HELPERS ----------

def make_global_line(df_long: pd.DataFrame):
    global_by_year = compute_global_summary(df_long)

    if global_by_year.empty:
        return px.line()  # empty fig if no data

    fig = px.line(
        global_by_year,
        x="Year",
        y="EV_Sales",
    )

    fig.update_traces(
        mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(
            size=6,
            color=BLUE,
            opacity=0.9,
            line=dict(width=0),
        ),
    )

    latest_year = int(global_by_year["Year"].max())
    latest_value = float(
        global_by_year.loc[global_by_year["Year"] == latest_year, "EV_Sales"].iloc[0]
    )

    # Clean, subtle annotation near the last point
    fig.add_annotation(
        x=latest_year,
        y=latest_value,
        text=f"Highest EV sales ({latest_year})",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=50,
        font=dict(
            color=DARK_GREY,
            size=13,
            family="Arial",
            weight="bold",
        ),
        bgcolor="white",
        bordercolor=DARK_GREY,
        borderwidth=0.7,
        borderpad=4,
    )

    fig.update_layout(
        title_text="",
        height=400,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickmode="linear",
            dtick=2,
            tickfont=dict(color="#888888", size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f2f2f2",   # very light gridlines
            zeroline=False,
            tickfont=dict(color="#888888", size=10),
        ),
    )

    return fig


def make_yoy_chart(df_long: pd.DataFrame):
    """
    Year-over-year % growth in global EV sales.
    """
    global_by_year = compute_global_summary(df_long)

    if global_by_year.shape[0] < 2:
        return px.bar()  # not enough data for YoY

    # Compute YoY growth
    global_by_year = global_by_year.sort_values("Year").copy()
    global_by_year["EV_Sales_prev"] = global_by_year["EV_Sales"].shift(1)
    global_by_year["YoY_growth"] = (
        (global_by_year["EV_Sales"] - global_by_year["EV_Sales_prev"])
        / global_by_year["EV_Sales_prev"]
        * 100.0
    )
    global_by_year = global_by_year.dropna(subset=["YoY_growth"])

    fig = px.bar(
        global_by_year,
        x="Year",
        y="YoY_growth",
    )

    fig.update_traces(
        marker_color=LIGHT_BLUE,
    )

    fig.update_layout(
        title_text="",
        height=280,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickmode="linear",
            dtick=2,
            tickfont=dict(color="#888888", size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f2f2f2",
            zeroline=False,
            tickfont=dict(color="#888888", size=10),
        ),
    )

    return fig


def make_top_countries_bar(df_long: pd.DataFrame, year_selected: int, top_n: int = 10):
    df_year = df_long[df_long["Year"] == year_selected].copy()
    df_year = df_year.sort_values("EV_Sales", ascending=False).head(top_n)

    # Highlight the top market
    if not df_year.empty:
        top_country = df_year.iloc[0]["region_country"]
        df_year["color"] = df_year["region_country"].apply(
            lambda c: BLUE if c == top_country else LIGHT_BLUE
        )
    else:
        df_year["color"] = BLUE

    fig = px.bar(
        df_year,
        x="EV_Sales",
        y="region_country",
        orientation="h",
    )

    # descending layout (largest at top)
    fig.update_yaxes(categoryorder="total ascending")

    fig.update_traces(marker_color=df_year["color"])

    fig.update_layout(
        title_text="",
        title_x=0.0,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=120, r=40, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",   # VERY light grey gridlines
            gridwidth=0.5,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(showgrid=False, showline=False),
    )

    return fig


def get_fastest_growing_markets(df_long: pd.DataFrame, year_selected: int, top_n: int = 3):
    """
    Return top N countries by year-over-year % growth for the selected year.
    Only includes countries that have data in both the selected year and the previous year.
    """
    prev_year = year_selected - 1

    df_curr = df_long[df_long["Year"] == year_selected][["region_country", "EV_Sales"]].copy()
    df_prev = df_long[df_long["Year"] == prev_year][["region_country", "EV_Sales"]].copy()

    if df_curr.empty or df_prev.empty:
        return pd.DataFrame(columns=["region_country", "EV_Sales", "YoY_growth"])

    df_merged = df_curr.merge(
        df_prev,
        on="region_country",
        how="inner",
        suffixes=("_curr", "_prev"),
    )

    # avoid division by zero
    df_merged = df_merged[df_merged["EV_Sales_prev"] > 0].copy()
    if df_merged.empty:
        return pd.DataFrame(columns=["region_country", "EV_Sales", "YoY_growth"])

    df_merged["YoY_growth"] = (
        (df_merged["EV_Sales_curr"] - df_merged["EV_Sales_prev"])
        / df_merged["EV_Sales_prev"]
        * 100.0
    )

    df_merged = df_merged.sort_values("YoY_growth", ascending=False).head(top_n)

    # Rename back to simple columns
    df_merged = df_merged.rename(
        columns={
            "EV_Sales_curr": "EV_Sales",
        }
    )

    return df_merged[["region_country", "EV_Sales", "YoY_growth"]]


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Global EV Sales Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("Global EV Sales")
    st.caption(
        "Insights from the IEA EV Data Explorer (2025)"
    )

    if not DATA_PATH.exists():
        st.error(
            f"Excel file not found at `{DATA_PATH}`. "
            "Put `EV Data Explorer 2025.xlsx` in the same folder as this app."
        )
        return

    df_long = load_ev_sales(DATA_PATH, SHEET_NAME)
    df_long = df_long[
        df_long["region_country"].notna()
        & (df_long["region_country"].str.strip() != "")
        & (~df_long["region_country"].str.contains("undetermined|undefined", case=False))
    ]

    min_year = int(df_long["Year"].min())
    max_year = int(df_long["Year"].max())

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.header("Filters")

    year_selected = st.sidebar.slider(
        "Select year",
        min_value=min_year,
        max_value=max_year,
        value=max_year,
        step=1,
    )

    # Country / region filter (this is the "place" selector)
    countries = sorted(df_long["region_country"].unique())
    selected_countries = st.sidebar.multiselect(
        "Select countries / regions",
        options=countries,
        default=countries,  # all selected by default
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Data source: IEA Global EV Data Explorer (2025)")

    # Apply filters
    if selected_countries:
        df_filtered = df_long[df_long["region_country"].isin(selected_countries)]
    else:
        df_filtered = df_long.iloc[0:0]  # empty if nothing selected

    # ---------- KPIs ----------
    kpi = compute_kpis(df_filtered, year_selected)
    total_selected = kpi["total_selected"]
    yoy_growth = kpi["yoy_growth"]
    top_country = kpi["top_country"]
    top_sales = kpi["top_sales"]
    num_countries = kpi["num_countries"]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(
            label=f"Total EV sales in {year_selected}",
            value=f"{int(total_selected):,}" if total_selected > 0 else "n/a",
        )

    with kpi_col2:
        if yoy_growth is not None:
            st.metric(
                label=f"YoY growth vs {year_selected - 1}",
                value=f"{yoy_growth:.1f} %",
            )
        else:
            st.metric(
                label=f"YoY growth vs {year_selected - 1}",
                value="n/a",
            )

    with kpi_col3:
        if top_country is not None:
            st.metric(
                label=f"Top EV market in {year_selected}",
                value=top_country,
            )
        else:
            st.metric(
                label=f"Top EV market in {year_selected}",
                value="n/a",
            )

    with kpi_col4:
        st.metric(
            label=f"Countries / regions with sales data ({year_selected})",
            value=num_countries if num_countries > 0 else 0,
        )

    st.markdown("---")

    # ---------- MAIN CHARTS: LINE + YOY ----------
    line_col, = st.columns(1)
    with line_col:
        st.subheader("EV Sales Over Time (Selected Regions)")
        st.caption(
            "Global EV sales accelerated sharply after 2020, reaching their highest levels in the most recent year"
        )
        fig_line = make_global_line(df_filtered)
        st.plotly_chart(fig_line, use_container_width=True, key="line_ev_sales")

        st.subheader("Year-over-Year Growth in Global EV Sales")
        st.caption(
            "How fast EV sales are growing each year compared to the previous year<br>"
            "YoY change = EV_sales(year) − EV_sales(year−1)",
            unsafe_allow_html=True
        )
        fig_yoy = make_yoy_chart(df_filtered)
        st.plotly_chart(fig_yoy, use_container_width=True, key="bar_yoy_growth")

    st.markdown("")

    # ---------- SECOND ROW: BAR + TOP 3 ----------
    bar_col, table_col = st.columns([2, 1])

    with bar_col:
        st.subheader(f"Top EV Markets in {year_selected}")
        st.caption("A few countries drive most EV demand in the selected year and region")
        fig_bar = make_top_countries_bar(df_filtered, year_selected, top_n=10)
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_top_markets")

    with table_col:
        st.subheader("Top 3 Markets")
        df_top3 = (
            df_filtered[df_filtered["Year"] == year_selected]
            .sort_values("EV_Sales", ascending=False)
            .head(3)
        )

        for _, row in df_top3.iterrows():
            st.metric(
                label=row["region_country"],
                value=f"{int(row['EV_Sales']):,}",
            )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#666; font-size:13px;'>"
        "EV Insights Dashboard • Created by Moukthika Gunapaneedu"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
