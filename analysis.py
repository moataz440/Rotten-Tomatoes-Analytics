"""
=============================================================================
Rotten Tomatoes Movies – Full Data Analytics Project
Dataset : Rotten Tomatoes Movies (Maven Analytics)
         https://mavenanalytics.io/data-playground
Author  : Moataz
Workflow: Load → Clean → Explore → Analyse → Visualise → Insights
=============================================================================
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotten Tomatoes full data analytics pipeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("..", "rt_data", "Rotten Tomatoes Movies.csv"),
        help="Path to the Rotten Tomatoes Movies CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="Folder where chart PNGs will be saved (default: visualizations/).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE    = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]
BG_COLOR   = "#0D1117"
CARD_COLOR = "#161B22"
TEXT_COLOR = "#E6EDF3"
ACCENT     = "#E63946"

sns.set_theme(
    style="dark",
    rc={
        "axes.facecolor":   CARD_COLOR,
        "figure.facecolor": BG_COLOR,
        "text.color":       TEXT_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "axes.edgecolor":   "#30363D",
        "grid.color":       "#21262D",
        "axes.grid":        True,
        "grid.linewidth":   0.5,
        "font.family":      "DejaVu Sans",
    },
)


def save(fig, viz_dir, name):
    path = os.path.join(viz_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print("=" * 65)
    print(" STEP 1 – LOAD DATA")
    print("=" * 65)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}\n"
            "Download it from https://mavenanalytics.io/data-playground\n"
            "Then pass the path with:  --data path/to/file.csv"
        )

    df = pd.read_csv(path)
    print(f"Raw shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nColumn overview:")
    print(df.dtypes.to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print(" STEP 2 – DATA CLEANING")
    print("=" * 65)

    df = df_raw.copy()

    # Fix malformed MPAA rating values
    df["rating"] = df["rating"].str.replace(r"[^A-Z0-9\-]", "", regex=True)
    df["rating"] = df["rating"].apply(
        lambda x: "PG13" if x in ("PG13", "PG-13") else x
    )
    valid_ratings = ["G", "PG", "PG13", "R", "NC17", "NR"]
    df = df[df["rating"].isin(valid_ratings)]

    # Parse dates
    df["in_theaters_date"]  = pd.to_datetime(df["in_theaters_date"],  errors="coerce")
    df["on_streaming_date"] = pd.to_datetime(df["on_streaming_date"], errors="coerce")
    df["release_year"]   = df["in_theaters_date"].dt.year
    df["streaming_year"] = df["on_streaming_date"].dt.year
    df["days_to_streaming"] = (
        df["on_streaming_date"] - df["in_theaters_date"]
    ).dt.days

    # Clip runtime outliers
    df = df[(df["runtime_in_minutes"] >= 40) & (df["runtime_in_minutes"] <= 300)]

    # Drop rows missing critical numeric columns
    df = df.dropna(subset=["tomatometer_rating", "audience_rating", "runtime_in_minutes"])

    # Primary genre (first listed)
    df["primary_genre"] = df["genre"].str.split(",").str[0].str.strip()

    # Fill remaining text NAs
    df["studio_name"] = df["studio_name"].fillna("Unknown Studio")
    df["directors"]   = df["directors"].fillna("Unknown")

    # Derived columns
    df["critic_audience_gap"] = df["tomatometer_rating"] - df["audience_rating"]
    df["audience_size"] = pd.cut(
        df["audience_count"],
        bins=[0, 5_000, 50_000, 200_000, df["audience_count"].max() + 1],
        labels=["Small", "Medium", "Large", "Blockbuster"],
    )

    print(f"Cleaned shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    key_cols = [
        "tomatometer_rating", "audience_rating",
        "runtime_in_minutes", "release_year", "rating", "primary_genre",
    ]
    print("\nNull counts after cleaning:")
    print(df[key_cols].isnull().sum().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────────────────────────────────────

def explore(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print(" STEP 3 – EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    print("\n--- Descriptive Statistics ---")
    print(
        df[["tomatometer_rating", "audience_rating",
            "runtime_in_minutes", "critic_audience_gap"]]
        .describe()
        .round(2)
    )

    print("\n--- Tomatometer Status Distribution ---")
    print(df["tomatometer_status"].value_counts())

    print("\n--- MPAA Rating Distribution ---")
    print(df["rating"].value_counts())

    print("\n--- Top 15 Primary Genres ---")
    print(df["primary_genre"].value_counts().head(15))

    print("\n--- Top 10 Studios by Movie Count ---")
    print(df["studio_name"].value_counts().head(10))

    num_cols = [
        "tomatometer_rating", "audience_rating",
        "runtime_in_minutes", "tomatometer_count", "critic_audience_gap",
    ]
    print("\n--- Correlation Matrix ---")
    print(df[num_cols].corr().round(3))


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def visualise(df: pd.DataFrame, viz_dir: str) -> list[str]:
    print("\n" + "=" * 65)
    print(" STEP 4 – ANALYSIS & VISUALISATION")
    print("=" * 65)

    os.makedirs(viz_dir, exist_ok=True)
    saved_paths: list[str] = []

    top_genres = df["primary_genre"].value_counts().head(12).index
    STATUS_PALETTE = {
        "Certified Fresh": "#2A9D8F",
        "Fresh":           "#E9C46A",
        "Rotten":          "#E63946",
    }

    # ── Chart 1: Critics vs Audience scatter ─────────────────────────────────
    print("\n[1] Critics vs Audience scatter …")
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    for status, grp in df.groupby("tomatometer_status"):
        ax.scatter(
            grp["tomatometer_rating"], grp["audience_rating"],
            color=STATUS_PALETTE[status], label=status,
            alpha=0.35, s=15, edgecolors="none",
        )
    ax.plot([0, 100], [0, 100], "--", color="#FFFFFF", linewidth=1.2,
            alpha=0.4, label="Perfect agreement")
    ax.set_xlabel("Tomatometer Rating (Critics %)", fontsize=12)
    ax.set_ylabel("Audience Rating (%)", fontsize=12)
    ax.set_title(
        "Critics vs Audience Scores\nDoes the crowd agree with the critics?",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15,
    )
    ax.legend(fontsize=10, facecolor=CARD_COLOR,
              edgecolor="#30363D", labelcolor=TEXT_COLOR)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    saved_paths.append(save(fig, viz_dir, "01_critics_vs_audience.png"))

    # ── Chart 2: Rating distributions ────────────────────────────────────────
    print("[2] Rating distributions …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(CARD_COLOR)
    sns.histplot(df["tomatometer_rating"], bins=40, color=ACCENT, alpha=0.85,
                 kde=True, line_kws={"linewidth": 2}, ax=axes[0])
    axes[0].set_title("Tomatometer Rating Distribution",
                      fontsize=13, fontweight="bold", color=TEXT_COLOR)
    axes[0].set_xlabel("Tomatometer Rating (%)")
    axes[0].set_ylabel("Count")
    sns.histplot(df["audience_rating"], bins=40, color="#457B9D", alpha=0.85,
                 kde=True, line_kws={"linewidth": 2}, ax=axes[1])
    axes[1].set_title("Audience Rating Distribution",
                      fontsize=13, fontweight="bold", color=TEXT_COLOR)
    axes[1].set_xlabel("Audience Rating (%)")
    axes[1].set_ylabel("")
    fig.suptitle("Rating Score Distributions", fontsize=15,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "02_rating_distributions.png"))

    # ── Chart 3: Genre × Status heatmap ──────────────────────────────────────
    print("[3] Genre × Status heatmap …")
    hm_df = (
        df[df["primary_genre"].isin(top_genres)]
        .groupby(["primary_genre", "tomatometer_status"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(CARD_COLOR)
    sns.heatmap(hm_df, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="#21262D",
                cbar_kws={"label": "Movie Count"}, ax=ax)
    ax.set_title(
        "Genre × Tomatometer Status\nWhich genres dominate Fresh vs Rotten?",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15,
    )
    ax.set_xlabel("Tomatometer Status", fontsize=11)
    ax.set_ylabel("Primary Genre", fontsize=11)
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "03_genre_status_heatmap.png"))

    # ── Chart 4: Boxplot – scores by MPAA rating ──────────────────────────────
    print("[4] Tomatometer by MPAA rating boxplot …")
    rating_order = ["G", "PG", "PG13", "R", "NC17", "NR"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(CARD_COLOR)
    for ax, col, title in zip(
        axes,
        ["tomatometer_rating", "audience_rating"],
        ["Critics Score by MPAA Rating", "Audience Score by MPAA Rating"],
    ):
        sns.boxplot(
            data=df[df["rating"].isin(rating_order)],
            x="rating", y=col, order=rating_order,
            palette=PALETTE, width=0.55, linewidth=1.2, fliersize=2, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.set_xlabel("MPAA Rating")
        ax.set_ylabel("Rating %")
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "04_mpaa_ratings_boxplot.png"))

    # ── Chart 5: Critic-audience gap by genre ─────────────────────────────────
    print("[5] Critic–audience gap by genre …")
    gap_df = (
        df[df["primary_genre"].isin(top_genres)]
        .groupby("primary_genre")["critic_audience_gap"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(CARD_COLOR)
    colors = ["#E63946" if v < 0 else "#2A9D8F" for v in gap_df.values]
    bars = ax.barh(gap_df.index, gap_df.values, color=colors, alpha=0.85)
    ax.axvline(0, color="#FFFFFF", linewidth=1.2, alpha=0.5)
    for bar, val in zip(bars, gap_df.values):
        ax.text(
            val + (0.5 if val >= 0 else -0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}", va="center",
            ha="left" if val >= 0 else "right",
            color=TEXT_COLOR, fontsize=9,
        )
    ax.set_title(
        "Critics vs Audience Gap by Genre\n"
        "(Positive = Critics more generous; Negative = Audience more generous)",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12,
    )
    ax.set_xlabel("Mean Score Gap (Critic% − Audience%)")
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "05_critic_audience_gap_genre.png"))

    # ── Chart 6: Runtime violin ───────────────────────────────────────────────
    print("[6] Runtime vs status violin …")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(CARD_COLOR)
    sns.violinplot(
        data=df, x="tomatometer_status", y="runtime_in_minutes",
        order=["Certified Fresh", "Fresh", "Rotten"],
        palette=["#2A9D8F", "#E9C46A", "#E63946"],
        inner="quartile", linewidth=1.2, ax=ax,
    )
    ax.set_title(
        "Movie Runtime by Tomatometer Status\nDo longer movies score better?",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12,
    )
    ax.set_xlabel("Tomatometer Status")
    ax.set_ylabel("Runtime (minutes)")
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "06_runtime_violin.png"))

    # ── Chart 7: Score trends over years ─────────────────────────────────────
    print("[7] Annual trends …")
    year_df = (
        df[(df["release_year"] >= 1980) & (df["release_year"] <= 2022)]
        .groupby("release_year")
        .agg(
            avg_critic=("tomatometer_rating", "mean"),
            avg_audience=("audience_rating", "mean"),
            count=("movie_title", "count"),
        )
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR); ax1.set_facecolor(CARD_COLOR)
    ax1.fill_between(year_df["release_year"], year_df["avg_critic"],
                     alpha=0.25, color=ACCENT)
    ax1.plot(year_df["release_year"], year_df["avg_critic"],
             color=ACCENT, linewidth=2.5, label="Avg Critic Score")
    ax1.fill_between(year_df["release_year"], year_df["avg_audience"],
                     alpha=0.20, color="#457B9D")
    ax1.plot(year_df["release_year"], year_df["avg_audience"],
             color="#457B9D", linewidth=2.5, label="Avg Audience Score")
    ax2 = ax1.twinx()
    ax2.bar(year_df["release_year"], year_df["count"],
            color="#E9C46A", alpha=0.25, label="Movie Count")
    ax2.set_ylabel("Number of Movies", color="#E9C46A", fontsize=11)
    ax2.tick_params(axis="y", colors="#E9C46A")
    ax1.set_title("Score Trends Over the Years (1980–2022)",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Average Rating (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=CARD_COLOR, edgecolor="#30363D", labelcolor=TEXT_COLOR)
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "07_score_trends.png"))

    # ── Chart 8: Top studios by avg Tomatometer ───────────────────────────────
    print("[8] Top studios …")
    studio_df = (
        df.groupby("studio_name")
        .agg(avg_critic=("tomatometer_rating", "mean"),
             movie_count=("movie_title", "count"))
        .query("movie_count >= 30")
        .sort_values("avg_critic", ascending=False)
        .head(15)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(CARD_COLOR)
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(studio_df))]
    bars = ax.barh(studio_df["studio_name"], studio_df["avg_critic"],
                   color=bar_colors, alpha=0.9)
    for bar, cnt in zip(bars, studio_df["movie_count"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={cnt}", va="center", fontsize=8, color=TEXT_COLOR,
        )
    ax.set_title("Top 15 Studios by Average Tomatometer Score (min 30 movies)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlabel("Average Tomatometer Rating (%)")
    ax.invert_yaxis()
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "08_top_studios.png"))

    # ── Chart 9: Pairplot ─────────────────────────────────────────────────────
    print("[9] Pair plot …")
    pair_df = df[["tomatometer_rating", "audience_rating",
                  "runtime_in_minutes", "tomatometer_status"]].dropna()
    g = sns.pairplot(
        pair_df, hue="tomatometer_status",
        hue_order=["Certified Fresh", "Fresh", "Rotten"],
        palette=["#2A9D8F", "#E9C46A", "#E63946"],
        plot_kws={"alpha": 0.4, "s": 15, "edgecolor": "none"},
        diag_kind="kde", diag_kws={"linewidth": 2},
    )
    g.figure.set_facecolor(BG_COLOR)
    for ax in g.axes.flatten():
        ax.set_facecolor(CARD_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
    g.figure.suptitle("Pairplot of Key Numeric Variables",
                      fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01)
    path = os.path.join(viz_dir, "09_pairplot.png")
    g.figure.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✓ Saved → {path}")
    saved_paths.append(path)

    # ── Chart 10: Days to streaming ───────────────────────────────────────────
    print("[10] Days to streaming by status …")
    stream_df = df.dropna(subset=["days_to_streaming"])
    stream_df = stream_df[
        (stream_df["days_to_streaming"] > 0) &
        (stream_df["days_to_streaming"] < 1500)
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(CARD_COLOR)
    sns.kdeplot(
        data=stream_df, x="days_to_streaming", hue="tomatometer_status",
        hue_order=["Certified Fresh", "Fresh", "Rotten"],
        palette=["#2A9D8F", "#E9C46A", "#E63946"],
        fill=True, alpha=0.35, linewidth=2, ax=ax,
    )
    ax.set_title(
        "Days from Theater to Streaming by Tomatometer Status\n"
        "Do bad movies hit streaming faster?",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12,
    )
    ax.set_xlabel("Days from Theater to Streaming")
    ax.set_ylabel("Density")
    plt.tight_layout()
    saved_paths.append(save(fig, viz_dir, "10_days_to_streaming.png"))

    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# 5. INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def print_insights(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print(" STEP 5 – KEY INSIGHTS")
    print("=" * 65)

    top_genres = df["primary_genre"].value_counts().head(12).index
    stream_df  = df.dropna(subset=["days_to_streaming"])
    stream_df  = stream_df[
        (stream_df["days_to_streaming"] > 0) &
        (stream_df["days_to_streaming"] < 1500)
    ]

    corr       = df["tomatometer_rating"].corr(df["audience_rating"])
    cert_avg   = df[df["tomatometer_status"] == "Certified Fresh"]["audience_rating"].mean()
    rotten_avg = df[df["tomatometer_status"] == "Rotten"]["audience_rating"].mean()
    median_days = stream_df.groupby("tomatometer_status")["days_to_streaming"].median()
    gap_by_genre = (
        df[df["primary_genre"].isin(top_genres)]
        .groupby("primary_genre")["critic_audience_gap"]
        .mean()
    )

    print(f"\n1. Critics & audiences agree moderately — Pearson r = {corr:.3f}")
    print(f"2. Certified Fresh films avg {cert_avg:.1f}% audience score"
          f" vs {rotten_avg:.1f}% for Rotten")
    print(f"3. Median days to streaming:\n{median_days.to_string()}")
    print(f"4. Genre with biggest critic-over-audience gap : "
          f"{gap_by_genre.idxmax()} ({gap_by_genre.max():.1f} pts)")
    print(f"5. Genre where audience outscores critics most : "
          f"{gap_by_genre.idxmin()} ({gap_by_genre.min():.1f} pts)")
    print(f"6. Total movies analysed : {len(df):,}")
    print(f"7. Year span             : "
          f"{df['release_year'].min():.0f} – {df['release_year'].max():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    df_raw = load_data(args.data)
    df     = clean_data(df_raw)
    explore(df)
    visualise(df, args.output)
    print_insights(df)
    print(f"\n✅ All charts saved to: {args.output}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
