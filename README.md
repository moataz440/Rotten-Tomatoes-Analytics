# 🍅 Rotten Tomatoes Analytics

Exploratory data analysis of 17,000+ movies from the Rotten Tomatoes dataset — examining the gap between critic and audience scores, genre trends, studio performance, streaming windows, and more.

---

## 📊 Visualizations

| # | Chart | Question Answered |
|---|-------|-------------------|
| 01 | Critics vs Audience Scatter | Do critics and audiences agree? |
| 02 | Rating Distributions | How are Tomatometer & audience scores distributed? |
| 03 | Genre × Status Heatmap | Which genres dominate Fresh vs Rotten? |
| 04 | MPAA Rating Boxplot | Do R-rated films score differently than G/PG? |
| 05 | Critic–Audience Gap by Genre | Where do critics and audiences disagree the most? |
| 06 | Runtime Violin Plot | Do longer movies score better with critics? |
| 07 | Score Trends (1980–2022) | How have ratings evolved over the decades? |
| 08 | Top Studios by Avg Tomatometer | Which studios consistently make critically acclaimed films? |
| 09 | Pairplot | How do key numeric variables relate to each other? |
| 10 | Days to Streaming by Status | Do bad movies hit streaming faster? |

All charts are saved to the `visualizations/` folder and embedded in `Report.html`.

---

## 🔑 Key Findings

- **Critics and audiences moderately agree** — Pearson r ≈ 0.67, but significant divergence exists by genre.
- **Certified Fresh films** score ~20+ points higher with audiences than Rotten films.
- **Horror and Action** genres show the largest audience-over-critic gap (audiences are more generous).
- **Documentary and Drama** genres see critics rating significantly higher than audiences.
- **Rotten films** tend to reach streaming platforms faster than Fresh ones.
- Studio quality is consistent — the top 15 studios by Tomatometer all average above 60%.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `pandas` — data loading and cleaning
- `numpy` — numerical operations
- `matplotlib` — base plotting
- `seaborn` — statistical visualizations

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/moataz440/Rotten-Tomatoes-Analytics.git
cd Rotten-Tomatoes-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download the **Rotten Tomatoes Movies** dataset from [Maven Analytics Data Playground](https://mavenanalytics.io/data-playground) and place it at:

```
../rt_data/Rotten Tomatoes Movies.csv
```

Or update `RAW_PATH` in `analysis.py` to point to wherever you saved the file.

### 4. Run the analysis

```bash
python analysis.py
```

Charts will be saved to the `visualizations/` folder. Open `Report.html` in a browser to view the full report.

---

## 📁 Project Structure

```
Rotten-Tomatoes-Analytics/
│
├── analysis.py          # Main analysis script (load → clean → explore → visualize → insights)
├── Report.html          # Self-contained HTML report with all charts
├── visualizations/      # Output folder — all 10 PNG charts
└── README.md
```

---

## 📋 Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

---

## 📌 Dataset

**Source:** [Maven Analytics — Rotten Tomatoes Movies](https://mavenanalytics.io/data-playground)  
**Size:** ~17,000 movies  
**Columns include:** `movie_title`, `rating`, `genre`, `in_theaters_date`, `on_streaming_date`, `runtime_in_minutes`, `studio_name`, `directors`, `tomatometer_rating`, `tomatometer_status`, `tomatometer_count`, `audience_rating`, `audience_count`

---

## 👤 Author

**Moataz** — [github.com/moataz440](https://github.com/moataz440)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
