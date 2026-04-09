import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an HTML benchmark report.")
    parser.add_argument("--report_dir", required=True, help="Directory containing summary.csv")
    parser.add_argument("--title", default="NPR Benchmark Evaluation Report")
    return parser.parse_args()


def load_summary(summary_path):
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            "num_samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "roc_auc",
            "real_accuracy",
            "fake_accuracy",
            "tn",
            "fp",
            "fn",
            "tp",
        ]:
            row[key] = float(row[key]) if key not in {"num_samples", "tn", "fp", "fn", "tp"} else int(row[key])
    return rows


def metric_bar(value, label):
    width = max(0.0, min(100.0, value * 100.0))
    return (
        f'<div class="metric-row"><span>{html.escape(label)}</span>'
        f'<div class="metric-bar"><div class="metric-fill" style="width:{width:.1f}%"></div></div>'
        f'<strong>{value:.3f}</strong></div>'
    )


def confusion_matrix_svg(tn, fp, fn, tp):
    total = max(tn + fp + fn + tp, 1)

    def color(count):
        ratio = count / total
        blue = int(245 - ratio * 110)
        red = int(248 - ratio * 40)
        green = int(250 - ratio * 120)
        return f"rgb({red},{green},{blue})"

    cells = [
        ("TN", tn, 0, 0),
        ("FP", fp, 1, 0),
        ("FN", fn, 0, 1),
        ("TP", tp, 1, 1),
    ]
    rects = []
    for label, value, col, row in cells:
        x = 60 + col * 120
        y = 30 + row * 90
        rects.append(
            f'<rect x="{x}" y="{y}" width="110" height="80" rx="12" fill="{color(value)}" stroke="#cbd5e1" />'
            f'<text x="{x + 55}" y="{y + 28}" text-anchor="middle" class="cm-label">{label}</text>'
            f'<text x="{x + 55}" y="{y + 55}" text-anchor="middle" class="cm-value">{value}</text>'
        )
    return (
        '<svg viewBox="0 0 310 220" class="cm-svg" role="img" aria-label="Confusion matrix">'
        '<text x="115" y="18" text-anchor="middle" class="cm-axis">Pred Real</text>'
        '<text x="235" y="18" text-anchor="middle" class="cm-axis">Pred Fake</text>'
        '<text x="15" y="80" text-anchor="middle" class="cm-axis" transform="rotate(-90 15 80)">Actual Real</text>'
        '<text x="15" y="170" text-anchor="middle" class="cm-axis" transform="rotate(-90 15 170)">Actual Fake</text>'
        + "".join(rects)
        + "</svg>"
    )


def benchmark_aggregates(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)

    aggregates = []
    for benchmark, items in grouped.items():
        count = len(items)
        aggregates.append(
            {
                "benchmark": benchmark,
                "mean_accuracy": sum(r["accuracy"] for r in items) / count,
                "mean_ap": sum(r["average_precision"] for r in items) / count,
                "mean_f1": sum(r["f1"] for r in items) / count,
                "mean_auc": sum(r["roc_auc"] for r in items) / count,
                "count": count,
            }
        )
    return sorted(aggregates, key=lambda item: item["benchmark"])


def top_bottom_sections(rows):
    top_acc = sorted(rows, key=lambda row: (-row["accuracy"], row["benchmark"], row["subset"]))[:5]
    top_auc = sorted(rows, key=lambda row: (-row["roc_auc"], row["benchmark"], row["subset"]))[:5]
    low_acc = sorted(rows, key=lambda row: (row["accuracy"], row["benchmark"], row["subset"]))[:5]
    low_auc = sorted(rows, key=lambda row: (row["roc_auc"], row["benchmark"], row["subset"]))[:5]
    return {
        "top_acc": top_acc,
        "top_auc": top_auc,
        "low_acc": low_acc,
        "low_auc": low_auc,
    }


def ranking_table(title, rows, metric_key):
    items = []
    for row in rows:
        items.append(
            "<tr>"
            f"<td>{html.escape(row['benchmark'])}</td>"
            f"<td>{html.escape(row['subset'])}</td>"
            f"<td>{row['accuracy']:.3f}</td>"
            f"<td>{row['average_precision']:.3f}</td>"
            f"<td>{row['f1']:.3f}</td>"
            f"<td>{row[metric_key]:.3f}</td>"
            "</tr>"
        )
    return f"""
    <article class="ranking-card">
      <h3>{html.escape(title)}</h3>
      <table class="rank-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>Subset</th>
            <th>Acc</th>
            <th>AP</th>
            <th>F1</th>
            <th>{html.escape(metric_key.upper())}</th>
          </tr>
        </thead>
        <tbody>
          {''.join(items)}
        </tbody>
      </table>
    </article>
    """


def benchmark_chart_svg(aggregates):
    if not aggregates:
        return ""
    left = 180
    width = 760
    top = 34
    row_gap = 42
    svg_height = top + len(aggregates) * row_gap + 24
    bars = []
    labels = []
    ticks = []
    for index, item in enumerate(aggregates):
        y = top + index * row_gap
        labels.append(
            f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" class="chart-label">{html.escape(item["benchmark"])}</text>'
        )
        for offset, (field, fill) in enumerate(
            [
                ("mean_accuracy", "#92400e"),
                ("mean_ap", "#b45309"),
                ("mean_f1", "#d97706"),
                ("mean_auc", "#f59e0b"),
            ]
        ):
            bars.append(
                f'<rect x="{left}" y="{y + offset * 7}" width="{item[field] * width:.1f}" height="6" rx="3" fill="{fill}" />'
            )
        bars.append(
            f'<text x="{left + width + 10}" y="{y + 16}" class="chart-value">{item["mean_accuracy"]:.3f}</text>'
        )
    for tick in range(0, 11):
        x = left + width * (tick / 10)
        ticks.append(f'<line x1="{x}" y1="18" x2="{x}" y2="{svg_height - 14}" class="chart-grid" />')
        ticks.append(f'<text x="{x}" y="14" text-anchor="middle" class="chart-tick">{tick / 10:.1f}</text>')
    legend = """
      <g transform="translate(180, 8)">
        <rect width="10" height="10" rx="2" fill="#92400e"></rect><text x="16" y="9" class="chart-tick">Acc</text>
        <rect x="56" width="10" height="10" rx="2" fill="#b45309"></rect><text x="72" y="9" class="chart-tick">AP</text>
        <rect x="112" width="10" height="10" rx="2" fill="#d97706"></rect><text x="128" y="9" class="chart-tick">F1</text>
        <rect x="168" width="10" height="10" rx="2" fill="#f59e0b"></rect><text x="184" y="9" class="chart-tick">AUC</text>
      </g>
    """
    return (
        f'<svg viewBox="0 0 {left + width + 80} {svg_height}" class="bench-chart" role="img" aria-label="Benchmark comparison chart">'
        + "".join(ticks)
        + legend
        + "".join(labels)
        + "".join(bars)
        + "</svg>"
    )


def subset_card(report_dir, row):
    dataset_dir = report_dir / row["dataset"]
    xai_dir = dataset_dir / "xai"
    xai_images = sorted(xai_dir.glob("*.jpg"))[:6] if xai_dir.exists() else []
    gallery_html = ""
    if xai_images:
        gallery_html = '<div class="gallery">' + "".join(
            f'<a href="{html.escape(x.relative_to(report_dir).as_posix())}" target="_blank">'
            f'<img src="{html.escape(x.relative_to(report_dir).as_posix())}" alt="{html.escape(x.name)}"></a>'
            for x in xai_images
        ) + "</div>"
    else:
        gallery_html = '<p class="muted">No XAI images found for this subset.</p>'

    prediction_rel = (dataset_dir / "predictions.csv").relative_to(report_dir).as_posix()
    root_short = html.escape(row["dataset_root"])
    return f"""
    <details class="subset-card">
      <summary>
        <div>
          <h3>{html.escape(row['subset'])}</h3>
          <p>{html.escape(row['benchmark'])}</p>
        </div>
        <div class="summary-metrics">
          <span>Acc {row['accuracy']:.3f}</span>
          <span>AP {row['average_precision']:.3f}</span>
          <span>F1 {row['f1']:.3f}</span>
        </div>
      </summary>
      <div class="subset-grid">
        <section>
          <h4>Metrics</h4>
          {metric_bar(row['accuracy'], 'Accuracy')}
          {metric_bar(row['precision'], 'Precision')}
          {metric_bar(row['recall'], 'Recall')}
          {metric_bar(row['f1'], 'F1')}
          {metric_bar(row['average_precision'], 'Average Precision')}
          {metric_bar(row['roc_auc'], 'ROC-AUC')}
          <div class="meta-grid">
            <div><span>Samples</span><strong>{row['num_samples']}</strong></div>
            <div><span>Real Acc</span><strong>{row['real_accuracy']:.3f}</strong></div>
            <div><span>Fake Acc</span><strong>{row['fake_accuracy']:.3f}</strong></div>
            <div><span>Predictions</span><a href="{html.escape(prediction_rel)}" target="_blank">CSV</a></div>
          </div>
          <p class="dataset-root">{root_short}</p>
        </section>
        <section>
          <h4>Confusion Matrix</h4>
          {confusion_matrix_svg(row['tn'], row['fp'], row['fn'], row['tp'])}
        </section>
      </div>
      <section>
        <h4>XAI Samples</h4>
        <p class="muted">Mỗi ảnh gồm: ảnh gốc, heatmap, và ảnh overlay Grad-CAM.</p>
        {gallery_html}
      </section>
    </details>
    """


def build_xai_gallery_page(report_dir, title, rows):
    sections = []
    by_benchmark = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)

    for benchmark in sorted(by_benchmark):
        cards = []
        for row in sorted(by_benchmark[benchmark], key=lambda item: item["subset"]):
            xai_dir = report_dir / row["dataset"] / "xai"
            xai_images = sorted(xai_dir.glob("*.jpg"))
            cards.append(
                f"""
                <article class="xai-card">
                  <div class="xai-head">
                    <h3>{html.escape(row['subset'])}</h3>
                    <p>Acc {row['accuracy']:.3f} | AP {row['average_precision']:.3f} | F1 {row['f1']:.3f}</p>
                  </div>
                  <div class="xai-grid">
                    {''.join(
                        f'<a href="{html.escape(image.relative_to(report_dir).as_posix())}" target="_blank">'
                        f'<img src="{html.escape(image.relative_to(report_dir).as_posix())}" alt="{html.escape(image.name)}"></a>'
                        for image in xai_images
                    ) if xai_images else '<p class="muted">No XAI images found.</p>'}
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="gallery-section">
              <h2>{html.escape(benchmark)}</h2>
              <div class="gallery-stack">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - XAI Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f4efe7;
      color: #1f2937;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 22px 60px;
    }}
    .hero {{
      background: linear-gradient(135deg, #7c2d12, #b45309);
      color: #fff;
      border-radius: 24px;
      padding: 24px 28px;
    }}
    .hero a {{
      color: #fde68a;
      text-decoration: none;
      font-weight: 700;
    }}
    .gallery-section {{
      margin-top: 26px;
    }}
    .gallery-stack {{
      display: grid;
      gap: 16px;
    }}
    .xai-card {{
      background: #fffdf8;
      border: 1px solid #e5dccf;
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 16px 40px rgba(77, 48, 19, 0.08);
    }}
    .xai-head {{
      padding: 18px 20px 6px;
    }}
    .xai-head h3 {{
      margin: 0 0 4px;
    }}
    .xai-head p {{
      margin: 0;
      color: #6b7280;
    }}
    .xai-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      padding: 16px 20px 20px;
    }}
    .xai-grid a {{
      display: block;
      border: 1px solid #e5dccf;
      border-radius: 16px;
      overflow: hidden;
      background: #faf5eb;
    }}
    .xai-grid img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .muted {{
      color: #6b7280;
      margin: 0;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)} - XAI Gallery</h1>
      <p>Xem nhanh tất cả ảnh Grad-CAM theo từng subset. <a href="report.html">Mo report tong hop</a></p>
    </section>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def build_html(report_dir, title, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)

    aggregates = benchmark_aggregates(rows)
    highlight = top_bottom_sections(rows)

    aggregate_cards = []
    for item in aggregates:
        aggregate_cards.append(
            f"""
            <article class="benchmark-chip">
              <h3>{html.escape(item['benchmark'])}</h3>
              <p>{item['count']} subsets</p>
              {metric_bar(item['mean_accuracy'], 'Mean Accuracy')}
              {metric_bar(item['mean_ap'], 'Mean AP')}
              {metric_bar(item['mean_f1'], 'Mean F1')}
              {metric_bar(item['mean_auc'], 'Mean ROC-AUC')}
            </article>
            """
        )

    benchmark_sections = []
    for benchmark in sorted(grouped):
        items = sorted(grouped[benchmark], key=lambda row: row["subset"])
        benchmark_sections.append(
            f"""
            <section class="benchmark-section">
              <div class="section-head">
                <h2>{html.escape(benchmark)}</h2>
                <span>{len(items)} subsets</span>
              </div>
              <div class="subset-list">
                {''.join(subset_card(report_dir, row) for row in items)}
              </div>
            </section>
            """
        )

    payload = {
        "title": title,
        "generated_from": str((report_dir / "summary.csv").resolve()),
        "num_subsets": len(rows),
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --card: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5dccf;
      --accent: #b45309;
      --accent-soft: #f59e0b;
      --shadow: 0 16px 40px rgba(77, 48, 19, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 24%),
        linear-gradient(180deg, #f7f1e7 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 24px 72px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(180,83,9,0.96), rgba(120,53,15,0.92));
      color: white;
      border-radius: 28px;
      padding: 28px 32px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3.4rem);
    }}
    .hero p {{
      margin: 0;
      max-width: 900px;
      line-height: 1.55;
      color: rgba(255,255,255,0.9);
    }}
    .hero a {{
      color: #fde68a;
      text-decoration: none;
      font-weight: 700;
    }}
    .aggregate-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin: 26px 0 24px;
    }}
    .panel, .benchmark-chip, .subset-card, .ranking-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .panel {{
      padding: 18px 20px 22px;
      margin-bottom: 18px;
    }}
    .benchmark-chip {{
      padding: 18px;
    }}
    .benchmark-chip h3 {{
      margin: 0 0 4px;
    }}
    .benchmark-chip p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .ranking-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .ranking-card {{
      padding: 16px;
    }}
    .ranking-card h3 {{
      margin: 0 0 12px;
    }}
    .rank-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    .rank-table th, .rank-table td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid #eee5d8;
    }}
    .rank-table th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .bench-chart {{
      width: 100%;
      display: block;
      background: #fcfaf6;
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .chart-grid {{
      stroke: #eadfce;
      stroke-width: 1;
    }}
    .chart-label {{
      font-size: 13px;
      fill: #334155;
      font-weight: 700;
    }}
    .chart-tick {{
      font-size: 12px;
      fill: #64748b;
    }}
    .chart-value {{
      font-size: 12px;
      fill: #92400e;
      font-weight: 700;
    }}
    .benchmark-section {{
      margin-top: 28px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 1.7rem;
    }}
    .section-head span {{
      color: var(--muted);
    }}
    .subset-list {{
      display: grid;
      gap: 16px;
    }}
    .subset-card {{
      overflow: hidden;
    }}
    .subset-card summary {{
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      cursor: pointer;
    }}
    .subset-card summary::-webkit-details-marker {{
      display: none;
    }}
    .subset-card h3 {{
      margin: 0 0 4px;
      font-size: 1.3rem;
    }}
    .subset-card p {{
      margin: 0;
      color: var(--muted);
    }}
    .summary-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }}
    .summary-metrics span {{
      background: #fff3dd;
      color: #92400e;
      border: 1px solid #f5d9a6;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.95rem;
      font-weight: 700;
    }}
    .subset-grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      padding: 0 22px 20px;
    }}
    .metric-row {{
      display: grid;
      grid-template-columns: 128px 1fr 64px;
      align-items: center;
      gap: 10px;
      margin: 8px 0;
      font-size: 0.95rem;
    }}
    .metric-bar {{
      height: 12px;
      background: #efe7da;
      border-radius: 999px;
      overflow: hidden;
    }}
    .metric-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent-soft), var(--accent));
      border-radius: inherit;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .meta-grid div {{
      background: #fbf7ef;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .meta-grid span {{
      display: block;
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 4px;
    }}
    .meta-grid strong, .meta-grid a {{
      color: var(--ink);
      font-weight: 700;
      text-decoration: none;
    }}
    .dataset-root {{
      margin-top: 12px;
      font-size: 0.88rem;
      color: var(--muted);
      word-break: break-all;
    }}
    .cm-svg {{
      width: 100%;
      max-width: 440px;
      background: #fcfaf6;
      border: 1px solid var(--line);
      border-radius: 18px;
      display: block;
    }}
    .cm-label {{
      font-size: 13px;
      fill: #475569;
      font-weight: 700;
    }}
    .cm-value {{
      font-size: 18px;
      fill: #0f172a;
      font-weight: 700;
    }}
    .cm-axis {{
      font-size: 12px;
      fill: #64748b;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      padding: 0 22px 22px;
    }}
    .gallery a {{
      display: block;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #f8f3ea;
    }}
    .gallery img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .muted {{
      color: var(--muted);
      padding: 0 22px 14px;
      margin: 0;
    }}
    code {{
      background: rgba(255,255,255,0.22);
      padding: 2px 6px;
      border-radius: 8px;
    }}
    @media (max-width: 960px) {{
      .subset-grid {{
        grid-template-columns: 1fr;
      }}
      .subset-card summary {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .summary-metrics {{
        justify-content: flex-start;
      }}
      .metric-row {{
        grid-template-columns: 105px 1fr 56px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>
        Bao gồm metric định lượng, ma trận nhầm lẫn, liên kết đến <code>predictions.csv</code>,
        ảnh XAI Grad-CAM, biểu đồ benchmark, và danh sách best or worst cases.
        <a href="xai_gallery.html">Mo XAI gallery rieng</a>
      </p>
    </section>
    <section class="aggregate-grid">
      {''.join(aggregate_cards)}
    </section>
    <section class="panel">
      <div class="section-head">
        <h2>Benchmark Comparison</h2>
        <span>Mean scores by benchmark</span>
      </div>
      {benchmark_chart_svg(aggregates)}
    </section>
    <section class="ranking-grid">
      {ranking_table('Top 5 by Accuracy', highlight['top_acc'], 'roc_auc')}
      {ranking_table('Top 5 by ROC-AUC', highlight['top_auc'], 'roc_auc')}
      {ranking_table('Lowest 5 by Accuracy', highlight['low_acc'], 'roc_auc')}
      {ranking_table('Lowest 5 by ROC-AUC', highlight['low_auc'], 'roc_auc')}
    </section>
    {''.join(benchmark_sections)}
  </main>
  <script type="application/json" id="report-meta">{html.escape(json.dumps(payload))}</script>
</body>
</html>
"""


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    summary_path = report_dir / "summary.csv"
    rows = load_summary(summary_path)
    html_text = build_html(report_dir, args.title, rows)
    output_path = report_dir / "report.html"
    output_path.write_text(html_text, encoding="utf-8")
    xai_gallery_path = report_dir / "xai_gallery.html"
    xai_gallery_path.write_text(build_xai_gallery_page(report_dir, args.title, rows), encoding="utf-8")
    print(f"Report written to {output_path.resolve()}")
    print(f"XAI gallery written to {xai_gallery_path.resolve()}")


if __name__ == "__main__":
    main()
