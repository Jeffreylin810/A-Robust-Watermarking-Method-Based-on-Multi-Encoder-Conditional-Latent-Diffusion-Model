#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(
    "/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/efficiency_amortized_cost_experiment_20260624"
)
OUT = ROOT / "figures"
EXTRA_RGB_SECONDS = 914.26 + 3127.57


def hours(seconds: float) -> float:
    return seconds / 3600.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = [font_name]
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    training = pd.read_csv(ROOT / "training_timing_results.csv")
    measured_steps = [5000, 10000, 20000, 40000]

    pivot = training.pivot(index="num_steps", columns="method", values="total_seconds").sort_index()

    diffw_measured = [pivot.loc[s, "DiffW-like"] for s in measured_steps]
    base_measured = [pivot.loc[s, "O256-Base"] for s in measured_steps]
    rgb_measured = [pivot.loc[s, "O256-Base"] + EXTRA_RGB_SECONDS for s in measured_steps]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(
        measured_steps,
        [hours(v) for v in diffw_measured],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="#3b4cc0",
        label="DiffW-like（實測）",
    )
    ax.plot(
        measured_steps,
        [hours(v) for v in base_measured],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="#2f9e44",
        label="O256-Base（實測）",
    )
    ax.plot(
        measured_steps,
        [hours(v) for v in rgb_measured],
        marker="o",
        linewidth=2.5,
        markersize=6,
        color="#d9480f",
        label="O256-RGB（Base + 一次性修復成本）",
    )

    ax.set_xlabel("訓練步數", fontsize=12)
    ax.set_ylabel("訓練／準備時間（小時）", fontsize=12)
    ax.set_title("不同訓練步數下的實測時間成本比較", fontsize=14, pad=12)
    ax.grid(True, alpha=0.28)
    ax.set_ylim(bottom=0)
    ax.set_xticks(measured_steps)
    ax.set_xticklabels(["5k", "10k", "20k", "40k"])
    ax.legend(frameon=False, fontsize=10)

    note = (
        "註：O256-RGB 曲線為 O256-Base 實測訓練時間加上固定一次性成本\n"
        "（修復配對資料產生 00:15:14.26 + RGB 修復模組訓練 00:52:07.57）。"
    )
    fig.text(0.12, 0.02, note, ha="left", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    png = OUT / "fig_B_measured_cost_chinese_no_projection.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for s, d, b, r in zip(measured_steps, diffw_measured, base_measured, rgb_measured):
        rows.append(
            {
                "訓練步數": s,
                "DiffW-like_小時": hours(d),
                "O256-Base_小時": hours(b),
                "O256-RGB_小時": hours(r),
                "O256-RGB_minus_DiffW-like_小時": hours(r - d),
                "資料來源": "Experiment A 實測時間；O256-RGB 加上正式一次性修復成本",
            }
        )
    csv_path = OUT / "fig_B_measured_cost_chinese_no_projection_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md = OUT / "fig_B_measured_cost_chinese_no_projection_summary.md"
    md.write_text(
        "\n".join(
            [
                "# 中文實測時間成本圖（不含預測）",
                "",
                f"- 圖檔：`{png}`",
                f"- 資料：`{csv_path}`",
                "- 範圍：只包含 5k、10k、20k、40k 實測 timing benchmark。",
                "- 不包含：60k、80k、100k 預測資料。",
                "- O256-RGB：以 O256-Base 實測訓練時間加上固定一次性修復成本計算。",
                f"- 固定一次性修復成本：`{EXTRA_RGB_SECONDS:.2f}` 秒，約 `{hours(EXTRA_RGB_SECONDS):.4f}` 小時。",
                "",
                "建議 caption：",
                "",
                "> 不同訓練步數下之實測時間成本比較。O256-RGB 的時間包含 O256-Base 訓練時間，並額外加上一次性的修復配對資料產生與 RGB Refinement Module 訓練成本。本圖僅呈現 5k 至 40k 的實測 timing benchmark，不包含外推或預測值。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {png}")
    print(f"saved {csv_path}")
    print(f"saved {md}")


if __name__ == "__main__":
    main()
