import pandas as pd
import matplotlib.pyplot as plt

# Load data
close_corr = pd.read_csv("corr_close_avg.csv", index_col=0)
div_corr = pd.read_csv("corr_dividends_avg.csv", index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Close price correlations
colors = ["steelblue" if v >= 0 else "salmon" for v in close_corr["avg_spearman_corr"]]
ax1.barh(close_corr.index[::-1], close_corr["avg_spearman_corr"][::-1], color=colors[::-1])
ax1.set_xlabel("Average Spearman Correlation")
ax1.set_title("Correlation with Close Price (100 stocks)")
ax1.axvline(x=0, color="black", linewidth=0.5)

# Dividend correlations
colors = ["steelblue" if v >= 0 else "salmon" for v in div_corr["avg_spearman_corr"]]
ax2.barh(div_corr.index[::-1], div_corr["avg_spearman_corr"][::-1], color=colors[::-1])
ax2.set_xlabel("Average Spearman Correlation")
ax2.set_title("Correlation with Dividends (87 stocks)")
ax2.axvline(x=0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("correlation_charts.png", dpi=150)
plt.show()
