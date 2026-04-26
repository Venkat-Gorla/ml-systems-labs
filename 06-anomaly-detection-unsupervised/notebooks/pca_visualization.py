import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("anomaly_results.csv")

print(df["AnomalyLabel"].value_counts())

plt.figure()

plt.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["AnomalyLabel"]
)

plt.colorbar(label="AnomalyLabel")

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Anomaly Detection (Isolation Forest)")

plt.savefig("anomaly_plot.png", dpi=300, bbox_inches="tight")

plt.show()
