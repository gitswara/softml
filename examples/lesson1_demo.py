
"""Run this after installing requirements.

Demo:
- Loads iris from sklearn
- Runs quick_eda
- Shows how to create your own CSV via interactive prompts
"""
import pandas as pd
from sklearn.datasets import load_iris
from softml.eda import quick_eda
from softml.datasets import create_dataset_interactive

iris = load_iris(as_frame=True)
df = iris.frame
print("Iris shape:", df.shape)
report = quick_eda(df, target="target", show_plots=True)
print("Correlation matrix (first 5 rows):")
print(report["corr"].head())

# Uncomment to try interactive dataset creation
# mydf = create_dataset_interactive("myclass.csv", fields=["name","hours_studied","score"], cast=True)
# print(mydf.head())
