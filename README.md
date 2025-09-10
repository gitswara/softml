
# softml

**Soft Tech Talks** teaching library — tiny wrappers so students can focus on concepts.

## Quickstart (Colab)

```python
!pip install git+https://github.com/gitswara/softml.git
import softml as sm

# Create a dataset interactively
df = sm.datasets.create_dataset_interactive("myclass.csv")

# Run quick EDA (with correlation matrix and plots)
report = sm.eda(df, target="score")
print(report["corr"])
```
# softml
