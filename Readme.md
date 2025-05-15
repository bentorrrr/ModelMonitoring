# Airline Delay Classifier – Model Quality Report (Reference vs Current)

### Dataset split

Slice Filter Rows
Reference Month ≤ 6 5 000
Current Month > 6 5 000

The chronological split mimics production monitoring while keeping class balance via random sampling (seed = 42).

### Model

Gradient Boosting Classifier (scikit-learn, random_state=42) trained on the reference slice only and then evaluated on both slices.

### Quality metrics

Slice Accuracy Precision Recall F1
Reference 0.970 0.972 0.998 0.985
Current 0.977 0.977 1.000 0.988

### Key observations

Stable performance – all aggregate metrics differ by ≤ 0.7 pp between the two six-month windows, indicating no material performance drift.

Target imbalance remains – fewer than 2 % of flights are delayed > 15 min; high overall scores are driven by the dominant on-time class. Per-class metrics should be monitored to catch false-alarm rates.

Seasonal robustness – training on Jan-Jun generalises to Jul-Dec, suggesting the chosen features capture delay patterns consistently across the year.
