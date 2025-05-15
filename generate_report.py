import pandas as pd
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("DelayedFlights.csv").dropna()
df["month"]  = df["Month"]
df["Delayed"] = (df["ArrDelay"] > 15).astype(int)
features = ["DepDelay", "Distance", "AirTime", "month"]

ref_df = df[df["month"] <= 6].sample(5000, random_state=42)
cur_df = df[df["month"] >  6].sample(5000, random_state=42)

clf = GradientBoostingClassifier(random_state=42)
clf.fit(ref_df[features], ref_df["Delayed"])

ref_df = ref_df.assign(prediction=clf.predict(ref_df[features]))
cur_df = cur_df.assign(prediction=clf.predict(cur_df[features]))

definition = DataDefinition(
    classification=[BinaryClassification(
        target="Delayed",
        prediction_labels="prediction"
    )],
    numerical_columns=features
)

ref_ds = Dataset.from_pandas(ref_df, data_definition=definition)
cur_ds = Dataset.from_pandas(cur_df, data_definition=definition)

drift   = Report([DataDriftPreset()]).run(ref_ds, cur_ds)
quality = Report([ClassificationPreset()]).run(ref_ds, cur_ds)

drift.save_html("flight_data_drift_report.html")
quality.save_html("model_quality_report.html")
print("✅ Saved both HTML reports (new API)")
