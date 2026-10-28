import pandas as pd

from src.charts import (
    map_chart,
    transfusion_distribution_charts,
    transfusion_target_chart,
    transfusion_feature_importance_chart,
    transfusion_scatter_chart,
)
from src.charts import blood_component_chart
import pandas as pd
from src.charts import (
    missing_value_chart,
    coord_coverage_chart,
)
from src.analytics import missing_value_summary

from src.charts import (
    state_bar_chart,
    district_bar_chart,
    category_donut_chart,
    service_time_chart,
)

from src.analytics import (
    state_distribution,
    district_distribution,
)

# Load your real, cleaned blood-bank data
df = pd.read_csv("data/cleaned/blood_banks_clean.csv")

# Choose which chart to display
print("1. Blood banks by state")
print("2. Blood banks by district")
print("3. Blood banks by category")
print("4. Service-time distribution")
print("5. Data completeness")
print("6. Coordinate coverage")
print("7. Blood component & apheresis availability")
print("8. Blood Bank Map")
print("9. Donor Feature Distributions")
print("10. Donation Outcome")
print("11. Donor Feature Importance")
print("12. Donor Recency vs Donation Outcome")
choice = input("Enter chart number: ")

if choice == "1":
    fig = state_bar_chart(state_distribution(df))

elif choice == "2":
    fig = district_bar_chart(district_distribution(df))

elif choice == "3":
    counts = df["category"].fillna("Unknown").value_counts()
    category_df = counts.rename_axis("category").reset_index(name="count")
    fig = category_donut_chart(category_df)

elif choice == "4":
    counts = df["service_time"].fillna("Unknown").value_counts()
    service_df = counts.rename_axis("service_time").reset_index(name="count")
    fig = service_time_chart(service_df)
elif choice == "5":
    mv_df = missing_value_summary(df)
    fig = missing_value_chart(mv_df)

elif choice == "6":
    fig = coord_coverage_chart(df)
elif choice == "7":
    comp_df = pd.DataFrame({
        "Feature": ["Blood Component", "Apheresis"],
        "Yes": [
            (df["blood_component"].astype(str).str.strip().str.lower() == "yes").sum(),
            (df["apheresis"].astype(str).str.strip().str.lower() == "yes").sum()
        ],
        "No": [
            (df["blood_component"].astype(str).str.strip().str.lower() == "no").sum(),
            (df["apheresis"].astype(str).str.strip().str.lower() == "no").sum()
        ],
        "Unknown": [
            df["blood_component"].isna().sum(),
            df["apheresis"].isna().sum()
        ]
    })
    fig = blood_component_chart(comp_df)
elif choice == "8":
    fig = map_chart(df)
    

elif choice == "9":
    donor_df = pd.read_csv("data/cleaned/transfusion_clean.csv")
    fig = transfusion_distribution_charts(donor_df)
   

elif choice == "10":
    donor_df = pd.read_csv("data/cleaned/transfusion_clean.csv")
    fig = transfusion_target_chart(donor_df)
    
elif choice == "11":
    print("Feature importance requires a trained model.")
    print("We will connect this chart after checking the model outputs.")

elif choice == "12":
    donor_df = pd.read_csv("data/cleaned/transfusion_clean.csv")
    fig = transfusion_scatter_chart(donor_df)
    
else:
    print("Invalid choice!")
    raise SystemExit

fig.show(renderer="browser")
