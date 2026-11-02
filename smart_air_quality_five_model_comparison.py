# Smart Air Quality Forecasting, Health Risk Assessment & Pollution Alert System using Machine Learning
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# LOAD DATASET
df = pd.read_csv(r"C:\Users\Sachin Kumar\Downloads\archive (1)\city_day.csv")

print("Original Shape:", df.shape)
print("\nOriginal Data Summary:")
print(df.describe())

# DATA PREPROCESSING
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Bengaluru"]
df = df[df["City"].isin(cities)].copy()

df = df[["City", "Date", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]]
df.columns = ["city", "date", "pm2_5", "pm10", "co", "no2", "so2", "o3"]

df = df.dropna(subset=["date", "pm2_5"]).copy()

numeric_cols = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df = df.sort_values(["city", "date"]).reset_index(drop=True)

print("Cleaned Shape:", df.shape)

# FEATURE ENGINEERING
df["month"] = df["date"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Post-Monsoon"

df["Season"] = df["month"].apply(get_season)

pollutants = ["pm10", "pm2_5", "co", "no2", "so2", "o3"]

# BEFORE OUTLIER REMOVAL
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="city", y="pm2_5")
plt.title("PM2.5 BEFORE Outlier Removal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# REMOVE HIGH PM2.5 OUTLIERS
df_clean = df[df["pm2_5"] < 300].copy()
print("Before Removal:", df.shape)
print("After Removal:", df_clean.shape)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x="city", y="pm2_5")
plt.title("PM2.5 AFTER Outlier Removal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df = df_clean.copy()

# VISUALIZATION
plt.figure(figsize=(18, 10))
for i, col in enumerate(pollutants, 1):
    plt.subplot(2, 3, i)
    df.groupby("city")[col].mean().plot(kind="bar")
    plt.title(f"{col.upper()} by City")
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 10))
for i, col in enumerate(pollutants, 1):
    plt.subplot(2, 3, i)
    df.groupby("month")[col].mean().plot(marker="o")
    plt.title(f"{col.upper()} Monthly Trend")
    plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[pollutants].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# AQI / PM2.5 CATEGORY ANALYSIS
# These are simplified project categories based on PM2.5, not official CPCB AQI calculation.
def aqi_category(pm):
    if pm <= 50:
        return "Good"
    elif pm <= 100:
        return "Satisfactory"
    elif pm <= 200:
        return "Moderate"
    elif pm <= 300:
        return "Poor"
    elif pm <= 400:
        return "Very Poor"
    return "Severe"

def health_risk(pm):
    if pm <= 50:
        return "Low Risk"
    elif pm <= 100:
        return "Medium Risk"
    elif pm <= 200:
        return "High Risk"
    return "Very High Risk"

df["AQI_Category"] = df["pm2_5"].apply(aqi_category)
df["Health_Risk"] = df["pm2_5"].apply(health_risk)

# POLLUTION SCORE
df["pollution_score"] = (
    0.40 * df["pm2_5"]
    + 0.30 * df["pm10"]
    + 0.10 * (df["co"] * 100)
    + 0.10 * df["no2"]
    + 0.05 * df["so2"]
    + 0.05 * df["o3"]
)

print("\nTop polluted records:")
print(df[["date", "city", "pm2_5"]].sort_values("pm2_5", ascending=False).head())

print("\nAQI Category")
print(df["AQI_Category"].value_counts())

aqi_counts = df["AQI_Category"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(aqi_counts, labels=aqi_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("PM2.5 Category Distribution")
plt.tight_layout()
plt.show()

# POLLUTION HOTSPOT RANKING
print("\n===== POLLUTION HOTSPOT RANKING =====")
city_rank = df.groupby("city")["pollution_score"].mean().sort_values(ascending=False)
print(city_rank)

plt.figure(figsize=(8, 5))
city_rank.plot(kind="bar")
plt.title("Pollution Hotspot Ranking")
plt.ylabel("Average Pollution Score")
plt.xlabel("City")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# MOST POLLUTED DAY
top_day = df.loc[df["pm2_5"].idxmax()]
print("\n===== MOST POLLUTED DAY =====")
print("Date:", top_day["date"].date())
print("City:", top_day["city"])
print("PM2.5:", round(top_day["pm2_5"], 2))
print("PM2.5 Category:", top_day["AQI_Category"])

# SEASON-WISE ANALYSIS
season_analysis = df.groupby("Season")["pm2_5"].mean().sort_values(ascending=False)
print("\n===== SEASON-WISE POLLUTION =====")
print(season_analysis)

plt.figure(figsize=(8, 5))
season_analysis.plot(kind="bar")
plt.title("Season-wise Average PM2.5")
plt.ylabel("Average PM2.5")
plt.xlabel("Season")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nStatistical Summary (PM2.5):")
print(df["pm2_5"].describe())

# HYPOTHESIS TESTING
# 12. ONE-SAMPLE T-TEST
print("\n===== ONE-SAMPLE T-TEST =====")
threshold = 50.0
alpha = 0.05
sample = df["pm2_5"].dropna()

# H0: mean PM2.5 = 50; H1: mean PM2.5 > 50.
t_stat, p_value_two_sided = stats.ttest_1samp(sample, popmean=threshold)
p_value_one_sided = (
    p_value_two_sided / 2 if t_stat > 0
    else 1 - (p_value_two_sided / 2)
)

print("H0: Mean PM2.5 equals the selected threshold of 50.")
print("H1: Mean PM2.5 is greater than 50.")
print("Threshold:", threshold)
print("t-statistic:", round(float(t_stat), 4))
print("One-tailed p-value:", f"{p_value_one_sided:.10g}")
print("Significance level (alpha):", alpha)

if t_stat > 0 and p_value_one_sided < alpha:
    print("Result: Reject H0; evidence suggests the mean PM2.5 is above 50.")
else:
    print("Result: Fail to reject H0; insufficient evidence that mean PM2.5 is above 50.")

print("Note: This statistical test uses a project-selected threshold; it is not an official AQI determination.")


# MACHINE LEARNING: COMPARE FIVE REGRESSION MODELS
# Target: PM2.5
# Features: PM10, CO, NO2, SO2, O3, and month
feature_cols = ["pm10", "co", "no2", "so2", "o3", "month"]
X = df[feature_cols].copy()
y = df["pm2_5"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Multiple Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "Polynomial Regression (Degree 2)": Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=8, random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=1
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
    )
}

results = []
trained_models = {}

for model_name, regressor in models.items():
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    results.append({
        "Model": model_name,
        "R² Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })
    trained_models[model_name] = regressor

comparison = pd.DataFrame(results).sort_values(
    by="R² Score", ascending=False
).reset_index(drop=True)

print("\n" + "=" * 88)
print("===== FIVE-MODEL PERFORMANCE COMPARISON =====")
print("=" * 88)
print(comparison.to_string(
    index=False,
    formatters={
        "R² Score": "{:.4f}".format,
        "MAE": "{:.4f}".format,
        "MSE": "{:.4f}".format,
        "RMSE": "{:.4f}".format
    }
))

best_model_name = comparison.loc[0, "Model"]
best_model = trained_models[best_model_name]

print("\n===== BEST MODEL ON THIS TEST SPLIT =====")
print("Model:", best_model_name)
print("R² Score:", round(comparison.loc[0, "R² Score"], 4))
print("MAE:", round(comparison.loc[0, "MAE"], 4))
print("MSE:", round(comparison.loc[0, "MSE"], 4))
print("RMSE:", round(comparison.loc[0, "RMSE"], 4))

# MODEL COMPARISON GRAPHS
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison, x="R² Score", y="Model")
plt.title("Model Comparison - R² Score (Higher is Better)")
plt.xlim(min(-0.1, comparison["R² Score"].min() - 0.05), 1)
plt.tight_layout()
plt.show()

metric_data = comparison.set_index("Model")[["MAE", "RMSE"]]
metric_data.plot(kind="bar", figsize=(12, 6))
plt.title("Model Comparison - MAE and RMSE (Lower is Better)")
plt.ylabel("Error")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=comparison, x="Model", y="MSE")
plt.title("Model Comparison - MSE (Lower is Better)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# ACTUAL VS PREDICTED FOR THE BEST MODEL
best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_predictions, alpha=0.6)
min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title(f"Actual vs Predicted PM2.5 - {best_model_name}")
plt.tight_layout()
plt.show()

# USER INPUT PREDICTION USING THE BEST MODEL
print("\n===== PREDICT PM2.5 FROM USER INPUT =====")
print("Prediction uses the best-performing model from this comparison.")
print("Enter pollutant measurements in units consistent with the dataset.")

try:
    pm10_input = float(input("Enter PM10 (0-500): "))
    co_input = float(input("Enter CO (0-5): "))
    no2_input = float(input("Enter NO2 (0-200): "))
    so2_input = float(input("Enter SO2 (0-100): "))
    o3_input = float(input("Enter O3 (0-200): "))
    month_input = int(input("Enter Month (1-12): "))

    if not (
        0 <= pm10_input <= 500
        and 0 <= co_input <= 5
        and 0 <= no2_input <= 200
        and 0 <= so2_input <= 100
        and 0 <= o3_input <= 200
        and 1 <= month_input <= 12
    ):
        print("\nInvalid input! Please enter values within the specified ranges.")
    else:
        user_data = pd.DataFrame([{
            "pm10": pm10_input,
            "co": co_input,
            "no2": no2_input,
            "so2": so2_input,
            "o3": o3_input,
            "month": month_input
        }], columns=feature_cols)

        prediction = float(best_model.predict(user_data)[0])
        pred_pm = max(0.0, round(prediction, 2))

        print("\n===== PREDICTION RESULT =====")
        print("Selected Model:", best_model_name)
        print("Predicted PM2.5:", pred_pm)
        print("PM2.5 Category:", aqi_category(pred_pm))
        print("Health Risk:", health_risk(pred_pm))

        print("\n===== AIR QUALITY ALERT =====")
        if pred_pm <= 50:
            print("Lower PM2.5 category based on this project's simplified thresholds.")
        elif pred_pm <= 100:
            print("Moderate PM2.5 alert.")
        elif pred_pm <= 200:
            print("High PM2.5 alert.")
        else:
            print("Very high PM2.5 alert.")

        print("\n===== HEALTH RECOMMENDATIONS =====")
        if pred_pm <= 50:
            print("- Air pollution level is relatively lower by this project's threshold.")
            print("- Check local official air-quality guidance before outdoor activity.")
        elif pred_pm <= 100:
            print("- Consider reducing prolonged strenuous outdoor activity if sensitive.")
            print("- Check local official air-quality guidance.")
        elif pred_pm <= 200:
            print("- Reduce prolonged or heavy outdoor activity.")
            print("- Sensitive people should take extra care and follow local health guidance.")
        else:
            print("- Avoid strenuous outdoor activity where possible.")
            print("- Follow official local air-quality and public-health guidance.")

except ValueError:
    print("Invalid input. Please enter numeric values only.")
