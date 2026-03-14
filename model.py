import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

H = 4  # 20 minutes

df = pd.read_csv("out/HYPE.csv")
df = df.dropna(subset=[f"x_fwd_logret_{H}", "x_roll_vol_logret_576"])
month = pd.to_datetime(df["close_time"], unit="ms").dt.strftime("%Y-%m")

targets = [c for c in df.columns if c.startswith(("x_fwd_logret_", "x_sigma_", "x_z_", "x_score_"))]
X = df.drop(columns=["open_time", "close_time"] + targets)
y = (df[f"x_fwd_logret_{H}"] > 0).astype(int)
print(X.shape, "up share", round(y.mean(), 4))

# shuffled split leaks: neighbouring rows share 3 of the 4 bars in the label
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
last = month.max()
test = month == last
X_train, X_test, y_train, y_test = X[~test], X[test], y[~test], y[test]
print("test month", last, "train rows", len(X_train), "test rows", len(X_test))

model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("train acc", round(accuracy_score(y_train, model.predict(X_train)), 4))
print("test acc", round(accuracy_score(y_test, pred), 4))

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(15))
