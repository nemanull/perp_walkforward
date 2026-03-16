import lightgbm as lgb
import pandas as pd

H = 4

df = pd.read_csv("out/HYPE.csv")
df = df.dropna(subset=[f"x_fwd_logret_{H}", "x_roll_vol_logret_576"]).reset_index(drop=True)
month = pd.to_datetime(df["close_time"], unit="ms").dt.strftime("%Y-%m")

targets = [c for c in df.columns if c.startswith(("x_fwd_logret_", "x_sigma_", "x_z_", "x_score_"))]
X = df.drop(columns=["open_time", "close_time"] + targets)
y = (df[f"x_fwd_logret_{H}"] > 0).astype(int)

results = []
for m in sorted(month.unique())[2:]:
    train = month < m
    test = month == m
    # TODO the last H rows before m have labels that end inside m
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    model.fit(X[train], y[train])
    acc = (model.predict(X[test]) == y[test]).mean()
    print(m, train.sum(), test.sum(), round(acc, 4))
    # put it here

# pd.DataFrame(results).to_csv("out/walkforward.csv", index=False)
