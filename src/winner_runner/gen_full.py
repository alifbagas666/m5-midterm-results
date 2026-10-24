import sys, os, numpy as np, pandas as pd
# usage: gen_full.py OUTDIR ITEMS_PER_STORE LAYOUT(kaggle|organiser) [SEED]
out, ips, layout = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)
os.makedirs(out, exist_ok=True)
D, DC = 1941, 1969
stores = {"CA_1":"CA","CA_2":"CA","CA_3":"CA","CA_4":"CA","TX_1":"TX","TX_2":"TX","TX_3":"TX","WI_1":"WI","WI_2":"WI","WI_3":"WI"}
depts = {"HOBBIES_1":"HOBBIES","HOBBIES_2":"HOBBIES","FOODS_1":"FOODS","FOODS_2":"FOODS","FOODS_3":"FOODS","HOUSEHOLD_1":"HOUSEHOLD","HOUSEHOLD_2":"HOUSEHOLD"}
dept_list = list(depts)
rows = []
for s, st in stores.items():
    for j in range(ips):
        dept = dept_list[j % len(dept_list)]
        rows.append(dict(item_id=f"{dept}_{j:03d}", dept_id=dept, cat_id=depts[dept], store_id=s, state_id=st))
meta = pd.DataFrame(rows)
lam = rng.uniform(0, 3, len(meta)) * (rng.uniform(0, 1, len(meta)) > 0.3)
sales = rng.poisson(lam[:, None], size=(len(meta), D)).astype(np.int64)
sales_df = pd.concat([meta, pd.DataFrame(sales, columns=[f"d_{k+1}" for k in range(D)])], axis=1)
if layout == "kaggle":
    sales_df = pd.concat([pd.DataFrame({"id": meta["item_id"] + "_" + meta["store_id"] + "_evaluation"}), sales_df], axis=1)
sales_df.to_csv(os.path.join(out, "sales_train_evaluation.csv"), index=False)
# calendar
dates = pd.date_range("2011-01-29", periods=DC)
cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                    "wm_yr_wk": 11101 + np.arange(DC) // 7,
                    "weekday": dates.day_name(), "wday": dates.dayofweek + 1,
                    "month": dates.month, "year": dates.year})
if layout == "kaggle":
    cal["d"] = [f"d_{i+1}" for i in range(DC)]
cal["event_name_1"] = pd.Series(np.where(rng.uniform(0,1,DC) > 0.97, "Holiday", None), dtype=object)
cal["event_type_1"] = pd.Series(np.where(cal["event_name_1"].notna(), "National", None), dtype=object)
cal["event_name_2"] = np.nan; cal["event_type_2"] = np.nan
cal["snap_CA"] = rng.integers(0,2,DC); cal["snap_TX"] = rng.integers(0,2,DC); cal["snap_WI"] = rng.integers(0,2,DC)
cal = cal[[c for c in ["date","wm_yr_wk","weekday","wday","month","year","d","event_name_1","event_type_1","event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"] if c in cal.columns]]
cal.to_csv(os.path.join(out, "calendar.csv"), index=False)
# prices: weekly, some series start later (release)
weeks = np.unique(cal["wm_yr_wk"])
pr = []
for i, r in meta.iterrows():
    start = rng.integers(0, len(weeks)//2) if rng.uniform() < 0.3 else 0
    base = rng.uniform(1, 20)
    p = np.round(base + rng.normal(0, 0.5, len(weeks)).cumsum() * 0.05, 2).clip(0.5)
    for w in range(start, len(weeks)):
        pr.append((r["store_id"], r["item_id"], weeks[w], p[w]))
pd.DataFrame(pr, columns=["store_id","item_id","wm_yr_wk","sell_price"]).to_csv(os.path.join(out, "sell_prices.csv"), index=False)
# sample submission (Kaggle layout)
ids = list(meta["item_id"] + "_" + meta["store_id"])
ss = pd.DataFrame({"id": [i + "_validation" for i in ids] + [i + "_evaluation" for i in ids]})
for k in range(1, 29): ss[f"F{k}"] = 0
ss.to_csv(os.path.join(out, "sample_submission.csv"), index=False)
print("generated", out, layout, len(meta), "series")
