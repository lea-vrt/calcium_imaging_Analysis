import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler #SCALING MEAN 0 STD = 1
from sklearn.impute import SimpleImputer #FILL MISSING VALUES
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold # PREVENT SAME RECORDING TO APPEAR IN BOTH TRAINING AND TESTING
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib # TRAINED MODEL 

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_OUT   = os.path.abspath("suite2p_output")
MODEL_DIR  = os.path.abspath("models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(MODEL_DIR, "Healthy_vs_SMA_best.joblib")
THRESH_PATH = os.path.join(MODEL_DIR, "decision_threshold.txt")
SCHEMA_PATH = os.path.join(MODEL_DIR, "recording_level_columns.json") #LIST OF FEATURES USED IN TRAINING
TEST_SIZE   = 0.30
RANDOM_STATE= 42

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2) #SAME STRUCTURE FOR THE NEW DATA

def build_master_and_rec():
    rows=[]
    for d in sorted(os.listdir(BASE_OUT)):
        plane0=os.path.join(BASE_OUT,d,"suite2p","plane0")
        fcsv=os.path.join(plane0,"calcium_features_suite2p_refined.csv")
        if not os.path.isfile(fcsv): continue
        df=pd.read_csv(fcsv)
        df["recording_id"]=d
        lab="Healthy" if "healthy" in d.lower() else ("SMA" if "sma" in d.lower() else None)
        df["disease"]=lab
        rows.append(df)
    if not rows: raise SystemExit("No features found in suite2p_output/*/suite2p/plane0/")
    master=pd.concat(rows,ignore_index=True)
    master.to_csv("MASTER_cells_Healthy_vs_SMA.csv",index=False)
    drop=["recording_id","disease"]
    feat=[c for c in master.columns if c not in drop]
    g=master.groupby(["recording_id","disease"],dropna=False)
    rec=pd.concat([
        g[feat].median().add_suffix("_med"),
        g[feat].mean().add_suffix("_mean"),
        (g[feat].quantile(0.75)-g[feat].quantile(0.25)).add_suffix("_iqr")
    ],axis=1).reset_index()
    rec=rec.dropna(axis=1,how="all")
    rec.to_csv("RECORDING_LEVEL.csv",index=False)
    schema=[c for c in rec.columns if c not in ["recording_id","disease"]]
    save_json(schema,SCHEMA_PATH)
    return rec

def split_train_test_recordings(REC, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    df=REC[REC["disease"].isin(["Healthy","SMA"])].copy()
    rec_ids=df["recording_id"].values
    labels =(df["disease"]=="SMA").astype(int).values
    sss=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
    (tr_idx, te_idx)=next(sss.split(rec_ids.reshape(-1,1), labels))
    train_ids=set(rec_ids[tr_idx]); test_ids=set(rec_ids[te_idx])
    train_df=df[df["recording_id"].isin(train_ids)].copy()
    test_df =df[df["recording_id"].isin(test_ids)].copy()
    return train_df, test_df

def build_pipelines():
    pipe_lr=Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler()),
        ("clf",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)),
    ])
    pipe_rf=Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("clf",RandomForestClassifier(n_estimators=500,random_state=42,class_weight="balanced")),
    ])
    return pipe_lr, pipe_rf

def optimal_threshold_from_cv(X,y,groups,model):
    gkf=GroupKFold(n_splits=min(len(np.unique(groups)),max(2,len(np.unique(groups)))))
    probs_all=[]; y_all=[]
    for tr,te in gkf.split(X,y,groups):
        Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]
        if len(np.unique(ytr))<2: continue
        model.fit(Xtr,ytr)
        probs_all.extend(model.predict_proba(Xte)[:,1]); y_all.extend(yte)
    fpr,tpr,thr=roc_curve(y_all,probs_all)
    j=tpr-fpr
    return float(thr[int(np.argmax(j))])

def pick_and_train_on_train(train_df):
    y=(train_df["disease"]=="SMA").astype(int).values
    X=train_df.drop(columns=["recording_id","disease"])
    groups=train_df["recording_id"].values
    pipe_lr, pipe_rf = build_pipelines()

    def cv_auc(model):
        gkf=GroupKFold(n_splits=min(len(np.unique(groups)),max(2,len(np.unique(groups)))))
        probs,trues=[],[]
        for tr,te in gkf.split(X,y,groups):
            Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]
            if len(np.unique(ytr))<2: continue
            model.fit(Xtr,ytr)
            p=model.predict_proba(Xte)[:,1]; probs.extend(p); trues.extend(yte)
        return roc_auc_score(trues,probs) if len(set(trues))>1 else np.nan

    auc_lr=cv_auc(pipe_lr); auc_rf=cv_auc(pipe_rf)
    best=pipe_rf if (auc_rf>=auc_lr) else pipe_lr
    best.fit(X,y)
    thr=optimal_threshold_from_cv(X,y,groups,best)
    joblib.dump(best,MODEL_PATH)
    with open(THRESH_PATH,"w") as f: f.write(str(thr))
    print(f"[Saved model] {MODEL_PATH}")
    print(f"[Saved threshold] {thr:.6f} → {THRESH_PATH}")
    return best, thr

def eval_on_test(model, thr, test_df):
    y_true=(test_df["disease"]=="SMA").astype(int).values
    X=test_df.drop(columns=["recording_id","disease"])
    probs=model.predict_proba(X)[:,1]
    preds=(probs>=thr).astype(int)
    auc=roc_auc_score(y_true,probs) if len(np.unique(y_true))>1 else np.nan
    cm=confusion_matrix(y_true,preds)
    rep=classification_report(y_true,preds,target_names=["Healthy","SMA"],digits=3,zero_division=0)
    return auc,cm,rep,probs
def save_and_print_test_predictions(model, thr, test_df, out_csv):
    X = test_df.drop(columns=["recording_id","disease"])
    probs = model.predict_proba(X)[:, 1]
    preds = np.where(probs >= thr, "SMA", "Healthy")
    out = test_df[["recording_id","disease"]].copy()
    out["prob_SMA"] = probs
    out["threshold_used"] = thr
    out["prediction"] = preds
    out.to_csv(out_csv, index=False)
    print("\n--- Per-recording test predictions ---")
    for r in out.itertuples(index=False):
        print(f"{r.recording_id}: true={r.disease:7s}  pred={r.prediction:7s}  prob_SMA={r.prob_SMA:.3f}  thr={r.threshold_used:.3f}")
    print(f"[Saved] {out_csv}")

    import numpy as np, pandas as pd, os
    X = test_df.drop(columns=["recording_id","disease"])
    probs = model.predict_proba(X)[:, 1]
    preds = np.where(probs >= thr, "SMA", "Healthy")
    out = test_df[["recording_id","disease"]].copy()
    out["prob_SMA"] = probs
    out["threshold_used"] = thr
    out["prediction"] = preds
    out.to_csv(out_csv, index=False)
    print("\n--- Per-recording test predictions ---")
    for r in out.itertuples(index=False):
        print(f"{r.recording_id}: true={r.disease:7s}  pred={r.prediction:7s}  prob_SMA={r.prob_SMA:.3f}  thr={r.threshold_used:.3f}")
    print(f"[Saved] {out_csv}")

def main():
    REC=build_master_and_rec()
    train_df, test_df = split_train_test_recordings(REC)
    print(f"[Split] train={len(train_df)} recordings, test={len(test_df)} recordings")
    print("[Train IDs]:", list(train_df["recording_id"].values))
    print("[Test  IDs]:", list(test_df["recording_id"].values))
    save_json([c for c in REC.columns if c not in ["recording_id","disease"]], SCHEMA_PATH)
    print(f"[Saved schema] {SCHEMA_PATH}")
    model, thr = pick_and_train_on_train(train_df)
    auc, cm, rep, probs = eval_on_test(model, thr, test_df)
    print("\n=== TEST SET EVAL (70/30 split) ===")
    print("ROC-AUC:", auc)
    print("Confusion:\n", cm)
    print(rep)

if __name__=="__main__":
    main()
