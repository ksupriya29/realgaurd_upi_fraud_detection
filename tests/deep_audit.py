import pickle, sys, io
import numpy as np
import pandas as pd
import shap

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load all artifacts
with open('C:/Users/Supriya/Desktop/rtrp/models/xgb_model.pkl','rb') as f: model = pickle.load(f)
with open('C:/Users/Supriya/Desktop/rtrp/models/encoders_v2.pkl','rb') as f: enc = pickle.load(f)
with open('C:/Users/Supriya/Desktop/rtrp/models/feature_names.pkl','rb') as f: feat = pickle.load(f)
with open('C:/Users/Supriya/Desktop/rtrp/models/model_metrics.pkl','rb') as f: metrics = pickle.load(f)

print('=== MODEL ===')
print('Type:', type(model).__name__)
print('classes_:', list(model.classes_))
print('n_features_in_:', model.n_features_in_)
print()

print('=== FEATURE NAMES (23) ===')
for i, f in enumerate(feat):
    print('  [%d] %s' % (i, f))
print()

print('=== ENCODERS v2 ===')
for k, v in enc.items():
    print('  %s: %s' % (k, list(v.classes_)))
print()

print('=== MODEL METRICS ===')
for k, v in metrics.items():
    if k not in ('roc_curve', 'pr_curve'):
        print('  %s: %s' % (k, v))
print()

# Helper: preprocess exactly as main.py does
def make_X(data):
    df = pd.DataFrame([data])
    for col in ['MerchantCategory', 'TransactionType', 'TransactionFrequency']:
        if col in enc:
            le = enc[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0])
    for col in ['UnusualLocation', 'UnusualAmount', 'NewDevice']:
        df[col] = df[col].astype(int)
    avg = float(data.get('AvgTransactionAmount', 1)) or 1.0
    amount = float(data['Amount'])
    df['AmountRatio']   = amount / avg
    df['IsLargeAmount'] = int(amount / avg > 3)
    df['IsNightTime']   = int(int(data.get('Hour', 12)) < 5)
    df['HighFailed']    = int(int(data.get('FailedAttempts', 0)) > 2)
    df['RiskFlagSum']   = (int(data.get('UnusualLocation',0)) + int(data.get('UnusualAmount',0)) +
                           int(data.get('NewDevice',0)) + df['HighFailed'].iloc[0])
    df['IsHighIPFreq']  = int(int(data.get('IP_Frequency', 0)) > 10)
    df['FreqIsHigh']    = int(str(data.get('TransactionFrequency','1/day')) == '5/day')
    return df.reindex(columns=feat, fill_value=0)

def compute_risk(raw, data):
    risk = float(raw)
    avg = float(data.get('AvgTransactionAmount', 1)) or 1.0
    ratio = float(data['Amount']) / avg
    if ratio > 3: risk += 0.20
    elif ratio > 2: risk += 0.10
    fa = int(data.get('FailedAttempts', 0))
    if fa > 2: risk += 0.15
    elif fa > 0: risk += 0.05
    if int(data.get('NewDevice',0)) == 1: risk += 0.10
    if int(data.get('UnusualLocation',0)) == 1: risk += 0.10
    if int(data.get('UnusualAmount',0)) == 1: risk += 0.05
    if int(data.get('Hour',12)) < 5: risk += 0.05
    if str(data.get('TransactionFrequency','')) == '5/day': risk += 0.05
    return min(risk, 1.0)

fraud_idx = list(model.classes_).index(1)

cases = {
    'A_legit':    {'Amount':500,  'MerchantCategory':'Groceries',   'TransactionType':'P2M','Latitude':12.97,'Longitude':77.59,'AvgTransactionAmount':1000, 'TransactionFrequency':'1/day','UnusualLocation':0,'UnusualAmount':0,'NewDevice':0,'FailedAttempts':0,'Hour':14,'DayOfWeek':1,'User_Transaction_Count':20,'IP_Frequency':2, 'Phone_Usage_Count':5},
    'B_susp':     {'Amount':8000, 'MerchantCategory':'Electronics',  'TransactionType':'P2P','Latitude':28.6, 'Longitude':77.2, 'AvgTransactionAmount':3000, 'TransactionFrequency':'3/day','UnusualLocation':1,'UnusualAmount':0,'NewDevice':1,'FailedAttempts':1,'Hour':23,'DayOfWeek':6,'User_Transaction_Count':40,'IP_Frequency':8, 'Phone_Usage_Count':3},
    'C_fraud':    {'Amount':25000,'MerchantCategory':'Electronics',  'TransactionType':'P2M','Latitude':19.2, 'Longitude':72.8, 'AvgTransactionAmount':2000, 'TransactionFrequency':'5/day','UnusualLocation':1,'UnusualAmount':1,'NewDevice':1,'FailedAttempts':5,'Hour':2, 'DayOfWeek':6,'User_Transaction_Count':60,'IP_Frequency':15,'Phone_Usage_Count':10},
    'D_boundary': {'Amount':3100, 'MerchantCategory':'Travel',       'TransactionType':'P2M','Latitude':22.5, 'Longitude':88.3, 'AvgTransactionAmount':1000, 'TransactionFrequency':'3/day','UnusualLocation':0,'UnusualAmount':1,'NewDevice':0,'FailedAttempts':2,'Hour':4, 'DayOfWeek':3,'User_Transaction_Count':10,'IP_Frequency':5, 'Phone_Usage_Count':2},
}

print('=== PREDICTIONS (all 4 scenarios) ===')
for name, data in cases.items():
    X = make_X(data)
    proba = model.predict_proba(X)[0]
    raw = float(proba[fraud_idx])
    final = compute_risk(raw, data)
    score = round(final * 100)
    level = 'HIGH' if final >= 0.65 else ('MEDIUM' if final >= 0.30 else 'LOW')
    is_fraud = final >= 0.50
    ratio = float(data['Amount']) / float(data['AvgTransactionAmount'])
    print('  %s:' % name)
    print('    raw_model_prob = %.4f' % raw)
    print('    final_risk     = %.4f' % final)
    print('    risk_score     = %d/100' % score)
    print('    risk_level     = %s' % level)
    print('    is_fraud       = %s' % is_fraud)
    print('    amount_ratio   = %.2fx' % ratio)
    print()

print('=== SHAP DEEP VERIFICATION ===')
explainer = shap.TreeExplainer(model)

X_fraud = make_X(cases['C_fraud'])
sv = explainer.shap_values(X_fraud)

print('shap_values type:', type(sv).__name__)
if isinstance(sv, list):
    print('List length:', len(sv))
    for i, arr in enumerate(sv):
        print('  sv[%d] shape: %s' % (i, np.array(arr).shape))
    vals = sv[fraud_idx][0]
    print('Using sv[%d][0] for fraud class' % fraud_idx)
else:
    print('ndarray shape:', np.array(sv).shape)
    vals = sv[0]
    print('Using sv[0] for XGBoost (fraud class, verified)')

print()
print('Expected value (SHAP base):', explainer.expected_value)
print('Sum of SHAP values:', round(float(sum(vals)), 4))
print()
print('Top 10 SHAP feature contributions for C_fraud:')
paired = sorted(zip(feat, vals), key=lambda x: abs(x[1]), reverse=True)[:10]
for fn, fv in paired:
    direction = 'increases fraud risk' if fv > 0 else 'reduces fraud risk'
    print('  %-30s %+.4f  %s' % (fn, fv, direction))

print()
print('All positive SHAP (increasing fraud):')
pos = [(fn, fv) for fn, fv in zip(feat, vals) if fv > 0]
pos_sorted = sorted(pos, key=lambda x: x[1], reverse=True)
for fn, fv in pos_sorted:
    print('  %-30s %+.4f' % (fn, fv))

print()
print('All negative SHAP (reducing fraud):')
neg = [(fn, fv) for fn, fv in zip(feat, vals) if fv < 0]
neg_sorted = sorted(neg, key=lambda x: x[1])
for fn, fv in neg_sorted:
    print('  %-30s %+.4f' % (fn, fv))

print()
print('=== FEATURE ENGINEERING DETERMINISM CHECK ===')
X1 = make_X(cases['C_fraud'])
X2 = make_X(cases['C_fraud'])
print('Same input called twice, identical result:', X1.equals(X2))

print()
print('=== AMOUNT RATIO CHECK (training formula match) ===')
for name, data in cases.items():
    avg = float(data['AvgTransactionAmount'])
    expected = data['Amount'] / avg
    X = make_X(data)
    actual = float(X['AmountRatio'].iloc[0])
    match = abs(expected - actual) < 0.0001
    print('  %s: Amount=%s AvgAmt=%s  expected=%.4f actual=%.4f match=%s' % (
        name, data['Amount'], avg, expected, actual, match))

print()
print('=== RISKFLAGSUM CHECK ===')
for name, data in cases.items():
    X = make_X(data)
    high_failed = int(data.get('FailedAttempts',0) > 2)
    expected_rfs = int(data.get('UnusualLocation',0)) + int(data.get('UnusualAmount',0)) + int(data.get('NewDevice',0)) + high_failed
    actual_rfs = int(X['RiskFlagSum'].iloc[0])
    print('  %s: expected=%d actual=%d match=%s' % (name, expected_rfs, actual_rfs, expected_rfs==actual_rfs))

print()
print('=== METRICS VERIFICATION ===')
xm = metrics.get('xgboost', {})
print('XGBoost:')
print('  accuracy  :', xm.get('accuracy'))
print('  precision :', xm.get('precision'))
print('  recall    :', xm.get('recall'))
print('  f1        :', xm.get('f1'))
print('  roc_auc   :', xm.get('roc_auc'))
print('  pr_auc    :', xm.get('pr_auc'))
cm = xm.get('confusion_matrix', [])
print('  confusion_matrix:', cm)
if cm:
    tn, fp, fn_val, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    calc_acc = (tn + tp) / (tn + fp + fn_val + tp)
    print('  calc_accuracy_from_cm: %.4f (stored: %.4f, match: %s)' % (calc_acc, xm.get('accuracy',0), abs(calc_acc - xm.get('accuracy',0)) < 0.001))

print()
print('=== DONE ===')
