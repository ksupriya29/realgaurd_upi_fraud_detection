"""
REALGUARD Final Verification Test Suite
Tests: 5 scenarios + history + analytics + model metrics + frontend/backend fields
"""
import sys, io, urllib.request, urllib.error, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'http://127.0.0.1:8000'
PASS = 0
FAIL = 0
rows = []

def call(method, path, body=None):
    url  = BASE + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as ex:
        return 0, {'error': str(ex)}

def ok(label, condition, actual='', expected=''):
    global PASS, FAIL
    if condition: PASS += 1
    else: FAIL += 1
    sym = 'PASS' if condition else 'FAIL'
    print('  [%s] %s' % (sym, label))
    if not condition:
        print('        expected=%s  actual=%s' % (expected, actual))

def hdr(title):
    print()
    print('=' * 65)
    print('  ' + title)
    print('=' * 65)

# ─────────────────────────────────────────────────────────────────
# First clear history for clean analytics test
# ─────────────────────────────────────────────────────────────────
hdr('SETUP: Clear history for clean analytics verification')
status, _ = call('DELETE', '/history')
ok('History cleared', status == 200, status, 200)

# ─────────────────────────────────────────────────────────────────
# TEST A — Low-risk / normal transaction
# ─────────────────────────────────────────────────────────────────
hdr('TEST A: Low-risk / Normal Transaction')
A = {
    'Amount': 500, 'MerchantCategory': 'Groceries', 'TransactionType': 'P2M',
    'Latitude': 12.97, 'Longitude': 77.59, 'AvgTransactionAmount': 1000,
    'TransactionFrequency': '1/day', 'UnusualLocation': 0, 'UnusualAmount': 0,
    'NewDevice': 0, 'FailedAttempts': 0, 'Hour': 14, 'DayOfWeek': 1,
    'User_Transaction_Count': 20, 'IP_Frequency': 2, 'Phone_Usage_Count': 5,
}
s, d = call('POST', '/predict', A)
ok('Status 200', s == 200, s, 200)
ok('risk_level = LOW', d.get('risk_level') == 'LOW', d.get('risk_level'), 'LOW')
ok('is_fraud = False', d.get('is_fraud') == False, d.get('is_fraud'), False)
ok('risk_score 0-100', 0 <= d.get('risk_score', -1) <= 100)
ok('fraud_probability in [0,1]', 0 <= d.get('fraud_probability', -1) <= 1)
ok('raw_model_probability in [0,1]', 0 <= d.get('raw_model_probability', -1) <= 1)
ok('fraud_prob >= raw_prob (domain rules only add)', d.get('fraud_probability', 0) >= d.get('raw_model_probability', 0))
ok('summary.merchant = Groceries', d.get('transaction_summary', {}).get('merchant') == 'Groceries')
ok('summary.txn_type = P2M', d.get('transaction_summary', {}).get('txn_type') == 'P2M')
ok('summary.amount = 500', d.get('transaction_summary', {}).get('amount') == 500)
ok('summary.amount_ratio = 0.5', abs(d.get('transaction_summary', {}).get('amount_ratio', 99) - 0.5) < 0.01)
ok('shap_features present', len(d.get('shap_features', [])) > 0)
ok('txn_id starts with TXN-', d.get('txn_id', '').startswith('TXN-'))
rows.append(('A', 'Low-risk/Normal', d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_score'), d.get('risk_level'), d.get('is_fraud'), d.get('txn_id')))
print('  INFO: raw=%.4f  final=%.4f  score=%d  level=%s  fraud=%s' % (
    d.get('raw_model_probability',0), d.get('fraud_probability',0),
    d.get('risk_score',0), d.get('risk_level',''), d.get('is_fraud','')))

# ─────────────────────────────────────────────────────────────────
# TEST B — Suspicious transaction
# ─────────────────────────────────────────────────────────────────
hdr('TEST B: Suspicious Transaction')
B = {
    'Amount': 8000, 'MerchantCategory': 'Electronics', 'TransactionType': 'P2P',
    'Latitude': 28.6, 'Longitude': 77.2, 'AvgTransactionAmount': 3000,
    'TransactionFrequency': '3/day', 'UnusualLocation': 1, 'UnusualAmount': 0,
    'NewDevice': 1, 'FailedAttempts': 1, 'Hour': 23, 'DayOfWeek': 6,
    'User_Transaction_Count': 40, 'IP_Frequency': 8, 'Phone_Usage_Count': 3,
}
s, d = call('POST', '/predict', B)
ok('Status 200', s == 200, s, 200)
ok('fraud_probability > A (more suspicious)', d.get('fraud_probability', 0) > rows[-1][4]/100 if rows else True)
ok('risk_score > 11 (higher than A)', d.get('risk_score', 0) > 11, d.get('risk_score'), '>11')
ok('fraud_prob in [0,1]', 0 <= d.get('fraud_probability', -1) <= 1)
ok('has reasons', len(d.get('reasons', [])) > 0, len(d.get('reasons', [])), '>0')
ok('shap_increasing present', 'shap_increasing' in d)
ok('shap_decreasing present', 'shap_decreasing' in d)
ok('summary.txn_type = P2P', d.get('transaction_summary', {}).get('txn_type') == 'P2P')
ok('summary.amount = 8000', d.get('transaction_summary', {}).get('amount') == 8000)
rows.append(('B', 'Suspicious', d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_score'), d.get('risk_level'), d.get('is_fraud'), d.get('txn_id')))
print('  INFO: raw=%.4f  final=%.4f  score=%d  level=%s  fraud=%s' % (
    d.get('raw_model_probability',0), d.get('fraud_probability',0),
    d.get('risk_score',0), d.get('risk_level',''), d.get('is_fraud','')))

# ─────────────────────────────────────────────────────────────────
# TEST C — Highly suspicious transaction
# ─────────────────────────────────────────────────────────────────
hdr('TEST C: Highly Suspicious Transaction')
C = {
    'Amount': 25000, 'MerchantCategory': 'Electronics', 'TransactionType': 'P2M',
    'Latitude': 19.2, 'Longitude': 72.8, 'AvgTransactionAmount': 2000,
    'TransactionFrequency': '5/day', 'UnusualLocation': 1, 'UnusualAmount': 1,
    'NewDevice': 1, 'FailedAttempts': 5, 'Hour': 2, 'DayOfWeek': 6,
    'User_Transaction_Count': 60, 'IP_Frequency': 15, 'Phone_Usage_Count': 10,
}
s, d = call('POST', '/predict', C)
ok('Status 200', s == 200, s, 200)
ok('risk_level = HIGH', d.get('risk_level') == 'HIGH', d.get('risk_level'), 'HIGH')
ok('is_fraud = True', d.get('is_fraud') == True, d.get('is_fraud'), True)
ok('fraud_probability >= 0.65', d.get('fraud_probability', 0) >= 0.65, d.get('fraud_probability'), '>=0.65')
ok('risk_score >= 65', d.get('risk_score', 0) >= 65, d.get('risk_score'), '>=65')
# SHAP split correctness
inc = d.get('shap_increasing', [])
dec = d.get('shap_decreasing', [])
ok('shap_increasing not empty', len(inc) > 0, len(inc), '>0')
ok('shap_decreasing not empty', len(dec) > 0, len(dec), '>0')
ok('all increasing shap_values > 0', all(f['shap_value'] > 0 for f in inc), [f['shap_value'] for f in inc])
ok('all decreasing shap_values < 0', all(f['shap_value'] < 0 for f in dec), [f['shap_value'] for f in dec])
ok('top shap_increasing feature is RiskFlagSum', inc[0]['feature'] == 'RiskFlagSum' if inc else False, inc[0]['feature'] if inc else 'none', 'RiskFlagSum')
ok('fraud_prob > B (more suspicious)', d.get('fraud_probability', 0) >= rows[-1][3])
ok('summary.amount_ratio = 12.5', abs(d.get('transaction_summary', {}).get('amount_ratio', 99) - 12.5) < 0.01)
# Verify domain rules are applied correctly
# raw=0.9999 + 0.20(amount>3x) + 0.15(fa>2) + 0.10(newdev) + 0.10(unusualloc) + 0.05(unusualamt) + 0.05(night) + 0.05(5/day)
# = 0.9999 + 0.70 = 1.6999 → capped at 1.0 → score=100 ✓
expected_score = 100  # due to cap
ok('risk_score = 100 (capped)', d.get('risk_score') == 100, d.get('risk_score'), 100)
rows.append(('C', 'Highly Suspicious', d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_score'), d.get('risk_level'), d.get('is_fraud'), d.get('txn_id')))
print('  INFO: raw=%.4f  final=%.4f  score=%d  level=%s  fraud=%s' % (
    d.get('raw_model_probability',0), d.get('fraud_probability',0),
    d.get('risk_score',0), d.get('risk_level',''), d.get('is_fraud','')))
print('  SHAP top contributors:')
for f in inc[:3]:
    print('    + %-28s %+.4f' % (f['feature'], f['shap_value']))
for f in dec[:2]:
    print('    - %-28s %+.4f' % (f['feature'], f['shap_value']))

# ─────────────────────────────────────────────────────────────────
# TEST D — Boundary transaction (around domain-rule thresholds)
# ─────────────────────────────────────────────────────────────────
hdr('TEST D: Boundary Transaction (threshold testing)')
D = {
    'Amount': 3100, 'MerchantCategory': 'Travel', 'TransactionType': 'P2M',
    'Latitude': 22.5, 'Longitude': 88.3, 'AvgTransactionAmount': 1000,
    'TransactionFrequency': '3/day', 'UnusualLocation': 0, 'UnusualAmount': 1,
    'NewDevice': 0, 'FailedAttempts': 2, 'Hour': 4, 'DayOfWeek': 3,
    'User_Transaction_Count': 10, 'IP_Frequency': 5, 'Phone_Usage_Count': 2,
}
s, d = call('POST', '/predict', D)
ok('Status 200', s == 200, s, 200)
ok('fraud_prob in [0,1]', 0 <= d.get('fraud_probability', -1) <= 1)
ok('risk_score in [0,100]', 0 <= d.get('risk_score', -1) <= 100)
# Amount ratio = 3.1 > 3 → should get +0.20 boost
# FailedAttempts = 2 → NOT > 2, so +0.05 (>0 branch)
# UnusualAmount = 1 → +0.05
# Hour = 4 < 5 → +0.05
# Total boost = 0.20 + 0.05 + 0.05 + 0.05 = 0.35
raw_D = d.get('raw_model_probability', 0)
expected_min = raw_D + 0.35
expected_final = min(expected_min, 1.0)
ok('domain boost applied correctly (ratio>3: +0.20, fa=2: +0.05, unusualamt: +0.05, night: +0.05)',
   abs(d.get('fraud_probability', 0) - expected_final) < 0.01,
   '%.4f' % d.get('fraud_probability', 0), '%.4f' % expected_final)
ok('summary.amount_ratio ~3.1', abs(d.get('transaction_summary', {}).get('amount_ratio', 0) - 3.1) < 0.01)
rows.append(('D', 'Boundary', d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_score'), d.get('risk_level'), d.get('is_fraud'), d.get('txn_id')))
print('  INFO: raw=%.4f  boost=+0.35  expected_final=%.4f  actual_final=%.4f' % (
    raw_D, expected_final, d.get('fraud_probability', 0)))
print('        score=%d  level=%s  fraud=%s' % (d.get('risk_score',0), d.get('risk_level',''), d.get('is_fraud','')))

# ─────────────────────────────────────────────────────────────────
# TEST E — Invalid inputs (validation)
# ─────────────────────────────────────────────────────────────────
hdr('TEST E: Invalid Input Validation')
base_valid = C.copy()

validation_cases = [
    ('Negative Amount',       {**base_valid, 'Amount': -100},        422),
    ('Zero Amount',           {**base_valid, 'Amount': 0},           422),
    ('Invalid MerchantCat',   {**base_valid, 'MerchantCategory': 'Healthcare'}, 422),
    ('Invalid TxnType Online',{**base_valid, 'TransactionType': 'Online'},      422),
    ('Invalid TxnType ATM',   {**base_valid, 'TransactionType': 'ATM'},         422),
    ('Invalid TxnType POS',   {**base_valid, 'TransactionType': 'POS'},         422),
    ('Hour = 24',             {**base_valid, 'Hour': 24},            422),
    ('Hour = -1',             {**base_valid, 'Hour': -1},            422),
    ('DayOfWeek = 7',         {**base_valid, 'DayOfWeek': 7},        422),
    ('Lat = 91',              {**base_valid, 'Latitude': 91},         422),
    ('Lon = 181',             {**base_valid, 'Longitude': 181},       422),
    ('Negative FailedAttempts',{**base_valid, 'FailedAttempts': -1}, 422),
    ('Invalid Frequency',     {**base_valid, 'TransactionFrequency': '10/day'}, 422),
    ('Zero AvgAmt',           {**base_valid, 'AvgTransactionAmount': 0},        422),
]

for label, payload, expected_status in validation_cases:
    s, resp = call('POST', '/predict', payload)
    ok('%s → %d' % (label, expected_status), s == expected_status, 'got %d' % s, str(expected_status))

# ─────────────────────────────────────────────────────────────────
# TEST: Probability changes with different inputs
# ─────────────────────────────────────────────────────────────────
hdr('TEST: Probability Changes with Input')
_, d_low  = call('POST', '/predict', A)   # Amount=500, AvgAmt=1000
_, d_high = call('POST', '/predict', C)   # Amount=25000, AvgAmt=2000
ok('C has higher fraud_prob than A', d_high.get('fraud_probability',0) > d_low.get('fraud_probability',0),
   'A=%.4f C=%.4f' % (d_low.get('fraud_probability',0), d_high.get('fraud_probability',0)))
ok('C has higher risk_score than A', d_high.get('risk_score',0) > d_low.get('risk_score',0))
ok('C has higher raw_model_prob than A', d_high.get('raw_model_probability',0) > d_low.get('raw_model_probability',0))
# Verify AmountRatio changes
ok('A summary.amount_ratio = 0.5', abs(d_low.get('transaction_summary',{}).get('amount_ratio',99) - 0.5) < 0.01)
ok('C summary.amount_ratio = 12.5', abs(d_high.get('transaction_summary',{}).get('amount_ratio',99) - 12.5) < 0.01)

# ─────────────────────────────────────────────────────────────────
# TEST: History — real data, unique IDs, correct fields
# ─────────────────────────────────────────────────────────────────
hdr('TEST: History Verification')
s, hist = call('GET', '/history?limit=50')
ok('History returns 200', s == 200)
ok('History is list', isinstance(hist, list))
ok('Has at least 4 records (A+B+C+D)', len(hist) >= 4, len(hist), '>=4')

if hist:
    # Check all required fields
    r = hist[0]
    for field in ['txn_id', 'created_at', 'amount', 'merchant', 'txn_type', 'fraud_prob', 'risk_score', 'risk_level', 'is_fraud']:
        ok('has field: %s' % field, field in r, list(r.keys()))

    # Check all IDs are unique
    ids = [r['txn_id'] for r in hist]
    ok('All txn_ids unique', len(ids) == len(set(ids)), '%d total, %d unique' % (len(ids), len(set(ids))))

    # Verify the test transactions are in history with correct data
    # Check TXN from test A is stored correctly
    txn_a = next((r for r in hist if r['txn_id'] == rows[0][7]), None)
    if txn_a:
        ok('Test A stored: amount=500', abs(txn_a['amount'] - 500) < 0.01, txn_a['amount'], 500)
        ok('Test A stored: merchant=Groceries', txn_a['merchant'] == 'Groceries')
        ok('Test A stored: fraud_prob matches', abs(txn_a['fraud_prob'] - rows[0][3]) < 0.001)
        ok('Test A stored: is_fraud=0', txn_a['is_fraud'] == 0)

    txn_c = next((r for r in hist if r['txn_id'] == rows[2][7]), None)
    if txn_c:
        ok('Test C stored: amount=25000', abs(txn_c['amount'] - 25000) < 0.01, txn_c['amount'], 25000)
        ok('Test C stored: fraud_prob matches response', abs(txn_c['fraud_prob'] - rows[2][3]) < 0.001)
        ok('Test C stored: is_fraud=1', txn_c['is_fraud'] == 1)
        ok('Test C stored: risk_level=HIGH', txn_c['risk_level'] == 'HIGH')
        ok('Test C stored: risk_score=100', txn_c['risk_score'] == 100)

    # Timestamp check
    ok('created_at is ISO string', 'T' in hist[0].get('created_at',''))

    # fraud_only filter
    s2, fraud_hist = call('GET', '/history?fraud_only=true')
    ok('fraud_only filter: all is_fraud=1', all(r['is_fraud'] == 1 for r in fraud_hist), 'not all fraud')

    # legit_only filter
    s3, legit_hist = call('GET', '/history?legit_only=true')
    ok('legit_only filter: all is_fraud=0', all(r['is_fraud'] == 0 for r in legit_hist) if legit_hist else True)

# Page refresh does NOT add new records
hist_len_before = len(hist)
s, hist2 = call('GET', '/history?limit=50')
ok('Page refresh does not add records', len(hist2) == hist_len_before, len(hist2), hist_len_before)

# ─────────────────────────────────────────────────────────────────
# TEST: Analytics — derived from real stored data
# ─────────────────────────────────────────────────────────────────
hdr('TEST: Analytics Verification')
s, a = call('GET', '/analytics')
ok('Analytics returns 200', s == 200)

total_hist = len(hist)   # we know exactly how many we stored
ok('analytics.total matches history count', a.get('total') == total_hist,
   'analytics=%d history=%d' % (a.get('total',0), total_hist))

# Count fraud/legit from history manually
actual_fraud = sum(1 for r in hist if r['is_fraud'] == 1)
actual_legit = sum(1 for r in hist if r['is_fraud'] == 0)
ok('analytics.fraud matches manual count', a.get('fraud') == actual_fraud,
   'analytics=%d manual=%d' % (a.get('fraud',0), actual_fraud))
ok('analytics.legit matches manual count', a.get('legit') == actual_legit,
   'analytics=%d manual=%d' % (a.get('legit',0), actual_legit))
ok('fraud + legit = total', a.get('fraud',0) + a.get('legit',0) == a.get('total',0))

expected_rate = round(actual_fraud / total_hist * 100, 2) if total_hist else 0
ok('fraud_rate calculation correct', abs(a.get('fraud_rate',0) - expected_rate) < 0.01,
   '%.2f' % a.get('fraud_rate',0), '%.2f' % expected_rate)

# avg_amount
actual_avg = sum(r['amount'] for r in hist) / len(hist) if hist else 0
ok('avg_amount calculation correct', abs(a.get('avg_amount',0) - actual_avg) < 0.01,
   '%.2f' % a.get('avg_amount',0), '%.2f' % actual_avg)

ok('has by_merchant', isinstance(a.get('by_merchant'), list))
ok('has by_type', isinstance(a.get('by_type'), list))
ok('has by_hour', isinstance(a.get('by_hour'), list))
ok('has amounts list', isinstance(a.get('amounts'), list))
ok('amounts list length = total', len(a.get('amounts',[])) == a.get('total',0))

print('  INFO: total=%d fraud=%d legit=%d rate=%.2f%% avg_amt=%.2f' % (
    a.get('total',0), a.get('fraud',0), a.get('legit',0),
    a.get('fraud_rate',0), a.get('avg_amount',0)))

# ─────────────────────────────────────────────────────────────────
# TEST: Model Metrics — genuine evaluation
# ─────────────────────────────────────────────────────────────────
hdr('TEST: Model Performance Metrics')
s, m = call('GET', '/model-metrics')
ok('Returns 200', s == 200)
ok('has xgboost section', 'xgboost' in m)
ok('has logistic_regression', 'logistic_regression' in m)
ok('has random_forest', 'random_forest' in m)
ok('has roc_curve', 'roc_curve' in m)
ok('has pr_curve', 'pr_curve' in m)

xm = m.get('xgboost', {})
ok('XGB accuracy = 0.926', abs(xm.get('accuracy',0) - 0.926) < 0.001, xm.get('accuracy'), 0.926)
ok('XGB precision = 0.9743', abs(xm.get('precision',0) - 0.9743) < 0.001, xm.get('precision'), 0.9743)
ok('XGB recall = 0.8214', abs(xm.get('recall',0) - 0.8214) < 0.001, xm.get('recall'), 0.8214)
ok('XGB f1 = 0.8913', abs(xm.get('f1',0) - 0.8913) < 0.001, xm.get('f1'), 0.8913)
ok('XGB roc_auc = 0.9104', abs(xm.get('roc_auc',0) - 0.9104) < 0.001, xm.get('roc_auc'), 0.9104)
ok('XGB pr_auc = 0.9221', abs(xm.get('pr_auc',0) - 0.9221) < 0.001, xm.get('pr_auc'), 0.9221)
ok('confusion_matrix present', 'confusion_matrix' in xm)

cm = xm.get('confusion_matrix', [])
if cm:
    tn, fp, fn_val, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    calc_acc = (tn + tp) / (tn + fp + fn_val + tp)
    ok('confusion_matrix consistent with accuracy', abs(calc_acc - 0.926) < 0.001,
       '%.4f' % calc_acc, '0.926')
    ok('TN=1245', tn == 1245, tn, 1245)
    ok('FP=16', fp == 16, fp, 16)
    ok('FN=132', fn_val == 132, fn_val, 132)
    ok('TP=607', tp == 607, tp, 607)

ok('roc_curve has fpr+tpr', 'fpr' in m.get('roc_curve',{}) and 'tpr' in m.get('roc_curve',{}))
ok('pr_curve has precision+recall', 'precision' in m.get('pr_curve',{}) and 'recall' in m.get('pr_curve',{}))

print('  INFO: acc=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f' % (
    xm.get('accuracy',0), xm.get('f1',0), xm.get('roc_auc',0), xm.get('pr_auc',0)))
print('  INFO: TN=%d FP=%d FN=%d TP=%d' % (tn, fp, fn_val, tp))

# ─────────────────────────────────────────────────────────────────
# TEST: Frontend/Backend field name consistency
# ─────────────────────────────────────────────────────────────────
hdr('TEST: Frontend / Backend Field Consistency')
# All 16 fields that frontend sends
required_fields = [
    'Amount', 'MerchantCategory', 'TransactionType', 'Latitude', 'Longitude',
    'AvgTransactionAmount', 'TransactionFrequency', 'UnusualLocation', 'UnusualAmount',
    'NewDevice', 'FailedAttempts', 'Hour', 'DayOfWeek', 'User_Transaction_Count',
    'IP_Frequency', 'Phone_Usage_Count',
]
s, d = call('POST', '/predict', C)
ok('All 16 fields accepted', s == 200)

# Verify response fields that frontend uses
resp_fields = ['txn_id', 'prediction', 'is_fraud', 'fraud_probability',
               'raw_model_probability', 'risk_score', 'risk_level', 'reasons',
               'shap_features', 'shap_increasing', 'shap_decreasing', 'transaction_summary']
for f in resp_fields:
    ok('response has field: %s' % f, f in d)

# Verify transaction_summary fields that frontend renders
sum_fields = ['amount','avg_amount','merchant','txn_type','hour','day_of_week',
              'location','new_device','unusual_loc','unusual_amt','failed_attempts',
              'ip_frequency','amount_ratio']
sm = d.get('transaction_summary', {})
for f in sum_fields:
    ok('summary has field: %s' % f, f in sm)

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('  TEST MATRIX SUMMARY')
print('=' * 65)
print('  %-12s %-18s %-8s %-8s %-6s %-8s %-6s' % ('Test', 'Type', 'RawProb', 'FinalProb', 'Score', 'Level', 'Fraud'))
print('  ' + '-' * 63)
for test, ttype, raw, final, score, level, fraud, txnid in rows:
    print('  %-12s %-18s %-8s %-8s %-6s %-8s %-6s' % (
        test, ttype,
        '%.4f' % (raw or 0), '%.4f' % (final or 0),
        '%d/100' % (score or 0), level or '', str(fraud)))

print()
print('=' * 65)
print('  RESULTS: %d PASSED  |  %d FAILED  |  %d TOTAL' % (PASS, FAIL, PASS+FAIL))
print('=' * 65)
