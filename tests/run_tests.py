import sys, io, urllib.request, urllib.error, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'http://127.0.0.1:8000'
results = []
P, F = 0, 0

def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as ex:
        return 0, {'error': str(ex)}

def ok(label, condition, detail=''):
    global P, F
    if condition:
        P += 1
    else:
        F += 1
    sym = 'PASS' if condition else 'FAIL'
    line = '  [%s] %s' % (sym, label)
    if detail:
        line += ' -- ' + str(detail)
    results.append(line)

def section(title):
    results.append('')
    results.append('=' * 60)
    results.append('  ' + title)
    results.append('=' * 60)

legit = {
    'Amount': 500, 'MerchantCategory': 'Groceries', 'TransactionType': 'P2M',
    'Latitude': 12.97, 'Longitude': 77.59, 'AvgTransactionAmount': 1000,
    'TransactionFrequency': '1/day', 'UnusualLocation': 0, 'UnusualAmount': 0,
    'NewDevice': 0, 'FailedAttempts': 0, 'Hour': 14, 'DayOfWeek': 1,
    'User_Transaction_Count': 20, 'IP_Frequency': 2, 'Phone_Usage_Count': 5,
}
fraud_case = {
    'Amount': 25000, 'MerchantCategory': 'Electronics', 'TransactionType': 'P2M',
    'Latitude': 19.2, 'Longitude': 72.8, 'AvgTransactionAmount': 2000,
    'TransactionFrequency': '5/day', 'UnusualLocation': 1, 'UnusualAmount': 1,
    'NewDevice': 1, 'FailedAttempts': 5, 'Hour': 2, 'DayOfWeek': 6,
    'User_Transaction_Count': 60, 'IP_Frequency': 15, 'Phone_Usage_Count': 10,
}

# Test 1 - Health
section('TEST 1: API Health Check')
status, d = call('GET', '/health')
ok('Status 200', status == 200, status)
ok('status = healthy', d.get('status') == 'healthy', d.get('status'))
ok('model_loaded = True', d.get('model_loaded') == True)
ok('db_connected = True', d.get('db_connected') == True)
ok('fraud_class = 1', d.get('fraud_class') == 1, d.get('fraud_class'))
ok('features = 23', d.get('features') == 23, d.get('features'))

# Test 2 - Legit
section('TEST 2: Legitimate Transaction')
status, d = call('POST', '/predict', legit)
ok('Status 200', status == 200, status)
ok('risk_level = LOW', d.get('risk_level') == 'LOW', d.get('risk_level'))
ok('summary.merchant = Groceries', d.get('transaction_summary', {}).get('merchant') == 'Groceries')
ok('summary.txn_type = P2M', d.get('transaction_summary', {}).get('txn_type') == 'P2M')
ok('raw_model_prob in 0-1', 0 <= d.get('raw_model_probability', 99) <= 1)
results.append('  INFO: raw=%s, risk=%s, level=%s' % (d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_level')))

# Test 3 - Suspicious
section('TEST 3: Suspicious Transaction')
susp = {
    'Amount': 8000, 'MerchantCategory': 'Electronics', 'TransactionType': 'P2P',
    'Latitude': 28.6, 'Longitude': 77.2, 'AvgTransactionAmount': 3000,
    'TransactionFrequency': '3/day', 'UnusualLocation': 1, 'UnusualAmount': 0,
    'NewDevice': 1, 'FailedAttempts': 1, 'Hour': 23, 'DayOfWeek': 6,
    'User_Transaction_Count': 40, 'IP_Frequency': 8, 'Phone_Usage_Count': 3,
}
status, d = call('POST', '/predict', susp)
ok('Status 200', status == 200, status)
ok('fraud_prob > 0.10', d.get('fraud_probability', 0) > 0.10, d.get('fraud_probability'))
ok('has reasons', len(d.get('reasons', [])) > 0, len(d.get('reasons', [])))
results.append('  INFO: raw=%s, risk=%s, level=%s' % (d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_level')))

# Test 4 - High fraud
section('TEST 4: Highly Suspicious Transaction')
status, d = call('POST', '/predict', fraud_case)
ok('Status 200', status == 200, status)
ok('risk_level = HIGH', d.get('risk_level') == 'HIGH', d.get('risk_level'))
ok('is_fraud = True', d.get('is_fraud') == True)
ok('fraud_prob >= 0.65', d.get('fraud_probability', 0) >= 0.65, d.get('fraud_probability'))
ok('shap_increasing present', 'shap_increasing' in d)
ok('shap_decreasing present', 'shap_decreasing' in d)
si = d.get('shap_increasing', [])
sd = d.get('shap_decreasing', [])
ok('increasing values positive', all(f['shap_value'] > 0 for f in si) if si else True, len(si))
ok('decreasing values negative', all(f['shap_value'] < 0 for f in sd) if sd else True, len(sd))
results.append('  INFO: raw=%s, risk=%s, level=%s' % (d.get('raw_model_probability'), d.get('fraud_probability'), d.get('risk_level')))

# Test 5 - Validation
section('TEST 5: Input Validation')
for label, patch, expected in [
    ('Negative Amount', {'Amount': -100}, 422),
    ('Healthcare merchant', {'MerchantCategory': 'Healthcare'}, 422),
    ('TransactionType=Online', {'TransactionType': 'Online'}, 422),
    ('TransactionType=ATM', {'TransactionType': 'ATM'}, 422),
    ('Hour=25', {'Hour': 25}, 422),
    ('Freq=10/day', {'TransactionFrequency': '10/day'}, 422),
    ('AvgAmt=0', {'AvgTransactionAmount': 0}, 422),
]:
    s, _ = call('POST', '/predict', {**fraud_case, **patch})
    ok(label + ' rejected %d' % expected, s == expected, 'got %d' % s)

# Test 6 - SHAP
section('TEST 6: SHAP Verification')
status, d = call('POST', '/predict', fraud_case)
shap_all = d.get('shap_features', [])
ok('shap_features has 10 items', len(shap_all) == 10, len(shap_all))
ok('each has feature', all('feature' in s for s in shap_all))
ok('each has shap_value', all('shap_value' in s for s in shap_all))
ok('each has positive bool', all('positive' in s for s in shap_all))
inc = d.get('shap_increasing', [])
dec = d.get('shap_decreasing', [])
ok('shap_increasing not empty', len(inc) > 0, len(inc))
ok('shap_decreasing not empty', len(dec) > 0, len(dec))
ok('increasing all > 0', all(f['shap_value'] > 0 for f in inc))
ok('decreasing all < 0', all(f['shap_value'] < 0 for f in dec))
top_feat = shap_all[0]['feature'] if shap_all else 'N/A'
results.append('  INFO: top SHAP feature = %s (%s)' % (top_feat, shap_all[0].get('shap_value') if shap_all else ''))

# Test 7 - History
section('TEST 7: Transaction History')
status, hist = call('GET', '/history?limit=10')
ok('Status 200', status == 200, status)
ok('Returns list', isinstance(hist, list))
if hist:
    r0 = hist[0]
    ok('has txn_id', 'txn_id' in r0)
    ok('has fraud_prob', 'fraud_prob' in r0)
    ok('has merchant', 'merchant' in r0)
    ok('has txn_type', 'txn_type' in r0)
    ok('has risk_level', 'risk_level' in r0)
results.append('  INFO: %d history records' % len(hist))

status, fraud_hist = call('GET', '/history?fraud_only=true')
if fraud_hist:
    ok('fraud_only filter: all is_fraud=1', all(r['is_fraud'] == 1 for r in fraud_hist))

# Test 8 - Analytics
section('TEST 8: Analytics Dashboard')
status, a = call('GET', '/analytics')
ok('Status 200', status == 200, status)
ok('has total', 'total' in a)
ok('has fraud', 'fraud' in a)
ok('has legit', 'legit' in a)
ok('has by_merchant', 'by_merchant' in a)
ok('has by_type', 'by_type' in a)
ok('fraud + legit = total', a.get('fraud', 0) + a.get('legit', 0) == a.get('total', 0))
results.append('  INFO: total=%s, fraud=%s, legit=%s, rate=%s%%' % (
    a.get('total'), a.get('fraud'), a.get('legit'), a.get('fraud_rate')))

# Test 9 - Model Metrics
section('TEST 9: Model Performance Metrics')
status, m = call('GET', '/model-metrics')
ok('Status 200', status == 200, status)
ok('has xgboost', 'xgboost' in m)
ok('has roc_curve', 'roc_curve' in m)
ok('has pr_curve', 'pr_curve' in m)
xm = m.get('xgboost', {})
ok('accuracy in (0,1)', 0 < xm.get('accuracy', 0) < 1, xm.get('accuracy'))
ok('f1 in (0,1)', 0 < xm.get('f1', 0) < 1, xm.get('f1'))
ok('roc_auc in (0,1)', 0 < xm.get('roc_auc', 0) < 1, xm.get('roc_auc'))
ok('confusion_matrix present', 'confusion_matrix' in xm)
results.append('  INFO: acc=%s, f1=%s, roc_auc=%s, pr_auc=%s' % (
    xm.get('accuracy'), xm.get('f1'), xm.get('roc_auc'), xm.get('pr_auc')))

# Test 10 - Categories
section('TEST 10: Category Consistency')
status, cats = call('GET', '/categories')
ok('Status 200', status == 200, status)
mc = cats.get('merchant_categories', [])
tt = cats.get('transaction_types', [])
ok('7 merchant categories', len(mc) == 7, mc)
ok('Healthcare absent', 'Healthcare' not in mc)
ok('Fashion absent', 'Fashion' not in mc)
ok('Jewelry absent', 'Jewelry' not in mc)
ok('2 transaction types', len(tt) == 2, tt)
ok('P2M present', 'P2M' in tt)
ok('P2P present', 'P2P' in tt)
ok('Online absent', 'Online' not in tt)
ok('ATM absent', 'ATM' not in tt)
ok('POS absent', 'POS' not in tt)

# Risk level consistency check
section('BONUS: Risk Level / Score Consistency')
for case_name, case in [('legit', legit), ('fraud', fraud_case)]:
    _, d = call('POST', '/predict', case)
    score = d.get('risk_score', -1)
    level = d.get('risk_level', '')
    if level == 'HIGH':
        ok('%s: score %d >= 65 (HIGH)' % (case_name, score), score >= 65, score)
    elif level == 'MEDIUM':
        ok('%s: score %d in 30-64 (MEDIUM)' % (case_name, score), 30 <= score <= 64, score)
    elif level == 'LOW':
        ok('%s: score %d < 30 (LOW)' % (case_name, score), score < 30, score)

# Print all
for line in results:
    print(line)

print('')
print('=' * 60)
print('  RESULTS: %d PASSED  |  %d FAILED  |  %d TOTAL' % (P, F, P + F))
print('=' * 60)
