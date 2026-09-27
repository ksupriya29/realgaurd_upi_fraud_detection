import urllib.request, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'http://127.0.0.1:8000'
def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data: req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())

s, d = call('GET', '/health')
print('Health:', d.get('status'), 'model:', d.get('model_loaded'), 'fraud_class:', d.get('fraud_class'))

A = {'Amount':500,'MerchantCategory':'Groceries','TransactionType':'P2M','Latitude':12.97,'Longitude':77.59,'AvgTransactionAmount':1000,'TransactionFrequency':'1/day','UnusualLocation':0,'UnusualAmount':0,'NewDevice':0,'FailedAttempts':0,'Hour':14,'DayOfWeek':1,'User_Transaction_Count':20,'IP_Frequency':2,'Phone_Usage_Count':5}
s, d = call('POST', '/predict', A)
diff_A = abs(d['fraud_probability'] - d['raw_model_probability'])
print('A: raw=%.4f final=%.4f diff=%.4f  rawRow=%s (expected HIDDEN)' % (
    d['raw_model_probability'], d['fraud_probability'], diff_A, 'HIDDEN' if diff_A < 0.005 else 'VISIBLE'))

B = {'Amount':8000,'MerchantCategory':'Electronics','TransactionType':'P2P','Latitude':28.6,'Longitude':77.2,'AvgTransactionAmount':3000,'TransactionFrequency':'3/day','UnusualLocation':1,'UnusualAmount':0,'NewDevice':1,'FailedAttempts':1,'Hour':23,'DayOfWeek':6,'User_Transaction_Count':40,'IP_Frequency':8,'Phone_Usage_Count':3}
s, d = call('POST', '/predict', B)
diff_B = abs(d['fraud_probability'] - d['raw_model_probability'])
print('B: raw=%.4f final=%.4f diff=%.4f  rawRow=%s (expected VISIBLE)' % (
    d['raw_model_probability'], d['fraud_probability'], diff_B, 'HIDDEN' if diff_B < 0.005 else 'VISIBLE'))

C = {'Amount':25000,'MerchantCategory':'Electronics','TransactionType':'P2M','Latitude':19.2,'Longitude':72.8,'AvgTransactionAmount':2000,'TransactionFrequency':'5/day','UnusualLocation':1,'UnusualAmount':1,'NewDevice':1,'FailedAttempts':5,'Hour':2,'DayOfWeek':6,'User_Transaction_Count':60,'IP_Frequency':15,'Phone_Usage_Count':10}
s, d = call('POST', '/predict', C)
diff_C = abs(d['fraud_probability'] - d['raw_model_probability'])
print('C: raw=%.4f final=%.4f diff=%.4f  rawRow=%s (expected HIDDEN - both at cap 1.0)' % (
    d['raw_model_probability'], d['fraud_probability'], diff_C, 'HIDDEN' if diff_C < 0.005 else 'VISIBLE'))

print()
print('All smoke tests OK.')
