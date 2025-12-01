def score_issues(issues):
    w = {'High':3, 'Medium':2, 'Low':1}
    total = 0.0
    for i in issues:
        sev = i.get('severity','Medium')
        total += w.get(sev,2)
    return total

def normalize_score(raw_score):
    max_raw = 90.0
    val = min(raw_score, max_raw)/max_raw*100
    return int(val)
