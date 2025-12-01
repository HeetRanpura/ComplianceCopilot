import re

def split_into_clauses(text):
    text = text.replace('\r','')
    parts = re.split(r'\n\s*\n+', text)
    clauses = []
    for p in parts:
        p = p.strip()
        if len(p) < 30:
            continue
        if len(p) > 1200:
            sub = re.split(r'(?<=\.|\?|\!)\s+', p)
            for s in sub:
                s = s.strip()
                if len(s) > 30:
                    clauses.append(s)
        else:
            clauses.append(p)
    return clauses
