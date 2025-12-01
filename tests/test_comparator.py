import os
from rag_engine import comparator

def test_heuristic_pricing_missing_apr():
    clause = "Interest will be charged as per bank policy."
    category = "pricing"
    regs = [{'title':'RBI Fair Practices','text':'Banks must disclose APR and interest method.'}]
    res = comparator.compare_clause_with_regs(clause, category, regs, mode='heuristic')
    assert res['status'] == 'Reviewed'
    assert isinstance(res['issues'], list)
    assert len(res['issues']) >= 1
    assert any('APR' in (i.get('reason','') or '').upper() or 'APR' in (i.get('reg_text','') or '').upper() for i in res['issues'])

def test_heuristic_grievance_present():
    clause = "For grievances, contact us at support@example.com"
    category = "grievance"
    regs = [{'title':'RBI','text':'Grievance mechanism must include contact details.'}]
    res = comparator.compare_clause_with_regs(clause, category, regs, mode='heuristic')
    assert res['status'] == 'Reviewed'
    sev = [i.get('severity') for i in res.get('issues',[])]
    assert 'High' not in sev or len(res.get('issues',[])) == 0
