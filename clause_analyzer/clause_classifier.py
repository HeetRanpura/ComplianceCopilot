def classify_clause(text):
    t = text.lower()
    mapping = {
        'pricing': ['interest','apr','annual percentage','rate of interest','emi','equated monthly installment'],
        'fees': ['fee','charges','processing fee','late payment','late fee','prepayment fee','foreclosure'],
        'grievance': ['grievance','complaint','ombudsman','contact','redressal'],
        'kyc': ['kyc','know your customer','documents','identity','address proof'],
        'security': ['security interest','collateral','pledge','hypothecation','mortgage'],
        'default': ['default','recovery','demand notice','repossession','non-payment']
    }
    for cat, keys in mapping.items():
        for k in keys:
            if k in t:
                return cat
    return 'other'
