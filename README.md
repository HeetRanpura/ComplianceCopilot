
# ComplianceCopilot  
### AI-Driven Compliance Analyzer for Financial Documents  

Regulation-Grounded • LLM-Powered • Audit-Ready  

---

![CI](https://img.shields.io/github/actions/workflow/status/HeetRanpura/ComplianceCopilot/ci.yml?label=CI&logo=github)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/HeetRanpura/ComplianceCopilot?style=social)
![Forks](https://img.shields.io/github/forks/HeetRanpura/ComplianceCopilot?style=social)

---

## Overview  

**ComplianceCopilot** is an AI-powered system that analyzes financial documents such as:  
- Loan Agreements  
- Terms & Conditions  
- Key Fact Statements (KFS)  

It compares them against RBI/SEBI guidelines using:  
- Document parsing  
- Clause classification  
- Regulation retrieval  
- LLM-based reasoning  
- Risk scoring  
- Explainable reporting  

The goal is to automate and streamline compliance reviews with accuracy, speed, and clarity.

---

## Features  

### 📝 Document Ingestion  
- Upload PDF, DOCX, or TXT files  
- Extracts clean text  
- Splits into meaningful clauses  

### 🔎 Clause Classification  
Automatically identifies:  
- Pricing / APR  
- Fees & Charges  
- Foreclosure  
- Grievance Redressal  
- KYC / Documentation  
- Security  
- Default & Recovery  

### 📚 Regulation Retrieval  
- Fetches the most relevant regulatory snippets  
- Uses TF-IDF or embeddings (expandable)  

### 🤖 LLM-Powered Gap Detection  
- Compares document clauses with regulatory expectations  
- Generates:  
  - Compliance status  
  - Issues  
  - Severity  
  - Suggested remediation  
  - Citations  
- Automatic fallback to heuristic comparator if needed  

### 🔥 Risk Scoring  
- Severity-based risk calculation  
- Document-level overall score  

### 🖥 Streamlit UI  
- File upload  
- “Process File” button  
- Progress display  
- Comparator mode switch  
- Detailed issue breakdown  
- Report export  

### 🧪 Test Suite  
- Comparator tests using pytest  

---

## Architecture  

```
📂 ComplianceCopilot
│
├── ingestion/            # PDF/DOCX loaders, text cleaning
├── clause_analyzer/      # Clause splitting & classification
├── rag_engine/
│     ├── retriever.py    # Retrieval logic
│     ├── comparator.py   # LLM + heuristic comparator
│     └── prompts.py      # Prompt templates
├── risk_engine/          # Risk scoring
├── report/               # Report generator
├── tests/                # Pytest suite
└── app.py                # Streamlit frontend
```

---

## Installation  

### 1️⃣ Create virtual environment  
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Optional: enable LLM mode  
```bash
export OPENAI_API_KEY="your-key-here"
```

### 4️⃣ Run the application  
```bash
streamlit run app.py
```

---

## Running Tests  
```bash
pytest -q
```

---

## Example Issue Output  

```
[HIGH] APR not clearly disclosed.
Regulation: RBI Fair Practices Code
Reason: Customers must be informed of total cost of credit.
Fix: Include APR + example calculation.
```

---

## Roadmap  

- [ ] OCR for scanned PDFs  
- [ ] SEBI + IRDAI regulation expansion  
- [ ] Embedding-based retriever  
- [ ] Version comparison for documents  
- [ ] Export full PDF report  
- [ ] Deploy to cloud (Render/Railway)  

---

## Contributing  
Pull requests are welcome. Open an issue before major changes.

---

## License  
Distributed under the **MIT License**.

---

Built with passion to simplify compliance workflows.
