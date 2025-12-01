import streamlit as st
from ingestion.pdf_loader import extract_text_from_uploaded
from clause_analyzer.clause_splitter import split_into_clauses
from clause_analyzer.clause_classifier import classify_clause
from regulation_index.build_reg_index import load_regulatory_corpus, build_tfidf_index, retrieve_top_k
from rag_engine.comparator import compare_clause_with_regs
from risk_engine.risk_scorer import score_issues, normalize_score
from report.report_generator import generate_html_report
import os, time

# Application UI (title and caption avoid forbidden headings)
st.set_page_config(page_title='ComplianceCopilot', layout='wide')
st.title("ComplianceCopilot")
st.caption("Upload a document (PDF / DOCX / TXT). Uses TF-IDF retrieval + comparator (LLM or heuristic).")

# Sidebar inputs
st.sidebar.header("Upload / Settings")
uploaded = st.sidebar.file_uploader("Upload a loan agreement or T&C (PDF, DOCX, TXT)", type=['pdf','txt','docx'], accept_multiple_files=False)
k = st.sidebar.slider("Regulatory retrieval top-K", min_value=1, max_value=5, value=3)
mode = st.sidebar.radio("Comparator mode", options=["Auto (LLM if API key)", "Heuristic only", "LLM only"])
st.sidebar.markdown("**Note:** LLM mode requires `OPENAI_API_KEY` environment variable to be set.")

# Load regulatory corpus
reg_docs = load_regulatory_corpus()
vectorizer, tfidf_matrix = build_tfidf_index(reg_docs)

if uploaded is None:
    st.info("Please upload a document (PDF, DOCX, or TXT) to run the compliance check.")
    st.stop()

# Process button
if 'process_clicked' not in st.session_state:
    st.session_state['process_clicked'] = False

if st.button("Process uploaded file"):
    st.session_state['process_clicked'] = True

if not st.session_state['process_clicked']:
    st.info("Click 'Process uploaded file' to start analysis.")
    st.stop()

# Extract text with feedback
with st.spinner("Extracting text from uploaded file..."):
    try:
        input_text = extract_text_from_uploaded(uploaded)
    except Exception as e:
        st.error(f"Failed to extract text from the uploaded file: {e}")
        st.stop()

if not input_text or len(input_text.strip()) < 50:
    st.error("No usable text could be extracted from the uploaded file. Try a different file or ensure the PDF is not a scanned image (OCR required).")
    st.stop()

st.header("Document preview")
with st.expander("Show extracted text", expanded=False):
    st.text_area("Extracted document text", input_text[:10000], height=300)

# Clause splitting
clauses = split_into_clauses(input_text)
st.write(f"Detected **{len(clauses)}** clauses/sections (heuristic split).")

# Allow processing limit
max_clauses = st.sidebar.number_input("Max clauses to process (0 = all)", min_value=0, value=0)
if max_clauses > 0:
    clauses = clauses[:max_clauses]

# Process clauses with progress and retry info
results = []
issues = []
progress_bar = st.progress(0)
total = max(1, len(clauses))
completed = 0

if mode == "Auto (LLM if API key)":
    chosen_mode = "auto"
elif mode == "Heuristic only":
    chosen_mode = "heuristic"
else:
    chosen_mode = "llm"

st.info(f"Comparator mode: **{mode}**. Processing {total} clauses...")

for i, clause in enumerate(clauses):
    with st.spinner(f"Processing clause {i+1}/{total}..."):
        category = classify_clause(clause)
        top_regs = retrieve_top_k(clause, reg_docs, vectorizer, tfidf_matrix, top_k=k)
        comp = compare_clause_with_regs(clause, category, top_regs, mode=chosen_mode)
        results.append({
            "clause_id": i+1,
            "clause_text": clause,
            "category": category,
            "retrieved_regs": top_regs,
            "analysis": comp
        })
        for issue in comp.get('issues', []):
            issue_record = {
                "clause_id": i+1,
                "category": category,
                "clause_text": clause,
                "reg_snippet": issue.get('reg_text',''),
                "reason": issue.get('reason',''),
                "severity": issue.get('severity','Medium'),
                "status": issue.get('status','Partial'),
                "suggested_fix": issue.get('suggested_fix','')
            }
            issues.append(issue_record)
    completed += 1
    progress_bar.progress(int(completed/total*100))
    time.sleep(0.2)

doc_score = normalize_score(score_issues(issues))

# Executive summary and details
st.header("Executive summary")
col1, col2 = st.columns([1,2])
with col1:
    st.metric("Document Risk Score", f"{doc_score}/100")
    if doc_score >= 70:
        st.markdown("<span style='color:red;font-weight:bold'>HIGH RISK</span>", unsafe_allow_html=True)
    elif doc_score >= 30:
        st.markdown("<span style='color:orange;font-weight:bold'>MEDIUM RISK</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green;font-weight:bold'>LOW RISK</span>", unsafe_allow_html=True)
with col2:
    st.write("Top issues (by severity)")
    sorted_issues = sorted(issues, key=lambda x: {'High':3,'Medium':2,'Low':1}[x['severity']], reverse=True)
    for issue in sorted_issues[:5]:
        st.write(f"- Clause {issue['clause_id']} ({issue['category']}) — **{issue['reason']}** — {issue['severity']}")

st.header("All Issues")
if not issues:
    st.success("No issues detected by the current comparator.")
else:
    for issue in issues:
        st.markdown(f"**Clause {issue['clause_id']} — {issue['category']} — {issue['severity']}**")
        st.write(issue['clause_text'][:400])
        st.write("Regulatory snippet:")
        st.write(issue['reg_snippet'][:400])
        if issue.get('suggested_fix'):
            st.write("Suggested fix:", issue['suggested_fix'])
        st.write("Reason:", issue['reason'])
        st.write("---")

# Generate report
if st.button("Generate HTML Report"):
    html = generate_html_report(input_text, issues, doc_score, results)
    b = html.encode('utf-8')
    st.download_button("Download HTML report", b, file_name="compliance_report.html", mime="text/html")
    st.success("Report generated. Download from the button above.")
