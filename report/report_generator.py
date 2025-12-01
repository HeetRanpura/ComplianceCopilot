import html
def generate_html_report(doc_text, issues, score, detailed_results):
    title = "ComplianceCopilot — Report"
    html_parts = []
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append(f"<h3>Overall Risk Score: {score}/100</h3>")
    html_parts.append("<h2>Key Issues</h2>")
    if not issues:
        html_parts.append("<p>No issues detected by heuristic/LLM checks.</p>")
    else:
        html_parts.append("<ol>")
        for it in issues:
            html_parts.append(f"<li><strong>Clause {it['clause_id']} ({it['category']})</strong>: {html.escape(it['reason'])} — <em>{it['severity']}</em><br/>")
            html_parts.append(f"<details><summary>Clause text (click)</summary><pre>{html.escape(it['clause_text'][:1000])}</pre></details>")
            html_parts.append(f"<details><summary>Regulatory snippet (click)</summary><pre>{html.escape(it['reg_snippet'][:1000])}</pre></details>")
            html_parts.append("</li>")
        html_parts.append("</ol>")
    html_parts.append("<h2>Detailed Clause Analysis</h2>")
    for r in detailed_results:
        html_parts.append(f"<h4>Clause {r['clause_id']} — {r['category']}</h4>")
        html_parts.append(f"<pre>{html.escape(r['clause_text'][:2000])}</pre>")
        html_parts.append("<p>Retrieved regulatory excerpts:</p>")
        for reg in r['retrieved_regs']:
            html_parts.append(f"<h5>{reg['title']}</h5><pre>{html.escape(reg['text'][:2000])}</pre>")
    return '\n'.join(html_parts)
