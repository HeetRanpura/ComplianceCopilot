# rag_engine/comparator.py
import os
import json
import re
import time
from typing import List, Dict, Any

# -------------------------
# Heuristic comparator (fallback)
# -------------------------
def _heuristic_compare(clause_text: str, category: str, reg_snippets: List[Dict[str, str]]):
    issues = []
    t = (clause_text or "").lower()
    checks = {
        "pricing": ["apr", "annual percentage rate", "rate of interest", "interest", "%", "per cent", "annualized"],
        "fees": ["fee", "charges", "processing fee", "late fee", "foreclosure", "prepayment"],
        "grievance": ["grievance", "complaint", "ombudsman", "contact", "address", "email", "phone"],
        "kyc": ["identity", "address proof", "documents", "kyc"],
        "security": ["collateral", "security interest", "hypothecation", "mortgage", "charge"],
        "default": ["default", "recovery", "repossession", "demand notice"],
    }
    mandatory = checks.get(category, [])
    found = any(m in t for m in mandatory)

    if not found and len(mandatory) > 0:
        reg_text = reg_snippets[0]["text"] if reg_snippets else ""
        issues.append({
            "status": "Non-compliant",
            "severity": "High" if category in ["pricing", "fees", "default"] else "Medium",
            "reason": f"Clause does not mention expected terms for category '{category}' (missing keywords like {mandatory[:3]}).",
            "reg_text": reg_text,
            "suggested_fix": "Add explicit disclosure as per regulation."
        })
    else:
        if category == "pricing":
            # check for APR phrase or percentage
            if not re.search(r"\b(apr|annual percentage rate|annualized rate)\b", t) and not re.search(r"\b\d+\.?\d*\s*%\b", t):
                reg_text = reg_snippets[0]["text"] if reg_snippets else ""
                issues.append({
                    "status": "Partial",
                    "severity": "High",
                    "reason": "Pricing clause lacks a clear APR / annualized rate disclosure or percentage figure.",
                    "reg_text": reg_text,
                    "suggested_fix": "Include APR and a worked example in the KFS and the agreement."
                })

    return {"status": "Reviewed", "issues": issues}


# -------------------------
# JSON schema for validating LLM response (optional)
# -------------------------
LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "severity": {"type": "string"},
                    "reason": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "object"}
                    }
                },
                "required": ["status", "reason"]
            }
        }
    },
    "required": ["status", "issues"]
}


def _validate_json_with_schema(obj: Any):
    try:
        from jsonschema import validate
        validate(instance=obj, schema=LLM_RESPONSE_SCHEMA)
        return True, ""
    except Exception as e:
        # If jsonschema missing or validation fails, return False with message
        return False, str(e)


# -------------------------
# OpenAI call with retries (exponential backoff)
# -------------------------
def _call_openai_with_retries(messages: List[Dict[str, str]],
                              max_retries: int = 3,
                              backoff_factor: float = 1.5,
                              model: str = "gpt-3.5-turbo"):
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
    if not openai_key:
        raise EnvironmentError("OpenAI API key not found")

    try:
        import openai
        openai.api_key = openai_key
    except Exception as e:
        raise e

    attempt = 0
    last_exc = None
    while attempt < max_retries:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=700,
                temperature=0.0,
            )
            return resp
        except Exception as e:
            last_exc = e
            wait = backoff_factor ** attempt
            time.sleep(wait)
            attempt += 1

    # raise last exception if all retries failed
    raise last_exc


# -------------------------
# Main comparator (LLM-backed with fallback)
# -------------------------
def compare_clause_with_regs(clause_text: str,
                             category: str,
                             reg_snippets: List[Dict[str, Any]],
                             mode: str = "auto"):
    """
    Compare a clause with regulatory snippets.
    mode:
      - "auto": use LLM if API key present, otherwise heuristic
      - "heuristic": force heuristic comparator
      - "llm": force LLM comparator (falls back to heuristic only if LLM call fails)
    Returns: {"status":"Reviewed", "issues":[{status,severity,reason,suggested_fix,reg_text}, ...]}
    """
    mode = (mode or "auto").lower()
    if mode not in ["auto", "heuristic", "llm"]:
        mode = "auto"

    # Force heuristic
    if mode == "heuristic":
        return _heuristic_compare(clause_text, category, reg_snippets)

    # If explicit LLM requested but no key found, return heuristic annotated
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
    if mode == "llm" and not openai_key:
        res = _heuristic_compare(clause_text, category, reg_snippets)
        for it in res.get("issues", []):
            it["reason"] = "(LLM requested but OPENAI_API_KEY not set) " + it.get("reason", "")
        return res

    # Auto mode with no key => heuristic
    if openai_key is None and mode == "auto":
        return _heuristic_compare(clause_text, category, reg_snippets)

    # Build prompt messages for LLM
    try:
        system_msg = {
            "role": "system",
            "content": (
                "You are a regulatory compliance analyst specialized in Indian banking regulations (RBI/SEBI). "
                "Compare the provided contract clause with the regulatory excerpts and identify any compliance gaps, missing disclosures, or misleading wordings. "
                "Produce a concise, structured JSON object with keys 'status' and 'issues'."
            )
        }

        user_content = {
            "clause": clause_text,
            "category": category,
            "regulatory_excerpts": [
                {"title": r.get("title", ""), "text": r.get("text", "")[:2000]} for r in reg_snippets[:3]
            ]
        }

        user_msg_text = "ANALYZE_CLAUSE_JSON:\n" + json.dumps(user_content, ensure_ascii=False)
        messages = [system_msg, {"role": "user", "content": user_msg_text}]

        # Call OpenAI with retries
        resp = _call_openai_with_retries(messages, max_retries=3, backoff_factor=2.0)
        ans = resp["choices"][0]["message"]["content"].strip()

        # Try to extract a JSON object from the end of the model output
        parsed = None
        jmatch = re.search(r"\{.*\}\s*$", ans, flags=re.DOTALL)
        if jmatch:
            s = jmatch.group(0)
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None

        if parsed is None:
            # try parse entire response
            try:
                parsed = json.loads(ans)
            except Exception:
                parsed = None

        if parsed is None:
            # fallback to heuristic if parsing fails
            return _heuristic_compare(clause_text, category, reg_snippets)

        # Validate JSON structure (if jsonschema available)
        ok, _err = _validate_json_with_schema(parsed)
        if not ok:
            # if validation fails, fallback to heuristic
            return _heuristic_compare(clause_text, category, reg_snippets)

        # Normalize issues for legacy UI
        out_issues = []
        for it in parsed.get("issues", []):
            its = {
                "status": it.get("status", "Ambiguous"),
                "severity": it.get("severity", "Medium"),
                "reason": it.get("reason", ""),
                "suggested_fix": it.get("suggested_fix", ""),
                "reg_text": "",
            }
            citations = it.get("citations") or it.get("citations", []) or []
            if citations:
                reg_parts = []
                for c in citations:
                    t = c.get("title", "")
                    txt = c.get("snippet", c.get("text", "")) or ""
                    txt = txt[:300]
                    reg_parts.append(f"{t}: {txt}")
                its["reg_text"] = "\n---\n".join(reg_parts)
            else:
                its["reg_text"] = reg_snippets[0]["text"][:1000] if reg_snippets else ""
            out_issues.append(its)

        return {"status": "Reviewed", "issues": out_issues}
    except Exception:
        # On any error, fallback to heuristic comparator
        return _heuristic_compare(clause_text, category, reg_snippets)
