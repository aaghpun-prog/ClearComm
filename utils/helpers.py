import re

def clean_text(text: str) -> str:
    """Removes excessive whitespace and standardizes punctuation."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_report_issue(module_name: str, issue_type: str, message: str, text_segment: str) -> dict:
    """Standardizes the output format for issues found by modules."""
    return {
        "module": module_name,
        "type": issue_type,
        "message": message,
        "segment": text_segment
    }
