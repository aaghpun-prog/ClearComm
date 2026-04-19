from modules.length_control import analyze_length
from modules.homonym_detector import analyze_homonyms
from modules.info_gap_detector import analyze_info_gaps
from utils.preprocess import get_sentences, get_words

def generate_clarity_report(text: str) -> dict:
    """Orchestrates the analysis and generates a final score."""
    
    sentences = get_sentences(text)
    words = get_words(text)
    
    all_issues = []
    all_issues.extend(analyze_length(text, sentences, words))
    all_issues.extend(analyze_homonyms(text, words))
    all_issues.extend(analyze_info_gaps(text))
    
    # Calculate score
    base_score = 100
    penalty_per_issue = 5
    
    final_score = max(0, base_score - (len(all_issues) * penalty_per_issue))
    
    summary = f"Clarity score is {final_score}/100. We found {len(all_issues)} potential issues to review."
    if final_score >= 90:
        summary = "Excellent clarity! " + summary
    elif final_score >= 70:
        summary = "Good clarity, but some improvements can be made. " + summary
    else:
        summary = "Needs revision. Multiple clarity issues detected. " + summary
        
    return {
        "score": final_score,
        "summary": summary,
        "warnings": all_issues,
        "text": text
    }
