import re

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def format_prompt(question, answer=None):
    if answer:
        return f"Question: {question} Answer: {answer}"
    return f"Question: {question} Answer:"