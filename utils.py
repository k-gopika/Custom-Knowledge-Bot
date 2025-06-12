# utils.py
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

def classify_intent(question: str) -> str:
    prompt = f"""Classify the following question into ONLY one of these categories:
    - yes_no
    - timeline
    - insight
    - guidance
    - general
    Respond with one of the above ONLY (lowercase, no punctuation).

    Question: {question}
    Intent:"""
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    return intent if intent in {"yes_no", "timeline", "insight", "guidance", "general"} else "general"

def build_tarot_prompt(question, intent, drawn_cards):
    card_text = "\n".join(
        f"{c['name']} ({c['orientation']}): {c['meanings'][c['orientation']]}" for c in drawn_cards
    )
    return (
        f"You are a helpful tarot expert. The user has asked a question.\n\n"
        f"**User's question:** {question}\n"
        f"**Detected Intent:** {intent}\n"
        f"**Cards drawn:**\n{card_text}\n\n"
        f"Give a detailed response based on the intent and meanings."
    )
