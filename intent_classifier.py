from langchain_ollama import ChatOllama

# Connect to your local LLaMA 3 model
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

    valid = {"yes_no", "timeline", "insight", "guidance", "general"}
    return intent if intent in valid else "general"

if __name__ == "__main__":
    print("Ask a question (type 'exit' to quit):")
    while True:
        question = input("\n> ")
        if question.lower() in {"exit", "quit"}:
            break
        intent = classify_intent(question)
        print(f"Intent: {intent}")


