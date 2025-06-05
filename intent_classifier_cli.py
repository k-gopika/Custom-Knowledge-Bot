import os
from langchain_ollama import ChatOllama

# Load Mistral model via Ollama
llm = ChatOllama(model="mistral")

def classify_intent_mistral(question):
    prompt = f"""
    You are an intent classification expert. Classify the user's question as one of:
    [yes_no, timeline, insight, guidance, general].
    
    Question: "{question}"
    Intent:
    """

    response = llm.invoke(prompt)
    return response.strip()

# --- CLI Loop ---
if __name__ == "__main__":
    print("\n🔮 Welcome to the Mistral-Powered Intent Classifier 🔮")
    print("Type your question or type 'exit' to quit.")

    while True:
        user_input = input("\n📝 Enter your question: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        intent_mistral = classify_intent_mistral(user_input)
        print(f"\n🎯 Mistral-Classified Intent: {intent_mistral}")
