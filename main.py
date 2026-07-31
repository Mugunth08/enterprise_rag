from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.retriever import retriever
from utils import get_model_from_gcp


llm = get_model_from_gcp()

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I don't have enough information to answer this question."

Context:
{context}

Question:
{question}
""")


def main():
    question = input("Enter your question: ")

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print(response.content)


if __name__ == "__main__":
    main()