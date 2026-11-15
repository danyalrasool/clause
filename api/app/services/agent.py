from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from app.services.chat import RAGService

SYSTEM = "You are a helpful Clause AI assistant!"


class ClauseAgent:
    def __init__(self) -> None:
        self.rag = RAGService()

        @tool(response_format="content_and_artifact")
        def search_handbook(query: str) -> tuple[str, list]:
            """Search the Clause shop handbook for policies, prices and rules."""
            results = self.rag.store.similarity_search_with_score(
                query=query, k=self.rag.settings.top_k
            )
            if not results:
                return "Nothing Found", []

            text = []
            sources = []
            for doc, score in results:
                text.append(doc.page_content)
                sources.append(
                    {"page": doc.metadata.get("page"), "score": round(score, 4)}
                )

            return "\n\n".join(text), sources

        self.agent = create_agent(
            model=self.rag.llm,
            tools=[search_handbook],
            system_prompt=SYSTEM,
        )

    def ask(self, question: str) -> str:
        result = self.agent.invoke(
            input={"messages": [HumanMessage(content=question)]}
        )
        return result["messages"][-1].text

    def close(self) -> None:
        self.rag.close()