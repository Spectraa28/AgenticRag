import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc["content"] for doc in raw_results if doc["content"]]
        
        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
        # Keep provenance intact after reranking. FlashRank returns text only, so
        # map each result back to the first matching Qdrant record.
        formatted_docs = []
        used_indexes = set()
        for content in reranked_contents:
            for index, document in enumerate(raw_results):
                if index not in used_indexes and document["content"] == content:
                    formatted_docs.append(document)
                    used_indexes.add(index)
                    break
    
    return {
        "documents": formatted_docs,
        "status": f"Found {len(formatted_docs)} technical context chunks.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
