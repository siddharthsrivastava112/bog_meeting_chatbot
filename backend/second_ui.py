from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# 🔐 Environment variables
GROQ_API_KEY = "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"
NEO4J_URL = "neo4j+s://1a7a75b4.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "OYX-RODrpLlzoDhrEa5vPZ7qhzMiK20No8UIYw3gn48"

# 🧠 Load sentence transformer for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🤖 Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# 🚀 FastAPI app
app = FastAPI(
    title="MNNIT Ordinance Assistant API",
    description="Ask questions about UG/PG ordinances and get detailed responses.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OR ["http://localhost:3000"] if you want to restrict to React app only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 📦 Request body model
class QueryRequest(BaseModel):
    query: str

# 📦 Response body model
class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    query_input = request.query.strip()

    if not query_input:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Create embedding
    embedding = model.encode(query_input).tolist()

    # 🧠 Vector search from Neo4j
    graph_retrieved_content = ""
    with GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        cypher_query = """
        CALL db.index.vector.queryNodes('chunkVectorIndex', 6, $queryEmbedding)
        YIELD node, score
        MATCH (c:Chunk)-[:HAS_EMBEDDING]->(node)
        RETURN c.text as Text, score as Score
        """
        records, _, _ = driver.execute_query(
            cypher_query,
            queryEmbedding=embedding,
            database_="neo4j"
        )

        for record in records:
            graph_retrieved_content += record['Text'] + "\n"

    # 💬 Use Groq API to get answer
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are MNNIT Allahabad ordinance assistant and handle all rules related to UG and PG. "
                    "You receive queries and always respond in English accurately, "
                    "thinking step by step and ensuring there are no spelling or grammatical mistakes. "
                    "Your main task is to search and summarize information based on a series of questions and answers:\n\n"
                    f"{graph_retrieved_content}\n\n"
                    f"I want you to build a step-by-step response or analysis about \"{query_input}\", "
                    "that reflects the content of the text. The response should be as complete as possible."
                )
            },
            {
                "role": "user",
                "content": f"{query_input}"
            }
        ],
        model="llama3-70b-8192"
    )

    generated_answer = response.choices[0].message.content

    return QueryResponse(answer=generated_answer)

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
