import os
import re
import requests
import heapq
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class TextRAGHandler:
    def __init__(self, vector_store_dir="vector_store", together_api_key=None, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.vector_store_dir = vector_store_dir
        # Updated: Using BAAI's bge-base-en-v1.5 embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_stores = {}
        self.TOGETHER_API_KEY = together_api_key or os.getenv("TOGETHER_API_KEY")
        self.MODEL_NAME = model_name

        if not self.TOGETHER_API_KEY:
            raise ValueError("Together API key missing. Set TOGETHER_API_KEY env variable or pass it explicitly.")

        self._load_all_vector_stores()

    def _load_all_vector_stores(self):
        print("[📁] Loading vector stores...")
        self.vector_stores.clear()

        if not os.path.exists(self.vector_store_dir):
            print(f"[!] Directory '{self.vector_store_dir}' does not exist.")
            return

        for folder_name in os.listdir(self.vector_store_dir):
            subfolder_path = os.path.join(self.vector_store_dir, folder_name)
            if os.path.isdir(subfolder_path):
                try:
                    db = FAISS.load_local(subfolder_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
                    self.vector_stores[folder_name] = db
                    print(f"[✓] Loaded vector DB: {folder_name}")
                except Exception as e:
                    print(f"[!] Failed to load {folder_name}: {e}")
        print(f"[📁] Vector stores available: {list(self.vector_stores.keys())}")

    def _query_together_ai(self, messages: list) -> str:
        try:
            response = requests.post(
                "https://api.together.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.TOGETHER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.MODEL_NAME,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                   },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"System error: {e}"

    def _generate_amplified_query(self, query: str, num_variations: int = 3) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant enhancing queries for better retrieval from a vector database. "
                    "Extract named entities such as BoG numbers, item numbers, names, years, etc., and generate "
                    "an amplified version of the query that repeats those terms in different factual phrasings. "
                    "Do not make things up. Your output should be a single expanded query containing multiple versions "
                    "of the original query phrasing the same key terms in different ways."
                )
            },
            {
                "role": "user",
                "content": f"Original user query: '{query}'\n\nGenerate a single amplified query with {num_variations} variations merged together."
            }
        ]
        response = self._query_together_ai(prompt)
        if response and not response.startswith("System error:"):
            return response.strip()
        return query

    def _extract_bog_folders_from_query(self, query: str):
        folder_names = list(self.vector_stores.keys())
        detected = set()

        bog_range_matches = re.findall(r"\bBoG\s*(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\b", query, flags=re.IGNORECASE)
        bog_single_matches = re.findall(r"\bBoG\s*(\d{1,3})\b", query, flags=re.IGNORECASE)

        year_range_matches = re.findall(r"\b(20\d{2})\s*(?:to|-)\s*(20\d{2})\b", query)
        year_single_matches = re.findall(r"\b(20\d{2})\b", query)

        folder_map_by_bog = {}
        folder_map_by_year = {}

        for folder in folder_names:
            bog_match = re.search(r'(\d{1,3})', folder)
            year_match = re.search(r'(20\d{2})', folder)

            if bog_match:
                bog_num = int(bog_match.group(1))
                folder_map_by_bog.setdefault(bog_num, []).append(folder)

            if year_match:
                year = int(year_match.group(1))
                folder_map_by_year.setdefault(year, []).append(folder)

        for start, end in bog_range_matches:
            for bog_num in range(int(start), int(end) + 1):
                detected.update(folder_map_by_bog.get(bog_num, []))

        for bog_str in bog_single_matches:
            bog_num = int(bog_str)
            detected.update(folder_map_by_bog.get(bog_num, []))

        for start, end in year_range_matches:
            for year in range(int(start), int(end) + 1):
                detected.update(folder_map_by_year.get(year, []))

        for year_str in year_single_matches:
            year = int(year_str)
            detected.update(folder_map_by_year.get(year, []))

        if not detected:
            print("[ℹ️] No BoG or year found. Using fallback 'db_faiss' vector store.")
            if "db_faiss" in self.vector_stores:
                return ["db_faiss"], True
            else:
                raise ValueError("[❌] No matching folders found and fallback 'db_faiss' does not exist.")

        print(f"[📌] Matched folders from query: {list(detected)}")
        return list(detected), False

    def _rerank_documents(self, docs, query, batch_size=100):
        # Preprocess query for BGE: prepend "query: "
        bge_query = f"query: {query}"
        query_embedding = self.embedding_model.embed_query(bge_query)
        all_ranked = []

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_ranked = []

            for doc in batch:
                doc_embedding = self.embedding_model.embed_query(f"passage: {doc.page_content[:512]}")
                similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
                batch_ranked.append((similarity, doc))

            all_ranked.extend(batch_ranked)

        top_docs = heapq.nlargest(100, all_ranked, key=lambda x: x[0])
        return [doc for _, doc in top_docs]

    def _query_with_context(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer user questions based only on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser Query:\n{query}\n\nAnswer the question based on the context above."
            }
        ]
        response = self._query_together_ai(messages)
        if not response:
            return "[Error] No response from Together AI."
        if response.startswith("System error:"):
            return response
        return response

    def handle_input(self, query: str, top_k: int = 100) -> str:
        self._load_all_vector_stores()

        try:
            store_keys, is_fallback = self._extract_bog_folders_from_query(query)
        except ValueError as ve:
            return str(ve)

        amplified_query = self._generate_amplified_query(query)
        print(f"[🧠] Amplified Query:\n{amplified_query}")
        print(f"[📂] Using vector stores: {store_keys}")

        if is_fallback:
            fallback_store = self.vector_stores["db_faiss"]
            top_docs = fallback_store.similarity_search(amplified_query, k=top_k)
        else:
            candidate_docs = []
            for store_key in store_keys:
                docs = self.vector_stores[store_key].similarity_search(amplified_query, k=top_k)
                for doc in docs:
                    doc.metadata["source_folder"] = store_key
                    candidate_docs.append(doc)

            print(f"[🔍] Reranking {len(candidate_docs)} documents in batches of 100...")
            top_docs = self._rerank_documents(candidate_docs, amplified_query)

        if not top_docs:
            return "[⚠️] No relevant context found in the selected documents."

        print(f"[📄] Top {len(top_docs)} Retrieved Chunks:\n")
        for i, doc in enumerate(top_docs, 1):
            print(f"#{i} [📂 Source: {doc.metadata.get('source_folder', 'unknown')}]:\n{doc.page_content[:300]}...\n{'-'*60}")

        combined_context = "\n\n".join(doc.page_content for doc in top_docs)
        return self._query_with_context(query, combined_context)
