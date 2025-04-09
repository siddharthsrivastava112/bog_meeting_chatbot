import os
import re
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class TextRAGHandler:
    def __init__(self, vector_store_dir="vector_store", together_api_key=None, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.vector_store_dir = vector_store_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"System error: {e}"

    def _extract_bog_folders_from_query(self, query: str):
        folder_names = list(self.vector_stores.keys())
        detected = set()

        bog_mentions = re.findall(r"\bBoG\s*(\d{1,3})\b", query, flags=re.IGNORECASE)
        year_mentions = re.findall(r"\b(20\d{2})\b", query)

        if not bog_mentions and not year_mentions:
            raise ValueError("[❌] No BoG number or year found in the query. Please mention something like 'BoG 54' or '2021'.")

        folder_map_by_bog = {}
        folder_map_by_year = {}

        for folder in folder_names:
            bog_match = re.search(r'(\d{1,3})', folder)
            year_match = re.search(r'(20\d{2})', folder)

            if bog_match:
                bog_num = bog_match.group(1)
                folder_map_by_bog.setdefault(bog_num, []).append(folder)

            if year_match:
                year = year_match.group(1)
                folder_map_by_year.setdefault(year, []).append(folder)

        for bog_num in bog_mentions:
            if bog_num in folder_map_by_bog:
                detected.update(folder_map_by_bog[bog_num])
            else:
                print(f"[⚠️] BoG {bog_num} not found in folder names.")

        for year in year_mentions:
            if year in folder_map_by_year:
                detected.update(folder_map_by_year[year])
            else:
                print(f"[⚠️] Year {year} not found in folder names.")

        if not detected:
            raise ValueError("[❌] No matching folders detected in the query.")

        print(f"[📌] Matched folders from query: {list(detected)}")
        return list(detected)

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

    def handle_input(self, query: str, top_k: int = 200) -> str:
        self._load_all_vector_stores()
        try:
            store_keys = self._extract_bog_folders_from_query(query)
        except ValueError as ve:
            return str(ve)

        all_docs = []
        seen = set()
        for store_key in store_keys:
            print(f"[→] Using vector store: {store_key}")
            docs = self.vector_stores[store_key].as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(query)

            for i, doc in enumerate(docs, start=1):
                if doc.page_content not in seen:
                    print(f"\n📄 Chunk {i} from '{store_key}':\n{'-' * 50}\n{doc.page_content}\n{'-' * 50}\n")
                    all_docs.append(doc)
                    seen.add(doc.page_content)

        combined_context = "\n\n".join(doc.page_content for doc in all_docs)

        if not combined_context.strip():
            return "[⚠️] No relevant context found in the specified folders."

        return self._query_with_context(query, combined_context)
