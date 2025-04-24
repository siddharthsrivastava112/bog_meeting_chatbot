# import os
# import fitz  # PyMuPDF
# import spacy
# from tqdm import tqdm
# from neo4j import GraphDatabase
# import logging
# from typing import Dict, List, Tuple
# import re

# # ----------------- Config -----------------
# data_folder = "data"
# uri = "neo4j+s://1a7a75b4.databases.neo4j.io"
# user = "neo4j"
# password = "OYX-RODrpLlzoDhrEa5vPZ7qhzMiK20No8UIYw3gn48"

# # Load English NLP model
# nlp = spacy.load("en_core_web_sm")

# # ----------------- Setup -----------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def extract_text_from_pdf(pdf_path: str) -> str:
#     """Robust PDF text extraction with page filtering"""
#     try:
#         text = []
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 # Skip reference/bibliography pages
#                 if not is_reference_page(page.get_text()):
#                     text.append(page.get_text())
#         return "\n".join(text)
#     except Exception as e:
#         logger.error(f"Failed to extract text from {pdf_path}: {e}")
#         return ""

# def is_reference_page(text: str) -> bool:
#     """Identify reference/bibliography pages"""
#     ref_keywords = ["references", "bibliography", "works cited"]
#     return any(keyword in text.lower()[:100] for keyword in ref_keywords)

# class AdvancedGraphBuilder:
#     def __init__(self):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))
#         self._setup_constraints()
#         self.entity_cache = set()
    
#     def _setup_constraints(self):
#         """Create database constraints once"""
#         with self.driver.session() as session:
#             session.run("""
#             CREATE CONSTRAINT IF NOT EXISTS 
#             FOR (e:Entity) 
#             REQUIRE e.name IS UNIQUE
#             """)
    
#     def extract_entities_relations(self, text: str) -> Dict[str, List]:
#         """Advanced NLP-based entity and relationship extraction"""
#         doc = nlp(text)
#         entities = set()
#         relationships = []
        
#         # Extract named entities
#         for ent in doc.ents:
#             if ent.label_ in ["ORG", "PERSON", "GPE", "NORP"]:
#                 entities.add((ent.text, ent.label_))
        
#         # Extract relationships using dependency parsing
#         for sent in doc.sents:
#             # Subject-Verb-Object patterns
#             subj = None
#             obj = None
#             relation = None
            
#             for token in sent:
#                 if "subj" in token.dep_:
#                     subj = token.text
#                 elif "obj" in token.dep_:
#                     obj = token.text
#                 elif token.pos_ == "VERB":
#                     relation = token.lemma_
            
#             if subj and obj and relation:
#                 relationships.append({
#                     'source': subj,
#                     'target': obj,
#                     'type': relation
#                 })
#                 entities.update([subj, obj])
        
#         # Convert to proper node format
#         nodes = [{'name': ent[0], 'type': ent[1]} for ent in entities]
        
#         return {
#             'nodes': nodes,
#             'relationships': relationships
#         }
    
#     def store_graph(self, graph_data: Dict[str, List]):
#         """Optimized bulk insertion"""
#         with self.driver.session() as session:
#             # Insert nodes with types
#             for node in graph_data['nodes']:
#                 if node['name'] not in self.entity_cache:
#                     session.run("""
#                         MERGE (n:Entity {name: $name})
#                         SET n.type = $type
#                         """, 
#                         name=node['name'],
#                         type=node['type'])
#                     self.entity_cache.add(node['name'])
            
#             # Insert relationships
#             for rel in graph_data['relationships']:
#                 session.run("""
#                     MATCH (a:Entity {name: $source})
#                     MATCH (b:Entity {name: $target})
#                     MERGE (a)-[r:RELATIONSHIP {type: $type}]->(b)
#                     """,
#                     source=rel['source'],
#                     target=rel['target'],
#                     type=rel['type'])
    
#     def close(self):
#         self.driver.close()

# def process_documents():
#     builder = AdvancedGraphBuilder()
#     pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
#     if not pdf_files:
#         logger.error("No PDF files found in data folder")
#         return
    
#     for filename in tqdm(pdf_files, desc="Processing PDFs"):
#         try:
#             filepath = os.path.join(data_folder, filename)
            
#             # 1. Extract text (skip references)
#             text = extract_text_from_pdf(filepath)
#             if not text:
#                 logger.warning(f"No text extracted from {filename}")
#                 continue
            
#             # 2. Extract entities and relationships
#             graph_data = builder.extract_entities_relations(text[:100000])  # Process first 100k chars
            
#             if not graph_data['nodes']:
#                 logger.warning(f"No entities found in {filename}")
#                 continue
            
#             # 3. Store in Neo4j
#             builder.store_graph(graph_data)
#             logger.info(f"Inserted {len(graph_data['nodes'])} nodes from {filename}")
            
#         except Exception as e:
#             logger.error(f"Error processing {filename}: {e}")
    
#     builder.close()
#     logger.info(f"Total unique entities stored: {len(builder.entity_cache)}")

# if __name__ == "__main__":
#     logger.info("Starting advanced knowledge graph construction")
#     process_documents()
#     logger.info("Processing completed")