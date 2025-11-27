import faiss
import json
import numpy as np

FAISS_INDEX_PATH = "faiss_index.bin"
DOCUMENTS_MAP_PATH = "documents_map.json"

# Carregar o índice FAISS
vector_index = faiss.read_index(FAISS_INDEX_PATH)

# Carregar o mapa de documentos (necessário para entender o que os vetores representam)
with open(DOCUMENTS_MAP_PATH, "r", encoding="utf-8") as f:
    documents_map = json.load(f)

# --- Impressão das Propriedades ---

print(f"Index loaded successfully!")
print(f"  > Tipo do Índice: {type(vector_index)}")
print(f"  > Dimensão dos Vetores: {vector_index.d}")
print(f"  > Total de Vetores (chunks) no Índice: {vector_index.ntotal}")

# Visualizar o primeiro chunk (referência)
primeiro_id = list(documents_map.keys())[0]
primeiro_chunk = documents_map[primeiro_id]
print("\nPrimeiro Chunk (para referência):")
print(f"  > ID: {primeiro_chunk['id']}")
print(f"  > Fonte: {primeiro_chunk['metadata']['fonte']}")
print(f"  > Trecho do Texto: {primeiro_chunk['text'][:100]}...")