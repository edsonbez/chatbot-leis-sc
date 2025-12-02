import os
import re
import shutil
import time
import json 
import numpy as np 
import faiss 
from bs4 import BeautifulSoup
from google import genai
from google.genai.errors import APIError 
import nltk 
from tenacity import retry, stop_after_attempt, wait_exponential 
import sys # Necessário para sys.exit(1) em caso de erro

# Certifique-se de que 'nltk' foi configurado com: python -c "import nltk; nltk.download('punkt')"

# --- CONFIGURAÇÃO DE SEGURANÇA CRÍTICA ---
# A chave de API DEVE ser lida de uma variável de ambiente por questões de segurança.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 

if not GEMINI_API_KEY:
    print("\nERRO DE CONFIGURAÇÃO CRÍTICO:")
    print("A chave de API do Gemini (GEMINI_API_KEY) não foi encontrada na variável de ambiente.")
    print("Por favor, defina a variável 'GEMINI_API_KEY' no seu sistema antes de executar este script.")
    sys.exit(1) # Força a interrupção da execução.

# Define o nome do modelo de embedding a ser usado 
EMBEDDING_MODEL = "text-embedding-004" 

# OTIMIZAÇÃO CRÍTICA DO CHUNK SIZE
CHUNK_SIZE_LIMIT = 1500

# Caminhos dos Arquivos
FAISS_INDEX_PATH = "faiss_index.bin"
DOCUMENTS_MAP_PATH = "documents_map.json"
CHUNKS_JSON_PATH = "chunks_processados.json"

# Inicializa o cliente para o Embedding
try:
    # Usa a chave lida da variável de ambiente
    client = genai.Client(api_key=GEMINI_API_KEY) 
except Exception as e:
    print(f"Erro ao inicializar o cliente Gemini no processar_leis.py: {e}")
    sys.exit(1)

# Função auxiliar para dividir a lista em lotes
def batch_list(iterable, n=1):
    """Divide um iterável em pedaços de tamanho n."""
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def generate_embeddings_batch(texts: list) -> list:
    """Gera embeddings em lote via API do Google com retries."""
    
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts, 
    )
    # Retorna a lista de ContentEmbedding objects
    return response.embeddings 

# --- FUNÇÃO DE NORMALIZAÇÃO DE ID CRÍTICA ---
def normalize_law_id(file_name: str) -> tuple[str, int]:
    """
    Extrai e normaliza o tipo, número e ano da lei a partir do nome do arquivo,
    criando um ID único padronizado (Ex: LEI_16852_2015).
    """
    
    name_lower = file_name.lower()
    
    # 1. Extrai o Tipo de Lei (LC, LEI, DECRETO)
    if 'complementar' in name_lower or re.search(r'lc\s*nº?', name_lower):
        law_type = 'LC'
    elif 'decreto' in name_lower:
        law_type = 'DECRETO'
    elif 'lei' in name_lower or 'promulgada' in name_lower or 'ordinária' in name_lower:
        law_type = 'LEI'
    else:
        law_type = 'DOC'
        
    # 2. Extrai o Número (limpo de pontos e vírgulas)
    # Procura por 'nº', 'n°', 'no' seguido por um número OU só o número (o grupo 2 captura o número).
    match_num = re.search(r'(nº|n°|no|numero)?\s*([\d\.]+)', name_lower)
    if match_num and match_num.group(2):
        law_number = re.sub(r'[^\d]', '', match_num.group(2))
    else:
        # Fallback: se não achar 'Nº', tenta a primeira sequência grande de dígitos (4 ou mais)
        match_num_fallback = re.search(r'(\d{4,})', name_lower)
        law_number = match_num_fallback.group(1).strip() if match_num_fallback else '0000'
        
    # 3. Extrai o Ano (o ano mais recente, se houver múltiplos)
    # Busca explicitamente por 4 dígitos de ano (19XX ou 20XX).
    anos = re.findall(r'(19\d{2}|20\d{2})', name_lower)
    
    # Se houver anos, pega o maior valor numérico para garantir a versão mais recente.
    law_year = max([int(a) for a in anos]) if anos else 0
    
    # Cria o ID ÚNICO e padronizado
    unique_id = f"{law_type}_{law_number}_{law_year}"
    
    return unique_id, law_year


# --- NOVA FUNÇÃO PARA ID ESPECÍFICO DO ARTIGO ---
def extract_article_id(chunk_text: str, base_id: str) -> str:
    """
    Tenta extrair o Artigo/Parágrafo do início do chunk para criar um ID único
    para busca determinística. Exemplo: LC_715_2018 + Art. 4º -> LC_715_2018_ART_4
    """
    
    # Regex para encontrar o início de um artigo, parágrafo ou caput
    # Prioriza o Art. N°
    # O padrão busca "Art.", "§", "Parágrafo único", ou "Caput" seguido por número/texto relevante.
    match_art = re.search(r'(Art\.\s*\d+º?|§\s*\d+º?|Parágrafo\s*único|Caput)', chunk_text, re.IGNORECASE)
    
    if match_art:
        # Normaliza o termo encontrado (ex: 'Art. 4º' -> 'ART_4')
        art_part = match_art.group(0).replace(' ', '_').replace('.', '').replace('º', '').upper()
        
        # Remove caracteres indesejados
        art_part = re.sub(r'[^\w_]', '', art_part)
        
        # Retorna o ID base + a parte do artigo
        return f"{base_id}_{art_part}"
    
    # Se não encontrar um artigo/parágrafo específico no começo, retorna o ID base.
    return base_id


# --- FUNÇÃO DE EXTRAÇÃO ATUALIZADA (ROBUSTA E REMOVENDO TAXADO) ---
def extrair_texto_de_html(caminho_arquivo):
    """
    Função que extrai o texto de forma mais agressiva de dentro do body,
    removendo scripts, styles, tags irrelevantes, e **texto revogado (taxado)**.
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            html_content = f.read()

        sopa = BeautifulSoup(html_content, 'html.parser')
        
        # 1. Remove elementos não-textuais (scripts, styles, cabeçalhos HTML)
        for script_or_style in sopa(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()
            
        # **NOVO: 1.5 - Remove conteúdo taxado/revogado**
        # Remove a tag <del> e elementos com classes comuns de revogação.
        for revoked_tag in sopa.find_all(['del', 'strike']):
            revoked_tag.decompose()
            
        # 2. Tenta encontrar a div principal do documento, ou usa o corpo inteiro.
        corpo_principal = sopa.find('main') or sopa.find('body')
        
        if corpo_principal:
            # Captura todos os blocos de texto principais (p, div, h1-h6) e concatena-os
            texto_parts = []
            
            # Itera sobre todas as tags que tipicamente contêm texto de artigo
            for tag in corpo_principal.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                # Extrai o texto e adiciona à lista
                texto_parts.append(tag.get_text(separator=' ', strip=True))
                
            texto_completo = ' '.join(texto_parts)
        else:
            return ""

        # 3. Remoção final de múltiplos espaços em branco
        texto_completo = re.sub(r'\s+', ' ', texto_completo)
        return texto_completo.strip()
        
    except Exception as e:
        print(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")
        return ""
    
# --- FUNÇÃO CRITICAMENTE CORRIGIDA PARA O CHUNKING (AGORA COM ANO DA LEI) ---
def chunk_text_optimized(text: str, file_name: str, law_year: int, max_chunk_size: int = CHUNK_SIZE_LIMIT) -> list[str]:
    """
    Divide o texto em chunks, garantindo coesão e um tamanho mínimo
    para evitar chunks vazios ou irrelevantes. Inclui o ano de publicação no cabeçalho.
    """
    sentences = nltk.sent_tokenize(text, language='portuguese')
    
    # Cabeçalho garante a fonte em cada chunk E adiciona o ano de publicação para contexto de vigência
    header = f"LEI JURÍDICA: {file_name} (Publicada em {law_year}). CONTEÚDO: "
    current_chunk = ""
    optimized_chunks = []
    
    # Define um tamanho mínimo para chunks (excluindo o cabeçalho)
    MIN_CHUNK_CONTENT_SIZE = 256
    
    # Loop para combinar sentenças em chunks de tamanho ideal
    for sentence in sentences:
        
        # 1. Verifica a condição de QUEBRA:
        # Se adicionar a próxima sentença (len(sentence)) for quebrar o limite máximo (max_chunk_size) E
        # se o chunk atual (current_chunk) já tiver um tamanho razoável (MIN_CHUNK_CONTENT_SIZE)
        if (len(current_chunk) + len(sentence) + 1 > max_chunk_size - len(header) and 
            len(current_chunk) > MIN_CHUNK_CONTENT_SIZE):
            
            # Adiciona o chunk completo
            optimized_chunks.append(header + current_chunk.strip())
            current_chunk = sentence # Começa um novo chunk com a nova sentença
        
        # 2. Caso contrário, ou se for o primeiro chunk, apenas adiciona a sentença ao chunk atual.
        else:
            current_chunk += " " + sentence
            
    # Adiciona o último chunk
    if current_chunk:
        final_chunk_content = current_chunk.strip()
        
        # Se o chunk final for maior que o mínimo OU se não houver chunks anteriores, indexa como novo.
        if len(final_chunk_content) > MIN_CHUNK_CONTENT_SIZE or not optimized_chunks:
             optimized_chunks.append(header + final_chunk_content)
             
        # CRÍTICO: Se o chunk final for pequeno (mas não vazio), anexa-o ao último chunk válido
        elif optimized_chunks:
            # Anexa ao último chunk para não perdê-lo.
            optimized_chunks[-1] += " " + final_chunk_content
            
    return optimized_chunks

# ----------------------------------------------------------------------
# FUNÇÃO ATUALIZADA: PROCESSAR, EMBEDDAR E SALVAR FAISS
# ----------------------------------------------------------------------
def processar_e_salvar_leis(caminho_da_pasta_raiz):
    """
    Processa arquivos HTML, gera chunks, calcula embeddings e salva FAISS.
    """
    
    print("\nProcessando arquivos e gerando chunks...")
    chunks_data = []
    id_counter = 1
    
    # === FASE 1: Extração e Chunking ===
    for raiz, pastas, arquivos in os.walk(caminho_da_pasta_raiz):
        for nome_arquivo in arquivos:
            if nome_arquivo.endswith('.html'):
                caminho_completo_arquivo = os.path.join(raiz, nome_arquivo)
                
                # --- NOVO: NORMALIZAÇÃO ---
                unique_id, ano_publicacao = normalize_law_id(nome_arquivo)
                # -------------------------
                
                texto_extraido = extrair_texto_de_html(caminho_completo_arquivo)
                
                if texto_extraido:
                    
                    # Usa o chunking otimizado, passando o ano_publicacao
                    chunks = chunk_text_optimized(texto_extraido, nome_arquivo, law_year=ano_publicacao, max_chunk_size=CHUNK_SIZE_LIMIT)
                    
                    for i, chunk in enumerate(chunks):
                        # --- NOVO: GERA ID ESPECÍFICO DO ARTIGO ---
                        # Cria um ID_UNICO específico, ex: LC_715_2018_ART_4
                        specific_id = extract_article_id(chunk, unique_id)
                        # ------------------------------------------

                        chunks_data.append({
                            "id": f"doc_{id_counter}",
                            "text": chunk,
                            "metadata": {
                                "fonte": nome_arquivo, 
                                "ano_publicacao": ano_publicacao,
                                "chunk_index": i,
                                "ID_UNICO": specific_id  # agora usa o ID Especifico
                            }
                        })
                        id_counter += 1

    total_chunks = len(chunks_data)
    print(f"\nTotal de {total_chunks} chunks prontos. Salvando em JSON...")

    # Salva os chunks processados em JSON
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    print(f"Dados salvos com sucesso em '{CHUNKS_JSON_PATH}'.")

    # === FASE 2: Geração de Embeddings e Construção do FAISS (COM BATCHING) ===
    if not chunks_data:
        print("Nenhum chunk para processar. Encerrando.")
        return
        
    texts = [item['text'] for item in chunks_data]
    
    # NOVO: Define o tamanho do lote para evitar o limite de payload
    BATCH_SIZE = 100 
    embeddings_list = []
    
    print(f"\nCalculando embeddings para {total_chunks} chunks via {EMBEDDING_MODEL} (API Gemini)...")
    
    start_time = time.time()
    
    try:
        # Itera sobre a lista de textos em lotes (batches) menores
        for i, batch_texts in enumerate(batch_list(texts, BATCH_SIZE)):
            print(f"        -> Processando lote {i+1}/{len(texts) // BATCH_SIZE + 1} de {len(batch_texts)} chunks...")
            
            # Gera os embeddings usando a função generate_embeddings_batch
            batch_embeddings = generate_embeddings_batch(batch_texts)
            embeddings_list.extend(batch_embeddings)
            time.sleep(1) # Pausa para evitar limites de taxa (rate limits)
            
        print(f"Geração de embeddings concluída em {time.time() - start_time:.2f} segundos.")
        
    except APIError as e:
        print(f"\nERRO CRÍTICO NA API DO GOOGLE (EMBEDDING): {e}")
        print("Verifique sua chave de API ou se atingiu o limite de taxa. Não foi possível criar o índice FAISS.")
        return
    # FIM DA FASE 2

    # 3. Criação e Salvamento do Índice FAISS e Mapeamento
    
    # Mapear a lista de objetos ContentEmbedding para extrair a lista de floats (.values)
    embeddings_values = [emb.values for emb in embeddings_list]
    
    embeddings_array = np.array(embeddings_values).astype('float32')
    dimension = embeddings_array.shape[1]
    
    print(f"Construindo índice FAISS. Dimensão: {dimension}...")

    # Cria o índice FAISS
    VECTOR_INDEX = faiss.IndexFlatL2(dimension)
    VECTOR_INDEX.add(embeddings_array)
    
    print("Índice FAISS construído com sucesso.")

    # Cria o Mapa de Documentos (ID -> Texto/Metadata)
    DOCUMENTS_MAP = {item['id']: item for item in chunks_data}
    
    # Salva o índice FAISS
    faiss.write_index(VECTOR_INDEX, FAISS_INDEX_PATH)
    print(f"Índice FAISS salvo em '{FAISS_INDEX_PATH}'.")
    
    # Salva o Mapa de Documentos
    with open(DOCUMENTS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(DOCUMENTS_MAP, f, ensure_ascii=False, indent=2)
    print(f"Mapa de Documentos salvo em '{DOCUMENTS_MAP_PATH}'.")
    
# --- EXECUÇÃO ---

# Configuração do caminho das leis
caminho_das_leis = r'C:\Users\etb1085\Leis Importantes' # <-- **ATENÇÃO: ATUALIZE ESTE CAMINHO**

# NOVO: Remove os arquivos antigos para garantir a re-ingestão completa
print("\n--- LIMPEZA INICIAL ---")

# Remoção de pasta antiga (se existir)
if os.path.exists("banco_de_leis"):
    shutil.rmtree("banco_de_leis") 
    print("Pasta 'banco_de_leis' removida (não é mais usada).")

# Remoção dos arquivos chave de índice
arquivos_a_remover = [CHUNKS_JSON_PATH, FAISS_INDEX_PATH, DOCUMENTS_MAP_PATH]
for arquivo in arquivos_a_remover:
    if os.path.exists(arquivo):
        os.remove(arquivo) 
        print(f"Arquivo '{arquivo}' antigo removido.")

print("--- INGESTÃO DE DADOS ---")
processar_e_salvar_leis(caminho_das_leis)

print("\nProcesso concluído. O arquivo 'faiss_index.bin' e o 'documents_map.json' estão prontos para o Chatbot.")