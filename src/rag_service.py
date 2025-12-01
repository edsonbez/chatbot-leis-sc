import os
import sys
import json
import time
import re
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai.errors import APIError
from typing import Union

# --- Configurações Iniciais ---


# --- FUNÇÃO GERADORA PARA PROCESSAR O STREAM ---
def text_generator(response_stream):
    """Função geradora que extrai e retorna o texto de cada chunk do stream."""
    for chunk in response_stream:
        # Acessa o texto de cada chunk. O .text é o acessor padrão do SDK.
        yield chunk.text 
# ------------------------------------------------------------

# Definição do modelo LLM a ser usado
LLM_MODEL = "gemini-2.5-flash"

# NOVO MODELO DE EMBEDDING (Google)
EMBEDDING_MODEL_GOOGLE = "text-embedding-004"

# Caminhos dos Arquivos
CHUNKS_JSON_PATH = "chunks_processados.json"
FAISS_INDEX_PATH = "faiss_index.bin"
DOCUMENTS_MAP_PATH = "documents_map.json"

# Variáveis globais
client = None
VECTOR_INDEX = None
DOCUMENTS_MAP = {}

# Instrução de Sistema (Persona e Regras de Grounding)
SYSTEM_INSTRUCTION = (
    """Você é um assistente jurídico especializado em análise e extração de informações das Leis e documentos fornecidos no contexto (corpus). 
    Sua área de atuação é estritamente voltada para as Leis e documentos do Estado de Santa Catarina. Siga rigorosamente as seguintes regras:

    1. **Fundamentação Exclusiva:** Baseie-sua resposta SOMENTE e EXCLUSIVAMENTE nas informações contidas nos documentos jurídicos que lhe foram fornecidos como contexto.
    2. **Citação Obrigatória:** Para cada informação ou trecho de lei utilizado, você DEVE citar a fonte (o nome do arquivo) que está no final da linha do contexto. As citações devem ser no formato: (NOME_DO_ARQUIVO).
    3. **Transparência na Ausência:** Se a resposta para a pergunta do usuário não for encontrada no contexto, você DEVE responder de forma clara e cortês: "A informação solicitada não foi encontrada nos documentos jurídicos que compõem a base de dados (corpus legal)."
    4. **Tratamento de Alterações (CRÍTICO):** Se o contexto incluir diferentes redações de um mesmo artigo ou a data de alteração, você DEVE listar *todas as redações* (ou a mais recente, se for muito longa) e **obrigatoriamente citar as leis que promoveram as alterações** (ex: Redação dada pela LC 789, de 2021). **Se nenhuma lei alteradora for citada no contexto para o artigo, use o ano de publicação da lei (que estará no cabeçalho) para confirmar que a redação apresentada é a vigente.**
    5. **Formato Detalhado:** Enumere todos os objetivos, princípios e ações detalhadas encontrados no contexto.
    6. **Evite Conhecimento Externo:** Não utilize qualquer conhecimento que não tenha sido extraído do contexto fornecido."""
)

def initialize_gemini_client(api_key: str):
    """Inicializa o cliente da API Gemini com a chave fornecida pelo frontend."""
    global client
    if client is None:
        try:
            # ATENÇÃO: Agora usa o 'api_key' recebido como parâmetro.
            client = genai.Client(api_key=api_key) 
            print("Cliente Gemini inicializado.")
        except Exception as e:
            print(f"Erro ao inicializar o cliente Gemini: {e}")
            raise

# --- Inicialização FAISS/Embedding com Persistência ---
def initialize_chromadb(api_key: str):
    """Tenta carregar o índice FAISS salvo."""
    global VECTOR_INDEX, DOCUMENTS_MAP, client
    
    if VECTOR_INDEX is not None:
        print("Índice FAISS já carregado.")
        return

    # 1. Tentar Carregar o Índice e o Mapeamento
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_MAP_PATH):
        try:
            print(f"Carregando índice FAISS de '{FAISS_INDEX_PATH}'...")
            VECTOR_INDEX = faiss.read_index(FAISS_INDEX_PATH)
            
            with open(DOCUMENTS_MAP_PATH, "r", encoding="utf-8") as f:
                DOCUMENTS_MAP = json.load(f)
                
            print(f"Índice FAISS e Mapa de Documentos carregados com sucesso. Total de {len(DOCUMENTS_MAP)} documentos.")
            # PASSA A CHAVE AQUI
            initialize_gemini_client(api_key) 
            return
            
        except Exception as e:
            print(f"AVISO: Falha ao carregar índice FAISS ou Mapa de Documentos ({e}).")
            raise Exception("Índice FAISS corrompido ou incompatível. É necessária a re-ingestão com o 'processar_leis.py'.")


    # Se a carga falhar porque os arquivos não existem:
    print(f"\nERRO CRÍTICO: O arquivo de índice FAISS '{FAISS_INDEX_PATH}' ou o mapa não existe.")
    raise Exception("Índice FAISS ausente. Por favor, execute o 'processar_leis.py' com o novo modelo de embedding.")


# --- Funções de Contexto e Chat ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
def _query_faiss(query_embedding: np.ndarray, n_results: int):
    """Função que executa a consulta real ao FAISS com retry."""
    global VECTOR_INDEX
    if VECTOR_INDEX is None:
        raise Exception("Índice FAISS não inicializado.")
    
    # A matriz de embedding deve ser 1 x Dimensão (e.g., 1 x 768)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
        
    D, I = VECTOR_INDEX.search(query_embedding, n_results)
    return I[0], D[0]

# --- Extração do ID ÚNICO via LLM (Com ano) ---
def extract_unique_id(user_query: str) -> Union[str, None]:
    """Usa o Gemini para extrair o ID ÚNICO de uma lei na query do usuário."""
    global client
    
    # REMOVIDO: if client is None: initialize_gemini_client()
        
    EXTRACT_PROMPT = f"""
    ANALISE a seguinte pergunta do usuário e TENTE extrair o identificador único da lei ou decreto, INCLUINDO O ANO.
    O formato de retorno DEVE ser EXATAMENTE um dos seguintes:
    - Se a lei for complementar: LC_NUMERO_ANO (Ex: LC_656_2015)
    - Se for lei ordinária/promulgada: LEI_NUMERO_ANO (Ex: LEI_16852_2015)
    - Se for decreto: DECRETO_NUMERO_ANO (Ex: DECRETO_123_2023)
    
    Se não for possível identificar o TIPO, NÚMERO E ANO, retorne SOMENTE a palavra: NULO.
    
    PERGUNTA DO USUÁRIO: {user_query}
    
    INSTRUÇÃO: Retorne APENAS o ID ÚNICO COMPLETO (FORMATO_NUMERO_ANO) ou NULO, sem explicações ou pontuações.
    ID ÚNICO:
    """
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=EXTRACT_PROMPT,
            config={"temperature": 0.0}
        )
        
        extracted_id = response.text.strip().upper()
        
        if extracted_id.startswith(('LC_', 'LEI_', 'DECRETO_')) and len(extracted_id.split('_')) == 3:
            print(f"ID ÚNICO COMPLETO extraído pelo LLM: {extracted_id}")
            return extracted_id
        else:
            return None
            
    except Exception as e:
        print(f"Erro na extração do ID ÚNICO pelo LLM: {e}. Retornando Nulo.")
        return None

# --- NOVO e FINAL: Extração do número da lei via Regex (Fallback Fuzzy) ---
def extract_law_number_from_query(query: str) -> Union[str, None]:
    """Extrai apenas o número da lei/decreto da consulta usando Regex, como fallback fuzzy."""
    
    # NOVO REGEX: 
    # 1. Procura por (LC|LEI|DECRETO)
    # 2. Ignora texto opcional e N°
    # 3. Captura o número da lei/decreto ([\d\.]+)
    match = re.search(r'(LC|LEI|DECRETO)\s*(?:[^\d]+)?\s*([\d\.]+)', query, re.IGNORECASE)

    if match:
        # Pega o segundo grupo, que é o número
        raw_number = match.group(2)
        # Remove pontos e vírgulas, deixando apenas os dígitos
        clean_number = re.sub(r'[^0-9]', '', raw_number)
        
        if clean_number:
            print(f"Número da Lei extraído via Regex (Fuzzy): {clean_number}")
            return clean_number
    
    # Fallback Adicional: Se não encontrou 'LEI/LC', tenta buscar por 'Nº' mas sem priorizar números pequenos.
    # Evita que 'Art. 5' seja pego.
    match_fallback = re.search(r'Nº?\s*([\d\.]{4,})', query, re.IGNORECASE) # Captura apenas números com 4 ou mais dígitos.
    if match_fallback:
        clean_number = re.sub(r'[^0-9]', '', match_fallback.group(1))
        if clean_number:
            print(f"Número da Lei extraído via Regex (Fallback N°): {clean_number}")
            return clean_number
            
    return None

def rewrite_query_with_context(user_query: str, history: list) -> str:
    """Usa o Gemini para reescrever uma consulta vaga baseada no contexto do histórico."""
    global client
    
    # REMOVIDO: if client is None: initialize_gemini_client()
    
    context_turns = []
    recent_history = history[:-1] 
    
    # Pega os últimos 4 elementos
    for role, content in recent_history[-4:]: 
        context_turns.append(f"[{role}]: {content}")
        
    context_string = "\n".join(context_turns)

    REWRITE_PROMPT = f"""
    Com base no HISTÓRICO da conversa abaixo, reescreva a ÚLTIMA PERGUNTA do usuário de forma completa, explícita e clara.
    O objetivo da reescrita é criar uma ÚNICA frase que possa ser usada em um sistema de busca de documentos.
    A frase reescrita DEVE ser autocontida e incluir o nome completo da lei ou decreto a que se refere (se aplicável), SEM AS CITAÇÕES EM PARÊNTESES.
    
    EXEMPLO:
    Histórico: [user]: Quais os objetivos da Lei 741? [assistant]: Os objetivos são...
    Última Pergunta: E o artigo 5º?
    FRASE REESCRITA ESPERADA: Qual o conteúdo do Artigo 5º da Lei Complementar Nº 741?
    
    INSTRUÇÃO: Retorne APENAS a frase reescrita, sem explicações ou pontuações adicionais.
    
    ---
    HISTÓRICO DA CONVERSA:
    {context_string}
    
    ÚLTIMA PERGUNTA DO USUÁRIO: {user_query}
    ---
    FRASE REESCRITA:
    """
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=REWRITE_PROMPT,
            config={"temperature": 0.0}
        )
        rewritten_query = response.text.strip().replace('\n', ' ')
        
        print(f"Query Reescrita: {rewritten_query}")
        
        return rewritten_query
        
    except Exception as e:
        print(f"Erro na reescrita da query pelo LLM: {e}. Usando query original.")
        return user_query


def get_context(query_text: str, history: list, k: int = 10) -> tuple[str, set]:
    """Busca os documentos mais relevantes no índice FAISS usando a Busca Híbrida (ID Único ou Número Fuzzy)."""
    global VECTOR_INDEX, DOCUMENTS_MAP, client
    
    if VECTOR_INDEX is None:
        return "", set()
    
    # REMOVIDO: if client is None: initialize_gemini_client()
        
    # ----------------------------------------------------------------------
    # *** BLINDAGEM ROBUSTA POR ID ÚNICO (Busca Forçada no Mapa) ***
    # ----------------------------------------------------------------------
    
    # 1. Tenta extrair ID ÚNICO COMPLETO (LLM - Preciso)
    target_unique_id = extract_unique_id(query_text) 
    
    # 2. Fallback: Se o LLM falhou (não encontrou o ano), tenta extrair só o número via Regex (Fuzzy)
    target_law_number = None
    if target_unique_id is None:
        target_law_number = extract_law_number_from_query(query_text)

    search_key = target_unique_id if target_unique_id else target_law_number

    if search_key:
        forced_context_parts = []
        forced_cited_sources = set()
        
        print(f"DEBUG: Tentando Busca Híbrida/Forçada para chave: {search_key}") 
        
        # Itera sobre o mapa de documentos para encontrar a chave
        for doc_id, item in DOCUMENTS_MAP.items():
            current_unique_id = item['metadata'].get('ID_UNICO', '')
            
            is_match = False
            
            if target_unique_id and current_unique_id == target_unique_id:
                # MATCH 1: Busca Exata (LLM forneceu o ID completo)
                is_match = True
            elif target_law_number and target_law_number in current_unique_id:
                # MATCH 2: Busca Fuzzy (Regex forneceu apenas o número, mas o ID o contém)
                is_match = True
                
            if is_match:
                doc = item['text']
                source = item['metadata'].get('fonte', 'Fonte Desconhecida')
                
                source_name = os.path.basename(source) if source else 'Fonte Desconhecida'
                
                forced_cited_sources.add(source_name)
                # Coleta todos os chunks dessa lei
                forced_context_parts.append(f"[Contexto Forçado ({source_name})]: {doc.strip()}")
                
        # 3. Se a busca forçada encontrou a lei (exata ou fuzzy), RETORNA APENAS O CONTEXTO FORÇADO.
        if forced_context_parts:
            match_type = "ID_UNICO" if target_unique_id else "Número Fuzzy"
            print(f"SUCESSO CRÍTICO: Lei encontrada via busca forçada por {match_type}. Ignorando FAISS e Query Rewrite.")
            forced_context = "\n\n---\n\n".join(forced_context_parts)
            return forced_context, forced_cited_sources

    # ----------------------------------------------------------------------
    # *** FIM DA BLINDAGEM ROBUSTA *** # ----------------------------------------------------------------------

    # --- Código FAISS Normal (Fallback) ---
    # 4. APLICAR REESCRITA DE QUERY SOMENTE SE A BUSCA BLINDADA FALHOU OU NÃO FOI ATIVADA.
    # ISSO ECONOMIZA TEMPO DE API.
    search_query = rewrite_query_with_context(query_text, history)
    
    final_search_query = search_query
    n_results_faiss = 10 
    
    # Se a busca forçada falhou, mas tínhamos um ID, otimiza a query FAISS (Embedding)
    if search_key: # Se tentamos buscar uma lei, mas não a encontramos no mapa (ex: lei não existe), otimizamos a busca FAISS
        # Cria um prefixo que força o modelo a focar no ID e nos termos chave.
        forced_prefix = (
            f"BUSCAR DETALHES DA LEI IDENTIFICADA COMO {search_key}. "
            f"Foco na Ementa, Artigo 1º e Objeto. "
        )
        final_search_query = forced_prefix + search_query
        print(f"Query Otimizada para Busca Exata (k=10): {final_search_query}")
        n_results_faiss = 10
    
    # ------------------------------------------------------------------
        
    # 5. Geração de Embedding da FINAL_SEARCH_QUERY via API do Google
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_GOOGLE,
            contents=[final_search_query], 
        )
        
        query_embedding = np.array(response.embeddings[0].values).astype('float32')
        
    except Exception as e:
        print(f"ERRO DE EMBEDDING CRÍTICO da QUERY (Google): {type(e).__name__} - {e}")
        return "", set()
    
    # 6. Busca no FAISS
    try:
        indices, _ = _query_faiss(query_embedding, n_results_faiss)
        
        context_parts = []
        cited_sources = set()
        
        for i, index in enumerate(indices):
            
            if index < len(DOCUMENTS_MAP):
                doc_id = list(DOCUMENTS_MAP.keys())[index]
                item = DOCUMENTS_MAP[doc_id]
                
                doc = item['text']
                source = item['metadata'].get('fonte', 'Fonte Desconhecida')
                
                full_source_name = os.path.basename(source) if source else 'Fonte Desconhecida'
                cited_sources.add(full_source_name)
                
                context_parts.append(f"[Contexto {i+1} ({full_source_name})]: {doc.strip()}")
            
        context = "\n\n---\n\n".join(context_parts)
        
        return context, cited_sources
        
    except Exception as e:
        print(f"Erro persistente durante a busca de contexto (após retries) no FAISS: {e}")
        return "", set()

# O restante das funções (classify_intent e get_response) permanece inalterado, exceto pela remoção das inicializações internas.
def classify_intent(query_text: str, history: list) -> str:
    """Classifica a intenção da pergunta do usuário."""
    global client
    
    # REMOVIDO: if client is None: initialize_gemini_client()
        
    contextualized_query = query_text
    
    if len(history) > 1:
        last_user_query = next((content for role, content in reversed(history) if role == 'user' and content != query_text), None)
        if last_user_query:
            contextualized_query = f"Com base na pergunta anterior ('{last_user_query}'), a nova pergunta é: '{query_text}'"

    CLASSIFICATION_PROMPT = f"""
    CLASSIFIQUE a seguinte pergunta/frase em UMA ÚNICA palavra, escolhendo estritamente entre: JURIDICA ou NAO_JURIDICA.
    
    JURIDICA: Se a pergunta for sobre leis, decretos, regulamentos, termos jurídicos, ou qualquer assunto relacionado à legislação de Santa Catarina.
    NAO_JURIDICA: Se for sobre conhecimento geral (história, geografia, previsão do tempo, pessoas, etc.) ou assuntos não-legais.
    
    PERGUNTA A SER CLASSIFICADA: {contextualized_query}
    """
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=CLASSIFICATION_PROMPT,
            config={"temperature": 0.0}
        )
        
        classification = response.text.strip().upper()
        
        if classification in ["JURIDICA", "NAO_JURIDICA"]:
            return classification
        else:
            return "JURIDICA"
        
    except Exception:
        return "JURIDICA"

# A função get_response aceita APENAS 2 argumentos, conforme necessário.
def get_response(user_query: str, history_for_context: list) -> tuple[Union[object, str], set]:
    """Gera a resposta do chatbot usando o contexto recuperado e o modelo Gemini (Modo Stream RAG)."""
    global client
    print("Chatbot: Pensando...")

    # 1. FILTRO DE INTENÇÃO
    intent = classify_intent(user_query, history_for_context)
    
    if intent == 'NAO_JURIDICA':
        fora_de_escopo_msg = (
            "A sua pergunta foi classificada como de conhecimento geral/não-jurídica. "
            "Minha função é estritamente a consultoria de leis estaduais de Santa Catarina. "
            "Por não possuir uma base de dados com fatos atuais ou informações não-jurídicas, "
            "não posso fornecer a informação solicitada. Por favor, reformule sua pergunta para um tema legal."
        )
        # Retorna uma string e um set vazio.
        return fora_de_escopo_msg, set()

    
    # 2. Recuperação de Contexto (RAG - CHAMA O FAISS)
    context, cited_sources = get_context(user_query, history_for_context, k=10)
    
    if not context:
        return "Desculpe, ocorreu um erro ao buscar os documentos. Por favor, tente novamente.", set()

    # 3. CONSTRUÇÃO DO PROMPT COM CONTEXTO RAG INJETADO
    rag_prompt = f"""
    CONTEXTO JURÍDICO (Leis de Santa Catarina - Inclui Leis Consolidadas com Alterações):
    ---
    {context}
    ---
    
    PERGUNTA DO USUÁRIO (Responda a esta pergunta com o contexto acima, aplicando a Regra 4):
    {user_query}
    
    INSTRUÇÃO ESPECIAL: Se a pergunta for sobre um Artigo de uma lei, detalhe todas as redações encontradas no contexto e cite as leis alteradoras.
    Assegure-se de citar as fontes em sua resposta (ex: (LEI Nº 19.142.html)).
    """
    
    # 4. CRÍTICO: CONSTRUÇÃO DO CONTEÚDO PARA STREAM - SOMENTE O PROMPT RAG
    contents = [
        {"role": 'user', "parts": [{"text": rag_prompt}]}
    ]

    try:
        # 5. Chamada à API Gemini USANDO generate_content_stream
        response_stream = client.models.generate_content_stream(
            model=LLM_MODEL,
            contents=contents,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.3
            }
        )
        
        # Retorna o objeto stream e o set de fontes.
        return response_stream, cited_sources
            
    except Exception as e:
        error_message = f"Ocorreu um erro no processamento do Chat Gemini (Stream). Erro: {e}"
        return error_message, cited_sources