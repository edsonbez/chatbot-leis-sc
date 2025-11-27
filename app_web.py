import streamlit as st
import time
import sys
import os

# CRÍTICO: Importa o módulo chatbot inteiro.
try:
    import chatbot
except ImportError:
    st.error("Erro: Não foi possível importar o módulo 'chatbot.py'. Verifique se o arquivo existe e se não contém erros de sintaxe.")
    st.stop()


# --- FUNÇÃO GERADORA PARA PROCESSAR O STREAM ---
def text_generator(response_stream):
    """Função geradora que extrai e retorna o texto de cada chunk do stream."""
    for chunk in response_stream:
        # Acessa o texto de cada chunk. O .text é o acessor padrão do SDK.
        yield chunk.text 
# ------------------------------------------------------------


# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Chatbot Jurídico de SC",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Chatbot de Consultoria Jurídica - Leis de Santa Catarina")
st.markdown("Este assistente é especializado em responder perguntas **exclusivamente** com base nas leis estaduais de Santa Catarina.")

# --- Inicialização da Base de Dados e Cliente (Caching) ---
@st.cache_resource
def setup_chatbot():
    """
    Inicializa os clientes da API e do Banco de Dados Vetorial.
    Tudo é executado APENAS UMA VEZ.
    """
    
    # 1. RECUPERAÇÃO SEGURA DA CHAVE DE API
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "Erro CRÍTICO: A chave 'GEMINI_API_KEY' não foi encontrada nos Segredos (Secrets) do Streamlit. Por favor, configure-a no painel do Streamlit Cloud."
    
    # 2. INICIALIZAÇÃO DO CHATBOT (FAISS e Cliente Gemini)
    try:
        # Chama a função principal de inicialização no chatbot.py, passando a chave segura
        # initialize_chromadb chamará initialize_gemini_client internamente.
        chatbot.initialize_chromadb(api_key) 
        
        return "Pronto para consultas."
    except Exception as e:
        # Captura e retorna erros de inicialização (ex: arquivo FAISS ausente/corrompido)
        return f"Erro na inicialização: {e}. Verifique se o índice FAISS e o Mapa de Documentos existem e não estão corrompidos."


# Exibe o status de inicialização e interrompe se houver erro
status = setup_chatbot()
if "Erro" in status:
    st.error(status)
    st.stop()

# --------------------------------------------------------------------------
# Função de limpeza do histórico.
def clear_chat_history():
    """Limpa a sessão de chat do Streamlit."""
    
    # Limpa o histórico do Streamlit
    st.session_state.messages = []
    # Adiciona a mensagem inicial novamente
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu assistente jurídico especializado em Leis de Santa Catarina. Como posso ajudar na sua consulta legal hoje?"})

    st.rerun() 
# --------------------------------------------------------------------------

# 1. INICIALIZA O ESTADO DA SESSÃO PARA O HISTÓRICO
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Adiciona uma mensagem inicial de boas-vindas
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu assistente jurídico especializado em Leis de Santa Catarina. Como posso ajudar na sua consulta legal hoje?"})


# 2. EXIBE O HISTÓRICO DE MENSAGENS NO CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Garante que a mensagem é sempre Markdown
        st.markdown(message["content"])

# --- Lógica de Consulta e Resposta (COM STREAMING) ---
if prompt := st.chat_input("Faça sua pergunta sobre as Leis de SC:"):
    
    # 3. Adiciona a pergunta do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara o histórico para o filtro de intenção e reescrita de query
    # É CRÍTICO que o histórico para o chatbot seja uma lista de tuplas (role, content)
    history_for_context = [
        (msg["role"], msg["content"]) 
        for msg in st.session_state.messages
    ]

    # 4. Chama a função de resposta
    with st.chat_message("assistant"):
        
        # Cria um contêiner para a resposta e o footer (fontes/tempo)
        response_container = st.container()
        
        # Novo Placeholder: Para o texto principal da resposta (Onde o stream vai aparecer)
        text_placeholder = response_container.empty()
        
        # Placeholder para o footer (Fontes/Tempo)
        response_footer_placeholder = response_container.empty()
        
        start_time = time.time()
        
        try:
            # AQUI ESTÁ A CHAMADA CORRIGIDA: A função get_response espera 2 argumentos.
            response_result, cited_sources = chatbot.get_response(
                prompt, 
                history_for_context
            ) 
        except Exception as e:
            response_result = f"Ocorreu um erro inesperado no back-end: {e}"
            cited_sources = set()


        # >>> LÓGICA DE CONSUMO DO STREAM/STRING <<<
        
        full_response = ""
        
        # Se o resultado for uma STRING (erro ou filtro NAO_JURIDICA), exibe de uma vez.
        if isinstance(response_result, str):
            full_response = response_result
            text_placeholder.markdown(full_response) # <-- Usa o placeholder de texto
        
        # Se o resultado for um ITERATOR/STREAM (resposta bem-sucedida do Gemini)
        else:
            # st.write_stream usa a função geradora para exibir o texto
            with text_placeholder.container():
                full_response = st.write_stream(text_generator(response_result))


        # ----------------------------------------------------------------------
        
        end_time = time.time()
        
        # 5. Salva a resposta COMPLETA no histórico (IMPORTANTE)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 6. Exibe o Footer (Fontes e Tempo)
        footer_content = f"Tempo de resposta: **{end_time - start_time:.2f} segundos**"
        
        if cited_sources:
            sources_list = "\n".join([f"- {source}" for source in sorted(list(cited_sources))])
            footer_content += f"\n\n**Fontes Recuperadas:**\n{sources_list}"
        else:
            if "não foi encontrada nos documentos" in full_response or "não-jurídica" in full_response:
                footer_content += "\n\n**Fontes Recuperadas:** Nenhuma fonte no corpus foi utilizada."
            else:
                footer_content += "\n\n**Fontes Recuperadas:** Nenhuma fonte foi citada (possível erro)." 

        # PREENCHE o placeholder do footer. 
        response_footer_placeholder.markdown(f"--- \n{footer_content}")

# --- Footer Estático ---
st.sidebar.markdown("---")
# O botão para limpar a conversa
st.sidebar.button('🗑️ Limpar Conversa (Resetar Memória)', on_click=clear_chat_history)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Status da Base de Dados:** {status}")
st.sidebar.markdown("Desenvolvido para análise jurídica de Leis do Estado de Santa Catarina.")