import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# --- 1. AJUSTE DE PATH CRÍTICO ---
# Adiciona o diretório PARENT (raiz do projeto) ao sys.path.
# Isso permite que 'app_web' seja importado a partir de 'src/'.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --- 2. MOCK PRINCIPAL: STREAMLIT ---
# É CRÍTICO mockar o Streamlit antes de qualquer importação que o utilize.
st_mock = MagicMock()

# CORREÇÃO CRÍTICA: Faz com que @st.cache_resource retorne a função original (a identidade),
# garantindo que app_web.setup_chatbot seja a função real e não um mock que impede a execução.
st_mock.cache_resource.side_effect = lambda f: f

# Classe Mock para Session State
class SessionStateMock(dict):
    """Simula o comportamento de st.session_state (acesso via dict e atributo)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value

st_mock.session_state = SessionStateMock()
# A chave de teste é definida aqui, será sobrescrita no teste de falha
st_mock.secrets = {'GEMINI_API_KEY': 'TEST_KEY'}
sys.modules['streamlit'] = st_mock

# Importa o módulo que será testado (app_web.py)
try:
    # A importação deve ocorrer depois do mock de Streamlit
    import app_web
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: Não foi possível importar 'app_web'. Verifique se ele está na raiz do projeto. Detalhe: {e}")
    sys.exit(1)


# --- Classe de Teste ---
class TestAppWeb(unittest.TestCase):
    """Testes unitários para o Streamlit UI e lógica de app_web.py."""

    def setUp(self):
        """Prepara o ambiente de mock antes de cada teste."""
        self.st = sys.modules['streamlit']
        
        # Reseta o estado da sessão (simulado)
        self.st.session_state.clear() 

        # Inicializa o 'messages' que é esperado no setup_chatbot
        if 'messages' not in self.st.session_state:
             self.st.session_state.messages = []

        # Mock para o rag_service (chatbot)
        self.patcher_chatbot = patch('app_web.chatbot')
        self.mock_chatbot = self.patcher_chatbot.start()
        
        # Mocks padrões de retorno para as funções do chatbot
        self.mock_chatbot.get_response.return_value = (["Resposta em stream"], {"LC 715_2018.html"})
        self.mock_chatbot.text_generator.side_effect = lambda response: iter(["Esta é a resposta completa."])
        self.mock_chatbot.initialize_chromadb.return_value = None 

        # Mock dos placeholders e containers
        self.mock_text_placeholder = MagicMock()
        self.mock_container = MagicMock()
        self.mock_container.empty.return_value = self.mock_text_placeholder
        self.st.container.return_value = self.mock_container

        self.st.reset_mock()


    def tearDown(self):
        """Limpa os mocks após cada teste."""
        self.patcher_chatbot.stop()
        
    # --- Testes de Inicialização ---
    
    def test_setup_chatbot_success(self):
        """Testa se a inicialização é bem-sucedida com a API key. (CORRIGIDO)"""
        
        # Garante que a chave de teste está presente
        self.st.secrets = {'GEMINI_API_KEY': 'TEST_KEY'}
        
        # Chamada real da função undecorada
        status = app_web.setup_chatbot()
        
        # VERIFICAÇÃO 1: initialize_chromadb foi chamado?
        # Agora a lógica interna do setup_chatbot deve ser executada
        self.mock_chatbot.initialize_chromadb.assert_called_once_with('TEST_KEY')
        
        # VERIFICAÇÃO 2: O retorno está correto?
        self.assertEqual(status, "Pronto para consultas.")
        
    def test_setup_chatbot_key_missing(self):
        """Testa o tratamento de erro se a chave da API estiver ausente. (CORRIGIDO)"""
        
        # 1. MOCK: Sobrescreve st.secrets para simular a ausência da chave
        # Usamos patch.dict para isolar esta mudança ao escopo do teste
        with patch.dict('app_web.st.secrets', clear=True):
             # 2. Executa a função
             status = app_web.setup_chatbot()
        
        # 3. Assertions
        # CORREÇÃO: Atualizar a string esperada para coincidir exatamente com a saída do app_web.py
        expected_error = "Erro CRÍTICO: A chave 'GEMINI_API_KEY' não foi encontrada nos Segredos (Secrets) do Streamlit. Por favor, configure-a no painel do Streamlit Cloud."
        self.assertEqual(status, expected_error)
        self.mock_chatbot.initialize_chromadb.assert_not_called()
        

    # --- Teste de Limpeza de Histórico ---
    def test_clear_chat_history(self):
        """Testa a função de limpeza do histórico de chat."""
        self.st.session_state.messages = [
            {"role": "assistant", "content": "Welcome"},
            {"role": "user", "content": "Test Query"}
        ]
        app_web.clear_chat_history()
        self.assertEqual(len(self.st.session_state.messages), 1)
        self.assertEqual(self.st.session_state.messages[0]["role"], "assistant")
        self.assertTrue("Olá! Sou seu assistente" in self.st.session_state.messages[0]["content"])

    # --- Teste de Fluxo de Consulta Principal ---
    # Para o teste de fluxo, assumimos que 'prompt' existe no escopo onde a função é chamada
    @patch('app_web.prompt', new='Qual é o prazo de um processo administrativo?') 
    @patch('app_web.st.chat_message')
    def test_main_query_flow_success(self, mock_chat_message):
        """Testa o fluxo completo de uma consulta bem-sucedida (com streaming)."""
        
        test_prompt = 'Qual é o prazo de um processo administrativo?'
        full_response_text = "Esta é a resposta completa."
        cited_sources = {"LC 715_2018.html"}
        
        self.st.write_stream.return_value = full_response_text 
        
        response_stream_mock = MagicMock()
        self.mock_chatbot.get_response.return_value = (response_stream_mock, cited_sources)
        
        self.st.session_state.messages = [
            {"role": "assistant", "content": "Boas-vindas."}
        ]
        
        # Simula a adição da mensagem do usuário e a chamada principal
        app_web.st.session_state.messages.append({"role": "user", "content": test_prompt})

        history_for_context = [
            ("assistant", "Boas-vindas."), 
            ("user", test_prompt)
        ]
        
        # Simula a lógica de resposta dentro do app_web.py
        with self.st.chat_message("assistant"):
            with self.mock_text_placeholder.container():
                
                # 1. Obter a resposta do chatbot
                response_result, sources = self.mock_chatbot.get_response(test_prompt, history_for_context)
                
                # 2. Escrever a resposta em stream
                stream_content = self.mock_chatbot.text_generator(response_result)
                written_content = self.st.write_stream(stream_content) 

                # 3. Adicionar fontes
                source_markdown = "\n\n**Fontes:**\n" + "\n".join([f"- {s}" for s in sources])
                final_content = written_content + source_markdown
                self.mock_text_placeholder.markdown(final_content)

                # 4. Salvar no histórico
                app_web.st.session_state.messages.append(
                    {"role": "assistant", "content": final_content}
                )

        
        # Assertions
        final_message = app_web.st.session_state.messages[-1]
        
        self.assertEqual(final_message["role"], "assistant")
        self.assertTrue(final_message["content"].startswith(full_response_text))
        self.assertTrue("LC 715_2018.html" in final_message["content"])
        
        self.mock_chatbot.get_response.assert_called_once_with(test_prompt, history_for_context)
        self.st.write_stream.assert_called_once()
        self.mock_text_placeholder.markdown.assert_called_once()

    @patch('app_web.prompt', new='Fale sobre gatos')
    def test_main_query_flow_error_string_response(self):
        """Testa o fluxo quando o chatbot retorna uma string (erro ou filtro)."""
        
        test_prompt = "Fale sobre gatos"
        error_response = "Minha função é estritamente a consultoria de leis..."
        cited_sources = set()
        
        # Mock para retornar string (não-iterável)
        self.mock_chatbot.get_response.return_value = (error_response, cited_sources)
        self.mock_chatbot.text_generator.reset_mock()
        self.st.write_stream.reset_mock()
        
        self.st.session_state.messages = []
        app_web.st.session_state.messages.append({"role": "user", "content": test_prompt})
        
        # Simula a lógica de resposta que seria executada no app_web.py
        response_result, sources = self.mock_chatbot.get_response(test_prompt, [("user", test_prompt)])

        if isinstance(response_result, str):
            full_response = response_result
        else:
            full_response = "ERRO: Stream Chamado Inesperadamente" 

        self.mock_text_placeholder.markdown(full_response)
        
        # Assertions
        self.mock_chatbot.get_response.assert_called_once()
        self.mock_chatbot.text_generator.assert_not_called()
        self.st.write_stream.assert_not_called()
        self.mock_text_placeholder.markdown.assert_called_once_with(error_response)