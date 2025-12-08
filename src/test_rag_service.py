import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# --- Configuração de Path para Importação ---
try:
    # Importa as funções globais do seu arquivo real
    from src.rag_service import (
        get_response, 
        classify_intent, 
        get_context, 
        text_generator, 
        SYSTEM_INSTRUCTION
    )
    # Tenta importar o erro real
    try:
        from google.genai.errors import APIError as RAGError
    except ImportError:
        # Fallback para o caso de a biblioteca não estar instalada ou APIError ter sido movido
        # Simula a assinatura real do APIError: RAGError(code, response_json=None, ...)
        class RAGError(Exception): 
            def __init__(self, code, *args, response_json=None, **kwargs):
                self.response_json = response_json or MagicMock()
                super().__init__(f"API Error {code}", *args, **kwargs)
    
    print("MÓDULO REAL: src.rag_service importado com sucesso.")

except ImportError as e:
    print(f"ERRO CRÍTICO: Não foi possível importar as funções do 'src/rag_service.py'. Verifique o path e a sintaxe. Detalhe: {e}")
    # Cria mocks básicos para permitir que os testes de estrutura rodem
    def get_response(*args): return "Mock Response", set()
    def classify_intent(*args): return "JURIDICA"
    def get_context(*args): return "Contexto Mock", {"source1"}
    def text_generator(*args): return []
    class RAGError(Exception): 
        def __init__(self, code, *args, response_json=None, **kwargs):
            super().__init__(f"API Error {code}", *args, **kwargs)
            self.response_json = response_json or MagicMock()
    SYSTEM_INSTRUCTION = "Instrução Mock"
# --- Fim da Lógica de Importação ---


class TestRAGServiceFunctions(unittest.TestCase):
    """Testes unitários para as funções globais do serviço RAG."""

    def setUp(self):
        """Dados de teste comuns."""
        self.user_query = "Qual a multa prevista na LC 715 de 2018?"
        self.history = [
            ("user", "Qual o objeto da LC 715?"),
            ("assistant", "O objeto é... (LC 715)."),
            ("user", self.user_query)
        ]
        self.mock_stream = [
            MagicMock(text="Esta é a "),
            MagicMock(text="resposta final ")
        ]
        
    @patch('src.rag_service.classify_intent')
    def test_get_response_nao_juridica(self, mock_classify_intent):
        """Testa o filtro de intenção não-jurídica (fora de escopo)."""
        
        mock_classify_intent.return_value = 'NAO_JURIDICA'
        
        response, sources = get_response(self.user_query, self.history)
        
        mock_classify_intent.assert_called_once_with(self.user_query, self.history)
        self.assertTrue("Minha função é estritamente a consultoria de leis" in response)
        self.assertEqual(sources, set())

    @patch('src.rag_service.get_context')
    @patch('src.rag_service.classify_intent', return_value='JURIDICA')
    @patch('src.rag_service.client') # Mock da variável global client
    def test_get_response_success(self, mock_client, mock_classify_intent, mock_get_context):
        """Testa o fluxo completo de sucesso do RAG."""
        
        # 1. Configuração do Mock de Geração de Conteúdo (Simula o Stream)
        mock_client.models.generate_content_stream.return_value = self.mock_stream
        
        # 2. Configuração do Contexto
        mock_context = "Art. 10º. A multa será de 10 URVs. (LC 715_2018)"
        mock_sources = {"LC 715_2018.html"}
        mock_get_context.return_value = (mock_context, mock_sources)
        
        # Execução
        response_stream, sources = get_response(self.user_query, self.history)
        
        # Asserts
        mock_get_context.assert_called_once_with(self.user_query, self.history, k=10)
        self.assertEqual(sources, mock_sources)
        self.assertEqual(response_stream, self.mock_stream) # Deve retornar o objeto stream

        # Verifica a chamada ao Gemini
        mock_client.models.generate_content_stream.assert_called_once()
        
        # Verifica se o prompt contém o contexto injetado (usando kwargs)
        args, kwargs = mock_client.models.generate_content_stream.call_args
        
        # Acessando 'contents' via kwargs
        contents_from_call = kwargs['contents']
        rag_prompt = contents_from_call[0]['parts'][0]['text']
        
        self.assertTrue(mock_context in rag_prompt)
        self.assertEqual(kwargs['config']['system_instruction'], SYSTEM_INSTRUCTION)
        self.assertEqual(kwargs['config']['temperature'], 0.3)


    @patch('src.rag_service.get_context')
    @patch('src.rag_service.classify_intent', return_value='JURIDICA')
    def test_get_response_no_context_found(self, mock_classify_intent, mock_get_context):
        """Testa o cenário onde o get_context falha ou não encontra nada."""
        
        mock_get_context.return_value = ("", set()) # Contexto vazio
        
        response, sources = get_response(self.user_query, self.history)
        
        self.assertTrue("ocorreu um erro ao buscar os documentos" in response)
        self.assertEqual(sources, set())
        mock_get_context.assert_called_once()

    @patch('src.rag_service.get_context')
    @patch('src.rag_service.classify_intent', return_value='JURIDICA')
    @patch('src.rag_service.client')
    def test_get_response_api_failure(self, mock_client, mock_classify_intent, mock_get_context):
        """Testa o tratamento de exceção da API Gemini. (CORRIGIDO PARA ARGUMENTO POSICIONAL E NOMEADO)"""
        
        # 1. Configuração do Contexto Válido
        mock_context = "Contexto Válido."
        mock_sources = {"source1"}
        mock_get_context.return_value = (mock_context, mock_sources)
        
        # 2. Configuração do Mock de Erro (response_json)
        mock_error_response = MagicMock()
        mock_error_response.json.return_value = {"error": {"message": "Erro de API simulado."}}
        mock_error_response.status_code = 401
        
        # 3. MOCK CRÍTICO: Passa o 'code' (posicional) e o 'response_json' (keyword)
        error_code = 401
        mock_client.models.generate_content_stream.side_effect = RAGError(
            error_code,                          # Argumento posicional 'code'
            response_json=mock_error_response    # Argumento nomeado 'response_json'
        )
        
        # Execução
        response, sources = get_response(self.user_query, self.history)
        
        # Assertions
        self.assertTrue("Ocorreu um erro no processamento do Chat Gemini" in response)
        self.assertEqual(sources, mock_sources)
        mock_client.models.generate_content_stream.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)