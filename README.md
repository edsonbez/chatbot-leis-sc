⚖️ Chatbot de Leis Municipais (RAG - Retrieval Augmented Generation)

Este projeto implementa um chatbot baseado em Streamlit e Google Gemini que atua como um assistente jurídico especializado em leis municipais (Lei Orgânica e Leis Complementares). Ele utiliza a arquitetura RAG (Retrieval Augmented Generation) para buscar informações em documentos locais e gerar respostas precisas e citadas.

🚀 Funcionalidades

Especialização em Leis: Treinado exclusivamente em documentos jurídicos específicos do município.

Geração Aumentada (RAG): Utiliza um Vector Store (ChromaDB ou FAISS, conforme configurado em src/rag_service.py) para buscar trechos relevantes das leis e usá-los como contexto para o modelo Gemini.

Streaming: A resposta do LLM é transmitida em tempo real para o usuário.

Citação de Fontes: Indica o nome do documento (ex: LC 715_2018.html) que fundamentou a resposta.

Tratamento de Chaves: Implementação robusta de inicialização com caching do Streamlit e tratamento de erro crítico para a chave GEMINI_API_KEY.

⚙️ Estrutura do Projeto

A estrutura do projeto está organizada da seguinte forma, incluindo os arquivos de teste:

.
├── app_web.py              # Aplicação Streamlit principal (UI)
├── .env                    # Arquivo para variáveis de ambiente (local)
├── requirements.txt        # Dependências do Python
├── README.md               # Este arquivo
├── documentos_map.json     # Mapeamento dos documentos processados
├── faiss_index.bin         # Índice vetorial (FAISS) ou diretório do ChromaDB
└── src/
    ├── rag_service.py      # Lógica de LLM, VectorDB e RAG
    ├── data_prep.py        # Scripts de processamento e preparação de dados
    ├── test_app_web.py     # Testes unitários para a lógica da aplicação web
    ├── test_data_prep.py   # Testes unitários para o módulo de preparação de dados
    └── test_rag_service.py # Testes unitários para o módulo RAG Service


🛠️ Como Executar Localmente

Pré-requisitos

Python 3.9+

Uma chave da Google AI Studio (Gemini API Key).

Instalação

Clone o Repositório:

git clone [https://docs.github.com/pt/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github](https://docs.github.com/pt/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
cd chatbot_leis


Crie e Ative o Ambiente Virtual:

python -m venv venv_chatbot
# No Windows:
.\venv_chatbot\Scripts\activate
# No macOS/Linux:
source venv_chatbot/bin/activate


Instale as Dependências:

pip install -r requirements.txt


(Nota: Assuma que requirements.txt contém streamlit, google-genai, chromadb/faiss e python-dotenv.)

Configuração da Chave API

Crie um arquivo chamado .env na raiz do projeto.

Adicione sua chave Gemini API Key:

GEMINI_API_KEY="SUA_CHAVE_GEMINI_AQUI"


Execução

Execute a aplicação Streamlit:

streamlit run app_web.py


O aplicativo será aberto automaticamente no seu navegador.

✅ Execução dos Testes

Para garantir que tudo está funcionando conforme o esperado, execute os testes unitários. Você pode rodar todos os testes de uma vez usando:

python -m unittest discover src


Ou rodar testes específicos:

python -m unittest src.test_app_web
python -m unittest src.test_data_prep
python -m unittest src.test_rag_service


Se os testes passarem, a saída será OK.