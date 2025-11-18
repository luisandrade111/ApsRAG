1- Este projeto implementa um agente de perguntas e respostas com RAG (Retrieval-Augmented Generation) usando:

- LangChain como framework de orquestração;

- Mistral AI como provedor de LLM;

- Embeddings do Mistral para representar o conteúdo em vetores;

- Um arquivo PDF local como base de conhecimento;

- Um vector store em memória para busca semântica.

- O tema escolhido é “História das Olimpíadas”.
- O agente responde perguntas em língua portuguesa sobre esse tema consultando o conteúdo do PDF.

2- Pré-requisitos

- Python 3.10+ instalado

- Utilizar a chave api passada no txt anexado junto com a atividade

3- Instalação das Dependências

- Ativar ambiente virtual:
python -m venv .venv
.venv\Scripts\activate

-instalar bibliotecas:
pip install -r requirements.txt

4- Com o ambiente virtual ativado e dependências instaladas, execute:
python main.py
