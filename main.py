import os
from dotenv import load_dotenv

# Modelo de chat genérico do LangChain (Mistral via init_chat_model)
from langchain.chat_models import init_chat_model

# Embeddings do Mistral
from langchain_mistralai import MistralAIEmbeddings

# Vector store em memória
from langchain_core.vectorstores import InMemoryVectorStore

# Carregador de PDF
from langchain_community.document_loaders import PyPDFLoader

# Split de texto em chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ferramentas (tools) para o agente
from langchain.tools import tool

# Criação do agente
from langchain.agents import create_agent


DATA_PATH = "data/historia_olimpiadas.pdf"


def load_env_and_model():
    """Carrega variáveis de ambiente e inicializa o modelo de chat (LLM)."""
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY não fornecida no arquivo .env")

    # Esse init_chat_model usa o provedor Mistral configurado
    # Modelo sugerido pelo professor: "mistral-small-latest"
    model = init_chat_model("mistral-tiny")
    return model


def build_vector_store():
    """Carrega o PDF, divide em chunks e cria o vector store em memória."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo PDF não encontrado em: {DATA_PATH}")

    # 1) Carrega o PDF
    loader = PyPDFLoader(DATA_PATH)
    docs = loader.load()

    if not docs:
        raise ValueError("Nenhum conteúdo foi carregado a partir do PDF.")

    print(f"[INFO] Documentos carregados do PDF: {len(docs)}")

    # 2) Split em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # tamanho de cada chunk em caracteres
        chunk_overlap=200,    # sobreposição entre chunks
        add_start_index=True, # guarda posição no texto original
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"[INFO] Documento dividido em {len(all_splits)} sub-documentos.")

    # 3) Cria embeddings Mistral
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # 4) Cria vector store em memória e indexa os chunks
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"[INFO] Vetores armazenados. Exemplo de IDs: {document_ids[:3]}")

    return vector_store


def build_tools(vector_store):
    """Define a tool de recuperação de contexto a partir do vector store."""

    from langchain.tools import tool

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """
        Recupera informações sobre História das Olimpíadas a partir do PDF
        para ajudar a responder uma pergunta.
        """
        # Força a query a ficar no contexto do tema
        query_for_search = f"No contexto da história das Olimpíadas: {query}"

        retrieved_docs = vector_store.similarity_search(query_for_search, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )

        # DEBUG opcional: ver o que está vindo do PDF
        # print("=== CONTEXTO RECUPERADO ===")
        # print(serialized[:1000])

        return serialized, retrieved_docs

    return [retrieve_context]


def build_agent(model, tools):
    """Cria o agente RAG usando o modelo de chat e a tool de retrieve."""
    system_prompt = (
        "Você é um agente especialista em História das Olimpíadas. "
        "Você tem acesso a uma ferramenta que recupera contexto de um PDF sobre o tema. "
        "Use SEMPRE essa ferramenta quando precisar de informação factual. "
        "Responda SEMPRE em português do Brasil. "
        "Se não encontrar a resposta no contexto recuperado, seja honesto e diga que "
        "não tem informação suficiente no documento."
    )

    agent = create_agent(model, tools, system_prompt=system_prompt)
    return agent


def run_interactive_loop(agent):
    """Loop simples de perguntas e respostas no terminal."""
    print("\n=== Agente RAG - História das Olimpíadas ===")
    print("Digite sua pergunta sobre Olimpíadas.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        user_input = input("Você: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Agente: Até mais! 👋")
            break

        # Enviamos a mensagem no formato esperado pelo agente
        try:
            # Aqui vou usar o stream para ficar próximo do exemplo do professor.
            # Se der problema, pode trocar por uma chamada simples sem stream.
            print("Agente (pensando...)\n")
            response_text = ""

            # Streaming da resposta
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
            ):
                last_msg = event["messages"][-1]
                # O pretty_print é útil em notebook; aqui vamos só pegar o conteúdo
                if getattr(last_msg, "content", None):
                    response_text = last_msg.content

            print(f"Agente: {response_text}\n")

        except Exception as e:
            print(f"[ERRO] Ocorreu um problema ao chamar o agente: {e}\n")


def main():
    # 1) Modelo
    model = load_env_and_model()

    # 2) Vector store com o PDF
    vector_store = build_vector_store()

    # 3) Tools (retrieve_context)
    tools = build_tools(vector_store)

    # 4) Agente RAG
    agent = build_agent(model, tools)

    # 5) Loop interativo
    run_interactive_loop(agent)


if __name__ == "__main__":
    main()
