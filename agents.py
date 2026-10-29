import re
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
 
from config import MODELO, OLLAMA_URL, TEMPERATURA, TEMPERATURA_RESPOSTA
from state import AssistentState
 
llm_classificador = ChatOllama(
    model=MODELO, base_url=OLLAMA_URL, temperature=TEMPERATURA
)
llm_resposta = ChatOllama(
    model=MODELO, base_url=OLLAMA_URL, temperature=TEMPERATURA_RESPOSTA
)
 
PROMPT_CLASSIFICADOR = (
    "Você é um classificador de perguntas. Responda SOMENTE com uma palavra:\n"
    "- PYTHON: se a pergunta é sobre a linguagem Python, seu código, sintaxe, "
    "bibliotecas, erros ou boas práticas.\n"
    "- GERAL: para qualquer outro assunto.\n"
    "Não explique, não use pontuação. Apenas PYTHON ou GERAL."
)
 
PROMPT_PYTHON = (
    "Você é um professor especialista em Python, atuando como assistente de uma "
    "instituição de ensino de tecnologia. Responda em português do Brasil, de "
    "forma clara e didática, com exemplos de código curtos quando ajudar."
)
 
PROMPT_GERAL = (
    "Você é um assistente virtual de conhecimentos gerais de uma instituição de "
    "ensino de tecnologia. Responda em português do Brasil, de forma clara, "
    "objetiva e contextualizada. Se não souber, diga que não sabe."
)
 
 
def classificar(estado: AssistentState) -> dict:
    """Nó classificador: define se a pergunta é sobre Python ou geral."""
    mensagens = [
        SystemMessage(content=PROMPT_CLASSIFICADOR),
        HumanMessage(content=estado["pergunta"]),
    ]
    saida = llm_classificador.invoke(mensagens).content
    # Remove blocos <think>...</think> (modelos como qwen3 imprimem raciocínio)
    saida = re.sub(r"<think>.*?</think>", "", saida, flags=re.DOTALL)
    saida = saida.strip().lower()
    categoria = "python" if "python" in saida else "geral"
    return {"categoria": categoria}
 
 
def agente_python(estado: AssistentState) -> dict:
    """Agente especializado em Python."""
    mensagens = [
        SystemMessage(content=PROMPT_PYTHON),
        HumanMessage(content=estado["pergunta"]),
    ]
    return {"resposta": llm_resposta.invoke(mensagens).content}
 
 
def agente_geral(estado: AssistentState) -> dict:
    """Agente de conhecimentos gerais."""
    mensagens = [
        SystemMessage(content=PROMPT_GERAL),
        HumanMessage(content=estado["pergunta"]),
    ]
    return {"resposta": llm_resposta.invoke(mensagens).content}
 
 
def rotear(estado: AssistentState) -> str:
    """Função de roteamento usada em add_conditional_edges."""
    return estado["categoria"]
 