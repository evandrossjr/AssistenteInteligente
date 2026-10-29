 
from langgraph.graph import END, START, StateGraph
 
from agents import agente_geral, agente_python, classificar, rotear
from state import AssistentState
 
 
def construir_grafo():
    grafo = StateGraph(AssistentState)
 
    # Nós
    grafo.add_node("classificador", classificar)
    grafo.add_node("agente_python", agente_python)
    grafo.add_node("agente_geral", agente_geral)
 
    # Arestas
    grafo.add_edge(START, "classificador")
    grafo.add_conditional_edges(
        "classificador",
        rotear,
        {"python": "agente_python", "geral": "agente_geral"},
    )
    grafo.add_edge("agente_python", END)
    grafo.add_edge("agente_geral", END)
 
    return grafo.compile()