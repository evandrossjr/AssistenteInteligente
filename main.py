from graph import construir_grafo

NOMES = {"python": "Agente Python", "geral": "Agente de Conhecimentos Gerais"}


def main() -> None:
    app = construir_grafo()

    print("=" * 60)
    print(" Assistente Inteligente - LangGraph + Ollama")
    print(" Digite sua pergunta (ou 'sair' para encerrar)")
    print("=" * 60)

    while True:
        pergunta = input("\nVocê: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até logo!")
            break

        try:
            resultado = app.invoke(
                {"pergunta": pergunta, "categoria": "", "resposta": ""}
            )
        except Exception as erro:  # Ollama desligado, modelo ausente etc.
            print(f"\n[Erro] Não foi possível obter resposta: {erro}")
            print("Verifique se o Ollama está rodando e se o modelo foi baixado.")
            continue

        print(f"\n[Classificação: {resultado['categoria']}] "
              f"-> {NOMES[resultado['categoria']]}")
        print(f"\nAssistente: {resultado['resposta']}")


if __name__ == "__main__":
    main()