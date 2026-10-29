import os

MODELO = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
 
OLLAMA_URL = "http://localhost:11434"

TEMPERATURA = 0.0

TEMPERATURA_RESPOSTA = 0.4