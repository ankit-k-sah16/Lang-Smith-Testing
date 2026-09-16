from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(
    model="nomic-embed-text"
)

result = emb.embed_query("Hello world")

print("Embedding generated!")
print("Dimensions:", len(result))
print(result[:5])