import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained LLM model and tokenizer
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example sentences
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast brown fox leaps over a lazy dog."

# Tokenize and encode the sentences
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

# Get sentence embeddings
with torch.no_grad():
    embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
    embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

# Compute cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Cosine Similarity: {cosine_similarity.item()}")