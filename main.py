from pypdf import PdfReader

from chunking import split_text
from embeddings import create_embeddings
from vector_store import create_vector_store, search_vector_store
from llm import generate_answer


# -------------------------
# 1. Load PDF
# -------------------------

pdf_path = "data/Tharun_Kumar_Resume.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text


# -------------------------
# 2. Chunk the document
# -------------------------

chunks = split_text(text)

print("Total characters:", len(text))
print("Number of chunks:", len(chunks))


# -------------------------
# 3. Create embeddings
# -------------------------

embeddings = create_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# -------------------------
# 4. Create FAISS index
# -------------------------

index = create_vector_store(embeddings)

print("Number of vectors in FAISS:", index.ntotal)


# -------------------------
# 5. Ask question - ONLY ONCE
# -------------------------

query = input("\nAsk a question about the PDF: ")


# -------------------------
# 6. Create query embedding
# -------------------------

query_embedding = create_embeddings([query])


# -------------------------
# 7. Search FAISS
# -------------------------

distances, indices = search_vector_store(
    index,
    query_embedding,
    top_k=3
)


# -------------------------
# 8. Build context from retrieved chunks
# -------------------------

retrieved_chunks = []

for index_number in indices[0]:
    retrieved_chunks.append(chunks[index_number])

context = "\n\n".join(retrieved_chunks)


# -------------------------
# 9. Generate final answer
# -------------------------

answer = generate_answer(
    query,
    context
)


# -------------------------
# 10. Display final answer
# -------------------------

print("\nFinal Answer:")
print(answer)