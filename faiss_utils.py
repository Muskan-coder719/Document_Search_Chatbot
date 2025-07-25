import faiss

def create_faiss_index(embedding_matrix):
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index
