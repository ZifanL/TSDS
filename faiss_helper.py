import faiss
import numpy as np

class FaissIndexIVFFlat:
    def __init__(self, data, nprobe=10):
        self.build(data, nprobe)

    def build(self, data, nprobe):
        nlist = int(np.sqrt(data.shape[0])) // 2
        quantizer = faiss.IndexFlatL2(data.shape[-1])
        self.index = faiss.IndexIVFFlat(quantizer, data.shape[-1], nlist)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = nprobe

    def search(self, query, K):
        return self.index.search(query, K)