import os
import logging
import yaml

import numpy as np
import heapq

from faiss_helper import FaissIndexIVFFlat

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    SAMPLE_SIZE = config["sample_size"]
    MAX_K = config["max_K"]
    KDE_K = config["kde_K"]
    SIGMA = config["sigma"]
    ALPHA = config["alpha"]
    C = config["C"]

    xq = np.load(config["query_embedding_path"]).astype(np.float32)
    assert xq.ndim == 2, f"Embeddings of the query examples should be in the form of a 2D array."
    logging.info(f"number of query examples: {xq.shape[0]}, embedding dimension: {xq.shape[1]}")

    xb = np.load(config["candidate_embedding_path"]).astype(np.float32)
    assert xb.ndim == 2, f"Embeddings of the candidates should be in the form of a 2D array."
    logging.info(f"number of candidates: {xb.shape[0]}, embedding dimension: {xb.shape[1]}")
    MAX_K = min(MAX_K, xb.shape[0] // 10)
    KDE_K = min(KDE_K, xb.shape[0] // 10)

    logging.info(f"Starting building index for the candidate examples.")
    index = FaissIndexIVFFlat(xb)

    logging.info(f"Start prefetching {MAX_K}-nearest neighbors for each query example.")
    top_dists, top_indices = index.search(xq, MAX_K)
    top_indices = top_indices.astype(int)
    sorted_indices = np.argsort(top_dists, axis=-1)
    static_indices = np.indices(top_dists.shape)[0]
    top_dists = np.sqrt(top_dists[static_indices, sorted_indices])
    # top_indices[i][j] is the index of the jth nearest neighbor
    # (among the candidates) of the ith query example
    top_indices = top_indices[static_indices, sorted_indices]

    # top_kde[i][j] is the KDE of the jth nearest neighbor of the ith query example
    if SIGMA == 0:
        logging.info("Sigma is zero, KDE (kernel density estimation) set to 1 for all the points.")
        top_kdes = np.ones_like(top_indices)
    else:
        logging.info(f"Start computing KDE (kernel density estimation), neighborhood size: {KDE_K}.")
        top_indices_set = list(set([i for i in top_indices.reshape(-1)]))
        top_features = xb[top_indices_set]
        index_for_kde = FaissIndexIVFFlat(top_features)
        D2, I = index_for_kde.search(top_features, KDE_K)
        kernel = 1 - D2 / (SIGMA ** 2)
        logging.info(f'A point has {(kernel > 0).sum(axis=-1).mean() - 1} near-duplicates on average.')
        kernel = kernel * (kernel > 0)
        kde = kernel.sum(axis=-1)
        kde_map = {top_indices_set[i]:kde[i] for i in range(len(top_indices_set))}
        kde_mapfunc = np.vectorize(lambda t: kde_map[t])
        top_kdes = kde_mapfunc(top_indices)
            
    logging.info("Start computing the probability assignment.")
    M, N = top_indices.shape[0], xb.shape[0]
    lastK = [0] * M
    heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(M)]
    heapq.heapify(heap)
    dist_weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(M)]
    s = 0
    cost = np.zeros(M)
    total_cost = 0
    while len(heap) > 0:
        count, curr_k, curr_j = heapq.heappop(heap)
        s = count
        # if we increase s by any positive amount, the 0, 1, ..., curr_k has to transport probability mass to curr_k + 1
        total_cost -= cost[curr_j]
        cost[curr_j] = top_dists[curr_j][curr_k + 1] * count - dist_weighted_sum[curr_j]
        total_cost += cost[curr_j]
        # If the condition breaks, the current s will be the final s
        if ALPHA / C * total_cost >= (1 - ALPHA) * M:
            break
        lastK[curr_j] = curr_k
        if curr_k < MAX_K - 2:
            count += 1.0 / top_kdes[curr_j][curr_k + 1]
            heapq.heappush(heap, (count, curr_k + 1, curr_j))
            dist_weighted_sum[curr_j] += top_dists[curr_j][curr_k + 1] / top_kdes[curr_j][curr_k + 1]
    global_probs = np.zeros(N)
    for j in range(M):
        prob_sum = 0
        for k in range(lastK[j] + 1):
            global_probs[top_indices[j][k]] += 1 / M / s / top_kdes[j][k]
            prob_sum += 1 / M / s / top_kdes[j][k]
        global_probs[top_indices[j][lastK[j] + 1]] += max(1.0 / M - prob_sum, 0)
        assert 1.0 / M - prob_sum >= -1e-9, f'{1.0 / M - prob_sum}'
        assert (1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s <= 1 + 1e-9 or lastK[j] == MAX_K - 2, f'{(1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s}'

    logging.info(f"Start sampling. Sample size: {SAMPLE_SIZE}.")
    sample_times = np.random.multinomial(SAMPLE_SIZE, global_probs)
    sample_indices = []
    for i in range(sample_times.shape[0]):
        sample_indices.extend([i] * sample_times[i])

    logging.info(f"Saving indices of the selected candidates.")
    os.makedirs(config["output_folder"], exist_ok=True)
    np.save(os.path.join(config["output_folder"], "selected_candidate_indices.npy"), sample_indices)

if __name__ == "__main__":
    main()
