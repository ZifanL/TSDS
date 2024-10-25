# TSDS
This repository contains the implementation of the framework described in [TSDS: Data Selection for Task-Specific Model Finetuning](https://arxiv.org/abs/2410.11303).

## Prerequisites

Before running the project, ensure you have Python installed. You can download the latest version of Python from [here](https://www.python.org/downloads/).

## Installation

1. Clone the repository:
    ```bash
    git https://github.com/ZifanL/TSDS.git
    cd TSDS
    ```

2. Install the required dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) If you're using `faiss-gpu`, ensure you have the correct GPU drivers installed. Refer to the [Faiss documentation](https://github.com/facebookresearch/faiss) for more information.

## Usage

After installing the dependencies, you can run the project as follows using the toy data:

```bash
python tsds.py
```
In the output folder, the output file `selected_candidate_indices.npy` will contain the indices of the selected candidates.

To run TSDS on your customized data, two embedding files are needed:
- An `.npy` file that stores the embeddings of the candidate examples. The shape of the array should be (number of candidates, embedding dimensions)
- An `.npy` file that stores the embeddings of the query examples. The shape of the array should be (number of query examples, embedding dimensions)
Change the file paths in `config.yaml`. Adjust the parameters in `config.yaml` as needed.
The implementation uses `faiss.IndexIVFFlat` for approximate nearest neighbor search. To use a customized index, add it to `faiss_helper.py` and substitute `FaissIndexIVFFlat` in `tsds.py`.

## Citation
Please cite our paper if you find this repo helpful in your work:
```
@misc{liu2024tsdsdataselectiontaskspecific,
      title={TSDS: Data Selection for Task-Specific Model Finetuning}, 
      author={Zifan Liu and Amin Karbasi and Theodoros Rekatsinas},
      year={2024},
      eprint={2410.11303},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.11303}, 
}
```
