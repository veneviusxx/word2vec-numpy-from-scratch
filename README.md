# word2vec-numpy-from-scratch
Implementation of the Word2Vec algorithm (Skip-gram with Negative Sampling) from scratch using pure NumPy.

This repository contains a from-scratch implementation of the **Word2Vec** optimization procedure, strictly adhering to the "pure NumPy" constraint. 

### Key Features:
* **Algorithm:** Skip-gram with Negative Sampling (SGNS).
* **Architecture:** Object-Oriented approach (`TextProcessor` and `Word2VecSGNS` classes).
* **Implementation:** Manual derivation and implementation of forward pass, binary cross-entropy loss, gradients, and parameter updates (SGD) without the use of external Deep Learning frameworks.
* **Evaluation:** Mathematical validation of the trained vector space using custom-built Cosine Similarity.
* **Corpus:** Trained on a subset of the *Tiny Shakespeare* dataset.
