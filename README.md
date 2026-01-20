# 🧬 BioGNN: Drug–Target Interaction Link Prediction

**GNN-based target prioritization for AI-driven drug discovery**

---

## 🔬 Overview

This repository demonstrates an **end-to-end Graph Neural Network (GNN) pipeline** for **drug–target interaction (DTI) link prediction**, built as a portfolio project for **AI-driven target discovery and translational research**.

It is intentionally minimal, reproducible, and runnable from the terminal, while reflecting **real industry patterns** used in computational biology and drug discovery.

This project complements an upstream LLM-based hypothesis generator:

LLM → biological relation extraction → candidate targets
↓
GNN → link prediction → prioritization

---

## 🚀 What This Project Demonstrates

### ✅ Technical Skills

* PyTorch Geometric (GCN-based GNN)
* Graph construction for biological networks
* End-to-end training & inference pipeline
* Model checkpointing and reuse
* CLI-driven reproducibility

### ✅ Drug Discovery Relevance

* Drug–target interaction modeling
* Latent biological representation learning
* Target prioritization via learned embeddings
* Ready to extend to ChEMBL / DrugBank / CRISPR / multi-omics graphs

---

## 🧠 Model Architecture

Node features (x)
↓
GCNConv (in_dim → hidden_dim)
↓
ReLU
↓
GCNConv (hidden_dim → hidden_dim)
↓
Node embeddings (latent biological space)
↓
Link score (dot product / classifier)

* **Nodes**: drugs, proteins (demo-scale)
* **Edges**: known or hypothesized interactions
* **Output**: link score representing interaction likelihood

---

## 📂 Project Structure

```
biognn-dti-link-prediction/
├── src/biognn/
│   ├── data.py        # graph construction
│   ├── model.py       # DTI_GNN (GCN)
│   ├── train.py       # training + checkpoint save
│   └── infer.py       # inference + link scoring
├── scripts/
│   ├── 01_build_graph.py
│   ├── 02_train.sh
│   └── 03_infer.sh
├── outputs/
│   └── gnn_model.pt   # trained model checkpoint
└── README.md
```

---

## ⚙️ How to Run (Reproducible Demo)

### 1) Build Graph

```bash
python scripts/01_build_graph.py
```

### 2) Train Model

```bash
bash scripts/02_train.sh
```

Example output:

```text
[INFO] Building demo graph...
[INFO] Graph: num_nodes=4, num_edges=3
[INFO] Starting training...
Epoch 001 | Loss: ...
...
[DONE] Model saved to outputs/gnn_model.pt
```

### 3) Run Inference

```bash
bash scripts/03_infer.sh
```

Example output:

```text
[INFO] Loading demo graph...
[INFO] Graph: num_nodes=4, num_edges=3
[INFO] Embeddings shape: (4, 32)
[RESULT] Example link score
  node_i=0  node_j=3  score=0.2740
```

---

## 🔍 Interpretation

* Node embeddings represent **learned biological states** in a latent space.
* Link score represents a **predicted interaction likelihood** between two nodes.
* In real applications, this supports:

  * target prioritization
  * drug repurposing hypotheses
  * mechanism exploration
  * experimental design guidance

---

## 🔄 Extension Ideas (Real-World Ready)

This pipeline is designed to scale to:

* ChEMBL / DrugBank networks
* PPI graphs and pathway graphs
* CRISPR perturbation graphs
* Multi-omics feature integration (transcriptomics, proteomics)
* LLM-generated hypothesis edges
* GNN + LLM hybrid reasoning for target discovery

---

## 👤 Author

**Dohoon Kim**
Senior Computational Biologist / Data Scientist
Focus: AI for drug discovery, target identification, and translational biology

---

## ⭐ Why This Matters

This repository demonstrates the ability to:

* translate biology into graphs
* apply GNNs to discovery problems
* build reproducible training/inference pipelines
* connect LLM-derived hypotheses to mechanistic graph modeling

These are core skills required for **AI Computational Biologist** roles.

