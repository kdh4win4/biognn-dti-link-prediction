BioGNN DTI Link Prediction
~~~~

Graph Neural Network (GNN) based drug–target interaction (DTI) link prediction
for hypothesis ranking in drug discovery pipelines.

LLM → Graph → GNN → Ranked hypotheses

~~~~
Purpose
~~~~

This project integrates biological relations extracted from LLMs
(e.g. PubMedBERT, BioLLM) into a unified knowledge graph, and applies
Graph Neural Networks to predict and rank plausible drug–target interactions.

~~~~
Overall Workflow
~~~~

1) LLM-based relation extraction (external)
2) Knowledge graph construction
3) GNN training (link prediction)
4) Candidate scoring and ranking

~~~~
Repository Structure
~~~~

biognn-dti-link-prediction/
- README.md
- requirements.txt
- .gitignore
- src/
  - biognn/
    - __init__.py
    - data.py
    - model.py
    - train.py
    - infer.py
- scripts/
  - 01_build_graph.py
  - 02_train.sh
  - 03_infer.sh

~~~~
Local Installation
~~~~

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

~~~~
Training
~~~~

bash scripts/02_train.sh

~~~~
Inference
~~~~

python -m src.biognn.infer --drug DRUG_ID --topk 20

~~~~
Input Data
~~~~

- Drug–protein interaction edges
- Protein–protein interaction edges (optional)
- LLM-extracted biomedical relations (optional)

~~~~
Output
~~~~

Ranked list of candidate protein targets with confidence scores.

~~~~
Design Notes
~~~~

- GNN architectures: GraphSAGE / GAT
- Task: link prediction
- Loss: binary cross-entropy with negative sampling
- Scalable to large biomedical graphs
- Designed for integration with LLM-based discovery pipelines

~~~~
Author
~~~~

Dohoon Kim  
https://github.com/kdh4win4
