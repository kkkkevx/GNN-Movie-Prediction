# 🎬 Movie Success & Revenue Prediction with Heterogeneous Graph Neural Networks

This project predicts **movie success** (binary classification) and **log-transformed box office revenue** (regression) using a **heterogeneous Graph Neural Network (GNN)**.  
The model integrates relational information among **movies, actors, directors, and genres** using PyTorch Geometric (PyG).

We use the **TMDB 5000 Movies Dataset** (Kaggle) and construct a full heterogeneous graph with millions of edges.

---

## 🚀 Features

- Heterogeneous GNN using `HeteroConv + GraphSAGE`
- Multi-task learning:  
  - **Success prediction** (`success` = ROI > 1 or rating-based fallback)  
  - **Revenue regression** (log-scale revenue)
- Graph construction with:
  - Movie nodes with numeric features  
  - Actor nodes
  - Director nodes
  - Genre nodes
- Bidirectional edges:
  - `movie ↔ actor`
  - `movie ↔ director`
  - `movie ↔ genre`
- Full preprocessing: log transforms, StandardScaler, handling missing values

---

## 📦 Dataset

We use **TMDB 5000 Movies** from Kaggle:

- 4,803 movies  
- ~9,390 unique actors  
- ~2,350 unique directors  
- ~20 genres  
- ~12 million total edges  

Movie-level numeric features:

- popularity  
- runtime  
- vote_avg  
- vote_count  
- budget_log  
- ROI  

Success label:
success = 1 if ROI > 1 else 0
(Or rating-based fallback for missing revenue)


---

## 🧠 Model Architecture

### 🔹 Heterogeneous GNN

```python
HeteroConv({
    ('movie', 'acted_in', 'actor'): SAGEConv,
    ('actor', 'acted_by', 'movie'): SAGEConv,
    ('movie', 'directed_by', 'director'): SAGEConv,
    ('director', 'directs', 'movie'): SAGEConv,
    ('movie', 'has_genre', 'genre'): SAGEConv,
    ('genre', 'in_genre', 'movie'): SAGEConv,
})
