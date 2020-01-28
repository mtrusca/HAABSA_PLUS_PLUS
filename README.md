# HAABSA++
The code for A Hybrid Approach for Aspect-Based Sentiment Analysis Using Contextual Word Emmbeddings and Hierarchical Attention

The hybrid approach for aspect-based sentiment analysis (HAABSA) is a two-step method that classifies target sentiments using a domain sentiment ontology and a Multi-Hop LCR-Rot model as backup.
 - Original Paper: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf
 
 Keeping the ontology, we optimise the embedding layer of the backup neural network with context-dependent word embeddings and integrate hierarchical attention in the model's architecture (HAABSA++).
 
 ## Software
HAABSA source code: https://github.com/ofwallaart/HAABSA 
- Updated files: att_layer.py, main.py, main_cross.py and main_hyper.py.
- New files: 
  - Hierarchical Attention: lcrModelAlt_hierarchical_v1, lcrModelAlt_hierarchical_v2, lcrModelAlt_hierarchical_v3, lcrModelAlt_hierarchical_v4;
  - Context-dependent word embeddings: getBERTusingColab.py, prepareBERT.py, prepareELMo.py;
  
 ## Pre-trained word embeddings:
- GloVe: https://nlp.stanford.edu/projects/glove/
- Word2vec: https://code.google.com/archive/p/word2vec/
- FastText: https://fasttext.cc/docs/en/english-vectors.html
