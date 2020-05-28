# HAABSA++
The code for A Hybrid Approach for Aspect-Based Sentiment Analysis Using Contextual Word Emmbeddings and Hierarchical Attention

The hybrid approach for aspect-based sentiment analysis (HAABSA) is a two-step method that classifies target sentiments using a domain sentiment ontology and a Multi-Hop LCR-Rot model as backup.
 - HAABSA paper: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf
 
 Keeping the ontology, we optimise the embedding layer of the backup neural network with context-dependent word embeddings and integrate hierarchical attention in the model's architecture (HAABSA++).
 
 ## Software
The HAABSA source code: https://github.com/ofwallaart/HAABSA needs to be installed. Then the following changes need to be done:
- Update files: config.py, att_layer.py, main.py, main_cross.py and main_hyper.py.
- Add files: 
  - Context-dependent word embeddings: 
    - getBERTusingColab.py (extract the BERT word embeddings);
    - prepareBERT.py (prepare the final BERT emebdding matrix, training and tesing datasets);
    - prepareELMo.py (extract the ELMo word emebddings and prepare the final ELMo embedding matrix, training and testing datasets);
    - raw_data2015.txt, raw_data2016.txt (Data folder).
  - Hierarchical Attention: 
    - lcrModelAlt_hierarchical_v1 (first method);
    - lcrModelAlt_hierarchical_v2 (second method);
    - lcrModelAlt_hierarchical_v3 (third method);
    - lcrModelAlt_hierarchical_v4 (fourth method).

The training and testing datasets are in the Data folder for SemEval 2015 and SemEval 2016. The files are available for Glove, ELMo and BERT word emebddings. 

*Even if the model is trained with contextul word emebddings, the ontology has to run on a dataset special designed for the non-contextual case.
  
 ## Word embeddings
 - GloVe word embeddings (SemEval 2015): https://drive.google.com/file/d/14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu/view?usp=sharing
 - GloVe word embeddings (SemEval 2016): https://drive.google.com/file/d/1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92/view?usp=sharing
 - ELMo word embeddings (SemEval 2015): https://drive.google.com/file/d/1GfHKLmbiBEkATkeNmJq7CyXGo61aoY2l/view?usp=sharing
 - ELMo word embeddings (SemEval 2016): https://drive.google.com/file/d/1OT_1p55LNc4vxc0IZksSj2PmFraUIlRD/view?usp=sharing
 - BERT word embeddings (SemEval 2015): https://drive.google.com/file/d/1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx/view?usp=sharing
 - BERT word embeddings (SemEval 2016): https://drive.google.com/file/d/1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2/view?usp=sharing
 
Download pre-trained word emebddings: 
- GloVe: https://nlp.stanford.edu/projects/glove/
- Word2vec: https://code.google.com/archive/p/word2vec/
- FastText: https://fasttext.cc/docs/en/english-vectors.html
