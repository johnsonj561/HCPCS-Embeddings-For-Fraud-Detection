Will use subset of 5M Medicare Part B samples to evaluate the importance of the HCPCS procedure code for Medicare fraud prediction.

HCPCS are high dimensional (> 2000), so will use sparse matrices and XGBoost learner to improve performance.

We will compare three methods on the first iteration:
- without hcpcs code attribute
- with hcpcs code one-hot vector
- with dense embedding of hcpcs code, pre-trained

In a future iteration we will create our own HCPCS embedding, if one does not already exist yet.

Cui2Vec has pre-trained embeddings for medical concepts available at http://cui2vec.dbmi.hms.harvard.edu/./ These have been downloaded and checked, they do not contain embeddings for HCPCS codes.

Choi et al. Learning Low-Dimensional Representations of Medical Concepts have embeddings publicly available, but they do not contain all required embeddings for our Medicare data set. 
The claims_codes_hs_300.txt data set is missing 11.83% of the HCPCS codes that are required for Medicare Part B 2012-2015 data. The stanford_cuis_svd_300.txt.gz data set does not contain any HCPCS embeddings. The claims_cuis_hs_300.txt file doesnot contain HCPCS embeddings. The DeVine_etal_200.txt file also does not contain any HCPCS embeddings. We can use the claims_codes_hs_300 embeddings as is and provide a constant unknown value for the missing 11%, but this is not ideal.

Med2Vec learns both codes and patient visit representations. We are only interested in code representations. They provide instructions to reproduce their embeddings, and it is probably advantageous to use their codes that are learned on a large medical corpora.

One thing to our advantage would be to learn embeddings from nothing more than the training data. This is valid, because we don't always have access to patient visit data or doctor notes. We also may not have access to large medical corpora. Sometimes, all we have is our training data!
