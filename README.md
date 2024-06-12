# CoCoRF
We propose a comment-based codabase refining framework utilizing unsupervised and supervised co-learning. It applies manually defined rules for syntax filtering and utilizes the WTFF mining algorithm to construct bootstrap query corpus for training the TVAE model to further semantic filtering.

The primary structure of the project is as follows:

```python
- Code_Search            # Framework effectiveness verification on the code search model DeepCS
- TVAE_master            # The TAVE-based semantic filter
- cocorf                 # Comment-based data refinement framework    
```

The project requires Python3.6 and the following packages:

```python
jsonlines
pandas
javalang==0.12.0
nltk
torch==1.3.1
numpy
tqdm
tables==3.4.3
```
