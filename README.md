# CSNMF
We use the model generated from the attention-based VAE training to filter the raw data, then train a new code search model on the filtered dataset and evaluate the performance of the trained model.

The primary structure of the project is as follows:

```python
- data                  # directory to store the vocab
- main_model            # directory to store the vae model
- models                # architecture of DeepCS     
- output                # directory to store the trained model
- processed_dataset_csn # directory to store the filtered and processed dataset
- raw_dataset           # directory to store the CodeSearchNet dataset
- attention_model.py    # Attention-based model structure
- configs.py            # the configuration file for DeepCS
- data_loader.py        # the data loader for DeepCS
- filtering.py          # our data filtering and processing script for DeepCS
- gru_model.py          # GRU-based model structure
- modules.py            # the modules for DeepCS
- repr_code.py          # the script to evaluate DeepCS model
- train.py              # the script to train DeepCS model
- utils.py              # utility functions for DeepCS
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

## Filter the dataset and process the filtered dataset into a format that can be used by DeepCS.

Run the following command for filtering：

```python
python filtering.py
```

The filtered and processed dataset is saved in ```./processed_dataset/```. We have pre-run this command and generated a processed dataset in this directory. Here we have uploaded only the processed data from the attention-based VAE model, for other models and their processing generated data click [here]([URL] "https://github.com/Zxqinn/CSNMF/tree/main") to download.

## Train the code search model on the filtered dataset.

Run the following command to train the model:

```python
python train.py
```

The trained model is saved in ```./output/```. We have pre-run this command and obtained a trained model ```./output/epo100.h5```.

## Evaluate the performance of the trained model.

Run the following command to evaluate the trained model:

```python
python repr_code.py
```

The evaluation results will be printed in the terminal.
