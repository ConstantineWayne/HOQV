# HOQV
## Usage
### Prerequisites
- Python 3.8
- PyTorch 1.9.0
- CUDA 11.4

### Datasets
Change the "--data_path" in the args to the path where you store the MVSA-Single dataset, and change the '--bert_model' in the args to the path you store the bertmodel. You can download the dataset through the following url: [MVSA-Single]([https://github.com/cvdfoundation/kinetics-dataset](https://www.kaggle.com/datasets/vincemarcs/mvsasingle), [BertModel](https://huggingface.co/google-bert/bert-base-uncased).

### Training
In order to train the model, you can use  
```python
bash my_train.sh
 ```
in the terminal.
To evaluate the model's performances under disturbances, you can use 
```python
bash my_eval.sh
 ```
in the terminal.
