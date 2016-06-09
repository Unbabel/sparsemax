# Attentive RNN with Sparsemax

This project allows to reproduce our SNLI experiments in the paper [1] below.

## Installation
```
make
```

## Preliminary steps

1. Obtain the SNLI dataset (http://nlp.stanford.edu/projects/snli/) and copy
the training, dev and test files to:

    ```
    ../data/snli_tok_train.txt
    ../data/snli_tok_dev.txt
    ../data/snli_tok_test.txt
    ```

    The format is: ```<label> <premise> <hypothesis>```, where the label is
    ```neutral```, ```entailment``` or ```contradiction```, and ```<premise>``` and
    ```<hypothesis>``` are tokenized sentences (words separated by whitespaces).
2. Obtain the GloVe word embeddings (http://nlp.stanford.edu/projects/glove/),
filter the words that occur in the SNLI dataset (to save memory) and put them
in:

    ```
    ../data/word_vectors_glove.txt
    ```

    Note: the first line of that file will be ignored.

## Running the system
```
cd scripts
```
To train the model without attention:
```
./run_snli_glove_dropout_attention_types.sh 0 0 0.0003 0.0001 &
```
To train the model with logistic attention:
```
./run_snli_glove_dropout_attention_types.sh 1 0 0.0003 0.0001 &
```
To train the model without softmax attention:
```
./run_snli_glove_dropout_attention_types.sh 1 1 0.0003 0.0001 &
```
To train the model without sparsemax attention:
```
./run_snli_glove_dropout_attention_types.sh 1 2 0.0003 0.0001 &
```

## References

[1] André F. T. Martins and Ramon Astudillo.  
"From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification."  
International Conference on Machine Learning (ICML'16), New York, USA, June 2016.  