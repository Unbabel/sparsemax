
embedding_name=glove
warm_start_on_epoch=0
use_attention=$1 #1
attention_type=$2 #2
learning_rate=$3 #0.0003
regularization_constant=$4 #0.0001
word_vector_file=../../data/word_vectors_${embedding_name}.txt

model_prefix=exp_attentiontypes_lstm-false_dropout-true_embedding-${embedding_name}_attention-${use_attention}_type-${attention_type}_startepoch-${warm_start_on_epoch}_eta-${learning_rate}_lambda-${regularization_constant}

echo ${model_prefix}

mkdir -p ${model_prefix}

../attentive_rnn train \
    ../../data/snli_tok_train.txt \
    ../../data/snli_tok_dev.txt \
    ../../data/snli_tok_test.txt \
    ${word_vector_file} \
    ${use_attention} \
    ${attention_type} \
    100 \
    ${warm_start_on_epoch} \
    200 \
    25 \
    ${learning_rate} \
    ${regularization_constant} \
    ${model_prefix}/ >& \
    log_${model_prefix}.txt


    
    
    
