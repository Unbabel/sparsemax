dataset=$1
for regularizer in 10.0 1.0 0.1 0.01 0.001 0.0001
do
    mode=logistic
    echo $mode $regularizer
    python -u train_multilabel_classifier_lbfgs.py ${mode} 100 ${regularizer} \
	${dataset}-train.txt ${dataset}-test.txt ${dataset}-test.txt 1 1 >& \
        log_${dataset}_lbfgs_bias_norm_dev_${mode}_epochs-100_regularizer-${regularizer}.txt
done
