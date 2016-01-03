dataset=$1
mode=$2
#for regularizer in 10.0 1.0 0.1 0.01 0.001 0.0001
for regularizer in 1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001
do
    echo $mode $regularizer
    python -u train_multilabel_classifier_lbfgs.py ${mode} 100 ${regularizer} \
	${dataset}-train.txt ${dataset}-test.txt ${dataset}-test.txt 1 5 >& \
        log_lbfgs2_${dataset}_norm_jack_dev_${mode}_epochs-100_regularizer-${regularizer}.txt

    python -u train_multilabel_classifier_lbfgs.py ${mode} 100 ${regularizer} \
	${dataset}-train.txt ${dataset}-test.txt ${dataset}-test.txt 1 1 >& \
        log_lbfgs2_${dataset}_norm_test_${mode}_epochs-100_regularizer-${regularizer}.txt
done
