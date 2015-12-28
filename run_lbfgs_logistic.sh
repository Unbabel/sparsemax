for regularizer in 0.0000001
do
    mode=logistic
    echo $mode $regularizer
    python -u train_multilabel_classifier_lbfgs.py ${mode} 50 ${regularizer} \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train.svm \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
        log_lbfgs_bias_test_${mode}_epochs-50_regularizer-${regularizer}.txt

#    python -u train_multilabel_classifier_lbfgs.py ${mode} 100 ${regularizer} \
#        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_0.svm \
#        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_1.svm \
#        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
#        log_lbfgs_bias_dev_${mode}_epochs-100_regularizer-${regularizer}.txt
done
