for regularizer in 1e-5 1e-6 1e-7
do
    mode=softmax
    echo $mode $regularizer
    python -u train_multilabel_classifier_lbfgs.py ${mode} 100 ${regularizer} \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train.svm \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm \
        /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm 0 1 >& \
        log_lbfgs2_reuters_test_${mode}_epochs-100_regularizer-${regularizer}.txt
done
