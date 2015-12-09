for eta in 10 100 1000
do
    for regularizer in 0.0000001 0.000001 0.00001
    do
        for mode in softmax sparsemax
        do
            echo $mode $eta $regularizer
            python -u train_multilabel_classifier.py ${mode} 20 ${eta} ${regularizer} \
                /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_0.svm \
                /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_1.svm \
                /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
                log_full6_${mode}_eta-${eta}_regularizer-${regularizer}.txt
        done
    done
done


