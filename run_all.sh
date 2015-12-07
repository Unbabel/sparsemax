for eta in 100 #1 10 100
do
    #for regularizer in 0.01 0.001 0.0001
    #for regularizer in 0.001 0.0001 0.00001
    #for regularizer in 0.0001 0.00001
    for regularizer in 0.0000001
    do
        for mode in softmax
        do
            echo $mode $eta $regularizer
            python -u multilabel_experiment.py ${mode} 20 ${eta} ${regularizer} \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_train_1.svm \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_test_1.svm \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_test_1.svm >& \
                log_new2_${mode}_eta-${eta}_regularizer-${regularizer}.txt
        done
    done
done


for eta in 1000 #1 10 100
do
    #for regularizer in 0.01 0.001 0.0001
    #for regularizer in 0.001 0.0001 0.00001
    #for regularizer in 0.0001 0.00001
    for regularizer in 0.000001 0.0000001
    do
        for mode in sparsemax softmax
        do
            echo $mode $eta $regularizer
            python -u multilabel_experiment.py ${mode} 20 ${eta} ${regularizer} \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_train_1.svm \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_test_1.svm \
                ~/research/corpora/multilabel/rcv1v2/rcv1subset_topics_test_1.svm >& \
                log_new2_${mode}_eta-${eta}_regularizer-${regularizer}.txt
        done
    done
done

