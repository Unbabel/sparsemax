for eta in 50 5
do
    for regularizer in 0.0001 0.00001
    do
	mode=sparsemax
        echo $mode $eta $regularizer
        python -u train_multilabel_classifier.py ${mode} 20 ${eta} ${regularizer} \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_0.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_1.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
                log_full_macro2_${mode}_eta-${eta}_regularizer-${regularizer}.txt
    done
done

#for eta in 100 10 1000
for eta in 50 200
do
    for regularizer in 0.0000001 0.000001 0.00001
    do
	mode=softmax
        echo $mode $eta $regularizer
        python -u train_multilabel_classifier.py ${mode} 20 ${eta} ${regularizer} \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_0.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_1.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
                log_full_macro2_${mode}_eta-${eta}_regularizer-${regularizer}.txt
    done
done

#for eta in 50 10 100
for eta in 40 60 70 
do
    for regularizer in 0.0000001 0.000001 0.00001 0.0001
    do
	mode=logistic
        echo $mode $eta $regularizer
        python -u train_multilabel_classifier.py ${mode} 20 ${eta} ${regularizer} \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_0.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_train_1.svm \
            /mnt/data/corpora/multilabel/rcv1v2_full/rcv1_topics_test.svm >& \
                log_full_macro2_${mode}_eta-${eta}_regularizer-${regularizer}.txt
    done
done

