mode=$1
num_classes=$2
#for length in 1100 1200 1300 1400 1600 1700 1800 1900 #50 100 #500 1000 1500 2000 2500 3000
for length in 200 400 600 800 1000 1200 1400 1600 1800 2000
do
#    ./run_lbfgs_dataset.sh toy-${num_classes}-classes-${length}-length ${mode} &
    ./run_lbfgs_dataset.sh toy-proportions-${num_classes}-classes-${length}-length ${mode} &
done

