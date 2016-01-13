#ifndef RNN_H_
#define RNN_H_

#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "snli_data.h"
#include "layer.h"

class RNN {
 public:
  RNN() {}
  RNN(Dictionary *dictionary,
      int embedding_dimension,
      int hidden_size,
      int output_size,
      bool use_attention,
      bool sparse_attention) : dictionary_(dictionary) {
    write_attention_probabilities_ = false;
    use_ADAM_ = false; //true;
    use_attention_ = use_attention;
    sparse_attention_ = sparse_attention;
    input_size_ = hidden_size; // Size of the projected embedded words.
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    activation_function_ = ActivationFunctions::TANH; //LOGISTIC;
    lookup_layer_ = new LookupLayer<float>(dictionary->GetNumWords(),
                                           embedding_dimension);
    linear_layer_ = new LinearLayer<float>(embedding_dimension, input_size_);
    //rnn_layer_ = new RNNLayer(input_size_, hidden_size);
    rnn_layer_ = new GRULayer<float>(input_size_, hidden_size);
    if (use_attention_) {
      attention_layer_ = new AttentionLayer<float>(hidden_size, hidden_size,
                                                   hidden_size,
                                                   sparse_attention_);
    } else {
      attention_layer_ = NULL;
    }
    hypothesis_selector_layer_ = new SelectorLayer<float>;
    if (use_attention_) {
      premise_extractor_layer_ = new SelectorLayer<float>;
      premise_selector_layer_ = NULL;
    } else {
      premise_extractor_layer_ = NULL;
      premise_selector_layer_ = new SelectorLayer<float>;
    }
    concatenator_layer_ = new ConcatenatorLayer<float>;
    feedforward_layer_ = new FeedforwardLayer<float>(2*hidden_size,
                                                     hidden_size);
    output_layer_ = new SoftmaxOutputLayer<float>(hidden_size, output_size);
  }
  virtual ~RNN() {
    delete lookup_layer_;
    delete linear_layer_;
    delete rnn_layer_;
    delete attention_layer_;
    delete hypothesis_selector_layer_;
    delete premise_extractor_layer_;
    delete premise_selector_layer_;
    delete concatenator_layer_;
    delete feedforward_layer_;
    delete output_layer_;
  }

  int GetEmbeddingSize() { return lookup_layer_->embedding_dimension(); }
  int GetInputSize() { return linear_layer_->output_size(); }

  virtual void CollectAllParameters(std::vector<FloatMatrix*> *weights,
                                    std::vector<FloatVector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    weights->clear();
    biases->clear();
    weight_names->clear();
    bias_names->clear();

    lookup_layer_->CollectAllParameters(weights, biases, weight_names,
                                        bias_names);

    linear_layer_->CollectAllParameters(weights, biases, weight_names,
                                        bias_names);

    rnn_layer_->CollectAllParameters(weights, biases, weight_names,
				     bias_names);

    if (use_attention_) {
      attention_layer_->CollectAllParameters(weights, biases, weight_names,
                                             bias_names);
    }

    feedforward_layer_->CollectAllParameters(weights, biases, weight_names,
                                             bias_names);

    output_layer_->CollectAllParameters(weights, biases, weight_names,
                                        bias_names);
  }

  void InitializeParameters() {
    srand(1234);

    lookup_layer_->InitializeParameters();
    linear_layer_->InitializeParameters();
    rnn_layer_->InitializeParameters();
    if (use_attention_) {
      attention_layer_->InitializeParameters();
    }
    feedforward_layer_->InitializeParameters();
    output_layer_->InitializeParameters();

    if (use_ADAM_) {
      double beta1 = 0.9;
      double beta2 = 0.999;
      double epsilon = 1e-8; //1e-5; // 1e-8;
      linear_layer_->InitializeADAM(beta1, beta2, epsilon);
      rnn_layer_->InitializeADAM(beta1, beta2, epsilon);
      if (use_attention_) attention_layer_->InitializeADAM(beta1, beta2, epsilon);
      feedforward_layer_->InitializeADAM(beta1, beta2, epsilon);
      output_layer_->InitializeADAM(beta1, beta2, epsilon);
    }
  }

  void SetFixedEmbeddings(const FloatMatrix &fixed_embeddings,
                          const std::vector<int> &word_ids) {
    lookup_layer_->SetFixedEmbeddings(fixed_embeddings, word_ids);
  }

  void Train(const std::vector<std::vector<Input> > &input_sequences,
             const std::vector<int> &output_labels,
             const std::vector<std::vector<Input> > &input_sequences_dev,
             const std::vector<int> &output_labels_dev,
             const std::vector<std::vector<Input> > &input_sequences_test,
             const std::vector<int> &output_labels_test,
             int num_epochs,
             double learning_rate,
             double regularization_constant) {

    // Initial performance.
    double accuracy_dev = 0.0;
    int num_sentences_dev = input_sequences_dev.size();
    for (int i = 0; i < input_sequences_dev.size(); ++i) {
      int predicted_label;
      Run(input_sequences_dev[i], &predicted_label);
      if (output_labels_dev[i] == predicted_label) {
        accuracy_dev += 1.0;
      }
    }
    accuracy_dev /= num_sentences_dev;
    std::cout << " Initial accuracy dev: " << accuracy_dev
              << std::endl;

    //assert(false);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      TrainEpoch(input_sequences, output_labels,
                 input_sequences_dev, output_labels_dev,
                 input_sequences_test, output_labels_test,
                 epoch, learning_rate, regularization_constant);
    }
  }

  void TrainEpoch(const std::vector<std::vector<Input> > &input_sequences,
                  const std::vector<int> &output_labels,
                  const std::vector<std::vector<Input> > &input_sequences_dev,
                  const std::vector<int> &output_labels_dev,
                  const std::vector<std::vector<Input> > &input_sequences_test,
                  const std::vector<int> &output_labels_test,
                  int epoch,
                  double learning_rate,
                  double regularization_constant) {
    timeval start, end;
    gettimeofday(&start, NULL);
    double total_loss = 0.0;
    double accuracy = 0.0;
    int num_sentences = input_sequences.size();
    for (int i = 0; i < input_sequences.size(); ++i) {
      RunForwardPass(input_sequences[i]);
      FloatVector p = output_layer_->GetOutput();
      double loss = -log(p(output_labels[i]));
      int prediction;
      p.maxCoeff(&prediction);
      if (prediction == output_labels[i]) {
        accuracy += 1.0;
      }
      total_loss += loss;
      RunBackwardPass(input_sequences[i], output_labels[i], learning_rate,
                      regularization_constant);
    }
    accuracy /= num_sentences;

    double accuracy_dev = 0.0;
    int num_sentences_dev = input_sequences_dev.size();
    for (int i = 0; i < input_sequences_dev.size(); ++i) {
      int predicted_label;
      Run(input_sequences_dev[i], &predicted_label);
      if (output_labels_dev[i] == predicted_label) {
        accuracy_dev += 1.0;
      }
    }
    accuracy_dev /= num_sentences_dev;

    write_attention_probabilities_ = true;
    if (sparse_attention_) {
      os_attention_.open("sparse_attention.txt", std::ifstream::out);
    } else {
      os_attention_.open("soft_attention.txt", std::ifstream::out);
    }

    double accuracy_test = 0.0;
    int num_sentences_test = input_sequences_test.size();
    for (int i = 0; i < input_sequences_test.size(); ++i) {
      int predicted_label;
      Run(input_sequences_test[i], &predicted_label);
      if (output_labels_test[i] == predicted_label) {
        accuracy_test += 1.0;
      }
    }
    accuracy_test /= num_sentences_test;

    write_attention_probabilities_ = false;
    os_attention_.flush();
    os_attention_.clear();
    os_attention_.close();

    gettimeofday(&end, NULL);
    std::cout << "Epoch: " << epoch+1
              << " Total loss: " << total_loss
              << " Accuracy train: " << accuracy
              << " Accuracy dev: " << accuracy_dev
              << " Accuracy test: " << accuracy_test
              << " Time: " << diff_ms(end,start)
              << std::endl;
  }

  void Run(const std::vector<Input> &input_sequence,
           int *predicted_label) {
    RunForwardPass(input_sequence);
    int prediction;
    FloatVector p = output_layer_->GetOutput();
    p.maxCoeff(&prediction);
    *predicted_label = prediction;
  }

  void RunForwardPass(const std::vector<Input> &input_sequence) {
    // Look for the separator symbol.
    int wid_sep = dictionary_->GetWordId("__START__");
    int separator = -1;
    for (separator = 0; separator < input_sequence.size(); ++separator) {
      if (input_sequence[separator].wid() == wid_sep) break;
    }
    assert(separator < input_sequence.size());

    lookup_layer_->set_input_sequence(input_sequence);
    lookup_layer_->RunForward();

    linear_layer_->SetNumInputs(1);
    linear_layer_->SetInput(0, lookup_layer_->GetOutput());
    linear_layer_->RunForward();

    rnn_layer_->SetNumInputs(1);
    rnn_layer_->SetInput(0, linear_layer_->GetOutput());
    rnn_layer_->RunForward();

    int t = input_sequence.size() - 1;

    if (use_attention_) {

      premise_extractor_layer_->SetNumInputs(1);
      premise_extractor_layer_->SetInput(0, rnn_layer_->GetOutput());
      premise_extractor_layer_->DefineBlock(0, 0, hidden_size_, separator);
      premise_extractor_layer_->RunForward();

      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
      hypothesis_selector_layer_->DefineBlock(0, t, hidden_size_, 1);
      hypothesis_selector_layer_->RunForward();

      attention_layer_->SetNumInputs(2);
      attention_layer_->SetInput(0, premise_extractor_layer_->GetOutput());
      attention_layer_->SetInput(1, hypothesis_selector_layer_->GetOutput());
      attention_layer_->RunForward();

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetInput(0, attention_layer_->GetOutput());
      concatenator_layer_->SetInput(1, hypothesis_selector_layer_->GetOutput());
      concatenator_layer_->RunForward();

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetInput(0, concatenator_layer_->GetOutput());
      feedforward_layer_->RunForward();

    } else {

      premise_selector_layer_->SetNumInputs(1);
      premise_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
      premise_selector_layer_->DefineBlock(0, separator, hidden_size_, 1);
      premise_selector_layer_->RunForward();

      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
      hypothesis_selector_layer_->DefineBlock(0, t, hidden_size_, 1);
      hypothesis_selector_layer_->RunForward();

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetInput(0, premise_selector_layer_->GetOutput());
      concatenator_layer_->SetInput(1, hypothesis_selector_layer_->GetOutput());
      concatenator_layer_->RunForward();

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetInput(0, concatenator_layer_->GetOutput());
      feedforward_layer_->RunForward();
    }

    output_layer_->SetNumInputs(1);
    output_layer_->SetInput(0, feedforward_layer_->GetOutput());
    output_layer_->RunForward();

    if (use_attention_ && write_attention_probabilities_) {
      os_attention_ << attention_layer_->GetAttentionProbabilities().transpose()
                    << std::endl;
    }
  }

  void RunBackwardPass(const std::vector<Input> &input_sequence,
                       int output_label,
                       double learning_rate,
                       double regularization_constant) {
    // Look for the separator symbol.
    int wid_sep = dictionary_->GetWordId("__START__");
    int separator = -1;
    for (separator = 0; separator < input_sequence.size(); ++separator) {
      if (input_sequence[separator].wid() == wid_sep) break;
    }
    assert(separator < input_sequence.size());

    int t = input_sequence.size() - 1;

    // Reset parameter gradients.
    output_layer_->ResetGradients();
    feedforward_layer_->ResetGradients();
    rnn_layer_->ResetGradients();
    linear_layer_->ResetGradients();
    lookup_layer_->ResetGradients();
    if (use_attention_) {
      attention_layer_->ResetGradients();
    }

    // Reset variable derivatives.
    feedforward_layer_->GetOutputDerivative()->setZero(hidden_size_, 1);
    concatenator_layer_->GetOutputDerivative()->setZero(2*hidden_size_, 1);
    if (use_attention_) {
      attention_layer_->GetOutputDerivative()->setZero(hidden_size_, 1);
      premise_extractor_layer_->GetOutputDerivative()->setZero(hidden_size_,
                                                               separator);
      hypothesis_selector_layer_->GetOutputDerivative()->setZero(hidden_size_,
                                                                 1);
    } else {
      premise_selector_layer_->GetOutputDerivative()->setZero(hidden_size_, 1);
      hypothesis_selector_layer_->GetOutputDerivative()->setZero(hidden_size_,
                                                                 1);
    }
    rnn_layer_->GetOutputDerivative()->setZero(hidden_size_,
                                               input_sequence.size());
    linear_layer_->GetOutputDerivative()->setZero(GetInputSize(),
                                                  input_sequence.size());
    lookup_layer_->GetOutputDerivative()->setZero(GetEmbeddingSize(),
                                                  input_sequence.size());

    // Backprop.
    output_layer_->set_output_label(output_label);
    output_layer_->
      SetInputDerivative(0, feedforward_layer_->GetOutputDerivative());
    output_layer_->RunBackward();

    feedforward_layer_->
      SetInputDerivative(0, concatenator_layer_->GetOutputDerivative());
    feedforward_layer_->RunBackward();

    if (use_attention_) {

      concatenator_layer_->
        SetInputDerivative(0, attention_layer_->GetOutputDerivative());
      concatenator_layer_->
        SetInputDerivative(1,
                           hypothesis_selector_layer_->GetOutputDerivative());
      concatenator_layer_->RunBackward();

      attention_layer_->
        SetInputDerivative(0, premise_extractor_layer_->GetOutputDerivative());
      attention_layer_->
        SetInputDerivative(1,
                           hypothesis_selector_layer_->GetOutputDerivative());
      attention_layer_->RunBackward();

      premise_extractor_layer_->
        SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      premise_extractor_layer_->DefineBlock(0, 0, hidden_size_, separator);
      premise_extractor_layer_->RunBackward();

      hypothesis_selector_layer_->
        SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      hypothesis_selector_layer_->DefineBlock(0, t, hidden_size_, 1);
      hypothesis_selector_layer_->RunBackward();

    } else {

      concatenator_layer_->
        SetInputDerivative(0, premise_selector_layer_->GetOutputDerivative());
      concatenator_layer_->
        SetInputDerivative(1,
                           hypothesis_selector_layer_->GetOutputDerivative());
      concatenator_layer_->RunBackward();

      premise_selector_layer_->
        SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      premise_selector_layer_->DefineBlock(0, separator, hidden_size_, 1);
      premise_selector_layer_->RunBackward();

      hypothesis_selector_layer_->
        SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      hypothesis_selector_layer_->DefineBlock(0, t, hidden_size_, 1);
      hypothesis_selector_layer_->RunBackward();

    }

    rnn_layer_->SetInputDerivative(0, linear_layer_->GetOutputDerivative());
    rnn_layer_->RunBackward();

    linear_layer_->SetInputDerivative(0, lookup_layer_->GetOutputDerivative());
    linear_layer_->RunBackward();

    lookup_layer_->RunBackward();

    // Check gradient.
    bool check_gradient = false; //true;
    int num_checks = 20;
    double delta = 1e-7;
    if (check_gradient) {
      FloatMatrix *output_derivative = output_layer_->GetOutputDerivative();
      const FloatMatrix &output = output_layer_->GetOutput();
      output_derivative->setZero(output_size_, 1);
      int l = output_layer_->output_label();
      (*output_derivative)(l) = -1.0 / output(l);
      output_layer_->CheckGradient(num_checks, delta);
      if (use_attention_) {
        attention_layer_->CheckGradient(num_checks, delta);
      }
      feedforward_layer_->CheckGradient(num_checks, delta);
      rnn_layer_->CheckGradient(num_checks, delta);
      linear_layer_->CheckGradient(num_checks, delta);
    }

    // Update parameters.
    if (use_ADAM_) {
      output_layer_->UpdateParametersADAM(learning_rate,
                                          regularization_constant);
      if (use_attention_) {
        attention_layer_->UpdateParametersADAM(learning_rate,
                                               regularization_constant);
      }
      feedforward_layer_->UpdateParametersADAM(learning_rate,
                                               regularization_constant);
      rnn_layer_->UpdateParametersADAM(learning_rate,
                                       regularization_constant);
      linear_layer_->UpdateParametersADAM(learning_rate,
                                          regularization_constant);
      //lookup_layer_->UpdateParameters(learning_rate);
    } else {
      output_layer_->UpdateParameters(learning_rate,
                                      regularization_constant);
      if (use_attention_) {
        attention_layer_->UpdateParameters(learning_rate,
                                           regularization_constant);
      }
      feedforward_layer_->UpdateParameters(learning_rate,
                                           regularization_constant);
      rnn_layer_->UpdateParameters(learning_rate,
                                   regularization_constant);
      linear_layer_->UpdateParameters(learning_rate,
                                      regularization_constant);

      // NOTE: no regularization_constant for the lookup layer.
      lookup_layer_->UpdateParameters(learning_rate, 0.0);
    }
  }

 protected:
  Dictionary *dictionary_;
  int activation_function_;
  LookupLayer<float> *lookup_layer_;
  LinearLayer<float> *linear_layer_;
  RNNLayer<float> *rnn_layer_;
  AttentionLayer<float> *attention_layer_;
  FeedforwardLayer<float> *feedforward_layer_;
  SelectorLayer<float> *hypothesis_selector_layer_;
  SelectorLayer<float> *premise_selector_layer_;
  SelectorLayer<float> *premise_extractor_layer_;
  ConcatenatorLayer<float> *concatenator_layer_;
  SoftmaxOutputLayer<float> *output_layer_;
  int input_size_;
  int hidden_size_;
  int output_size_;
  bool use_attention_;
  bool sparse_attention_;
  bool use_ADAM_;
  bool write_attention_probabilities_;
  std::ofstream os_attention_;

  //FloatMatrix x_;
  //FloatMatrix h_;
};

#if 0
class BiRNN : public RNN {
 public:
  BiRNN(Dictionary *dictionary,
        int window_size, int embedding_dimension, int affix_embedding_dimension,
        int hidden_size, int output_size) {
    activation_function_ = ActivationFunctions::LOGISTIC;
    dictionary_ = dictionary;
    window_size_ = window_size;
    embedding_dimension_ = embedding_dimension;
    affix_embedding_dimension_ = affix_embedding_dimension;
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    use_hidden_start_ = true;
  }
  ~BiRNN() {}

  void CollectAllParameters(std::vector<FloatMatrix*> *weights,
                            std::vector<FloatVector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    RNN::CollectAllParameters(weights, biases, weight_names, bias_names);

    Wxl_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Wll_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    Wly_ = FloatMatrix::Zero(output_size_, hidden_size_);
    bl_ = FloatVector::Zero(hidden_size_, 1);
    if (use_hidden_start_) {
      l0_ = FloatVector::Zero(hidden_size_);
    }

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);
    weights->push_back(&Wly_);

    biases->push_back(&bl_);
    if (use_hidden_start_) {
      biases->push_back(&l0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");
    weight_names->push_back("Wly");

    bias_names->push_back("bl");
    if (use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void RunForwardPass(const std::vector<Input> &input_sequence) {
    RunForwardLookupLayer(input_sequence);

    int hidden_size = Whh_.rows();
    int output_size = Why_.rows();

    h_.setZero(hidden_size, input_sequence.size());
    FloatVector hprev = FloatVector::Zero(h_.rows());
    if (use_hidden_start_) hprev = h0_;
    for (int t = 0; t < input_sequence.size(); ++t) {
      FloatMatrix result;
      EvaluateActivation(activation_function_,
                         Wxh_ * x_.col(t) + bh_ + Whh_ * hprev,
                         &result);
      h_.col(t) = result;

#if 0
      //std::cout << "x[" << t << "] = " <<  x_.col(t).transpose() << std::endl;
      //std::cout << "Wxh*x[" << t << "] = " <<  (Wxh_ * x_.col(t)).transpose() << std::endl;
      //std::cout << "Whh*hprev[" << t << "] = " <<  (Whh_ * hprev).transpose() << std::endl;
      //std::cout << "bh = " <<  bh_.transpose() << std::endl;
      std::cout << "h[" << t << "] = " << h_.col(t).transpose() << std::endl;
#endif

      hprev = h_.col(t);
    }

    l_.setZero(hidden_size, input_sequence.size());
    FloatVector lnext = FloatVector::Zero(l_.rows());
    if (use_hidden_start_) lnext = l0_;
    for (int t = input_sequence.size() - 1; t >= 0; --t) {
      FloatMatrix result;
      EvaluateActivation(activation_function_,
                         Wxl_ * x_.col(t) + bl_ + Wll_ * lnext,
                         &result);
      l_.col(t) = result;

#if 0
      std::cout << "l[" << t << "] = " << l_.col(t).transpose() << std::endl;
#endif

      lnext = l_.col(t);
    }

    y_.setZero(output_size, input_sequence.size());
    p_.setZero(output_size, input_sequence.size());
    for (int t = 0; t < input_sequence.size(); ++t) {
      y_.col(t) = Why_ * h_.col(t) + Wly_ * l_.col(t) + by_;
      double logsum = LogSumExp(y_.col(t));
      p_.col(t) = (y_.col(t).array() - logsum).exp();

#if 0
      std::cout << "p[" << t << "] = " << p_.col(t).transpose() << std::endl;
#endif

    }
  }

  void RunBackwardPass(const std::vector<Input> &input_sequence,
                       const std::vector<int> &output_sequence,
                       double learning_rate) {
    FloatMatrix dWhy = FloatMatrix::Zero(Why_.rows(), Why_.cols());
    FloatMatrix dWhh = FloatMatrix::Zero(Whh_.rows(), Whh_.cols());
    FloatMatrix dWxh = FloatMatrix::Zero(Wxh_.rows(), Wxh_.cols());
    FloatVector dby = FloatVector::Zero(Why_.rows());
    FloatVector dbh = FloatVector::Zero(Whh_.rows());
    FloatVector dhnext = FloatVector::Zero(Whh_.rows());

    FloatMatrix dWly = FloatMatrix::Zero(Wly_.rows(), Wly_.cols());
    FloatMatrix dWll = FloatMatrix::Zero(Wll_.rows(), Wll_.cols());
    FloatMatrix dWxl = FloatMatrix::Zero(Wxl_.rows(), Wxl_.cols());
    FloatVector dbl = FloatVector::Zero(Wll_.rows());
    FloatVector dlprev = FloatVector::Zero(Wll_.rows());

    FloatMatrix dx = FloatMatrix::Zero(GetInputSize(), input_sequence.size());

    for (int t = 0; t < input_sequence.size(); ++t) {
      FloatVector dy = p_.col(t);
      dy[output_sequence[t]] -= 1.0; // Backprop into y (softmax grad).

      dWly += dy * l_.col(t).transpose();

      FloatVector dl = Wly_.transpose() * dy + dlprev; // Backprop into l.
      FloatMatrix dlraw;
      DerivateActivation(activation_function_, l_.col(t), &dlraw);
      dlraw = dlraw.array() * dl.array();

      dWxl += dlraw * x_.col(t).transpose();
      dbl += dlraw;
      if (t < input_sequence.size() - 1) {
        dWll += dlraw * l_.col(t+1).transpose();
      }
      dlprev.noalias() = Wll_.transpose() * dlraw;

      dx.col(t) = Wxl_.transpose() * dlraw; // Backprop into x.
    }

    for (int t = input_sequence.size() - 1; t >= 0; --t) {
      FloatVector dy = p_.col(t);
      dy[output_sequence[t]] -= 1.0; // Backprop into y (softmax grad).

      dWhy += dy * h_.col(t).transpose();
      dby += dy;

      FloatVector dh = Why_.transpose() * dy + dhnext; // Backprop into h.
      FloatMatrix dhraw;
      DerivateActivation(activation_function_, h_.col(t), &dhraw);
      dhraw = dhraw.array() * dh.array();

      dWxh += dhraw * x_.col(t).transpose();
      dbh += dhraw;
      if (t > 0) {
        dWhh += dhraw * h_.col(t-1).transpose();
      }
      dhnext.noalias() = Whh_.transpose() * dhraw;

      dx.col(t) += Wxh_.transpose() * dhraw; // Backprop into x.
    }

    Why_ -= learning_rate * dWhy;
    by_ -= learning_rate * dby;
    Wxh_ -= learning_rate * dWxh;
    bh_ -= learning_rate * dbh;
    Whh_ -= learning_rate * dWhh;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dhnext;
    }

    Wly_ -= learning_rate * dWly;
    Wxl_ -= learning_rate * dWxl;
    bl_ -= learning_rate * dbl;
    Wll_ -= learning_rate * dWll;
    if (use_hidden_start_) {
      l0_ -= learning_rate * dlprev;
    }

    RunBackwardLookupLayer(input_sequence, dx, learning_rate);
  }

 protected:
  FloatMatrix Wxl_;
  FloatMatrix Wll_;
  FloatMatrix Wly_;
  FloatVector bl_;
  FloatVector l0_;

  FloatMatrix l_;
};


class RNN_GRU : public RNN {
 public:
  RNN_GRU() {}
  RNN_GRU(Dictionary *dictionary,
          int window_size, int embedding_dimension, int affix_embedding_dimension,
          int hidden_size, int output_size) {
    dictionary_ = dictionary;
    activation_function_ = ActivationFunctions::LOGISTIC;
    window_size_ = window_size;
    embedding_dimension_ = embedding_dimension;
    affix_embedding_dimension_ = affix_embedding_dimension;
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    use_hidden_start_ = true;
  }
  virtual ~RNN_GRU() {}

  virtual void CollectAllParameters(std::vector<FloatMatrix*> *weights,
                                    std::vector<FloatVector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    RNN::CollectAllParameters(weights, biases, weight_names, bias_names);

    Wxz_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Whz_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    Wxr_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Whr_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    bz_ = FloatVector::Zero(hidden_size_, 1);
    br_ = FloatVector::Zero(hidden_size_, 1);

    weights->push_back(&Wxz_);
    weights->push_back(&Whz_);
    weights->push_back(&Wxr_);
    weights->push_back(&Whr_);

    biases->push_back(&bz_);
    biases->push_back(&br_);

    weight_names->push_back("Wxz");
    weight_names->push_back("Whz");
    weight_names->push_back("Wxr");
    weight_names->push_back("Whr");

    bias_names->push_back("bz");
    bias_names->push_back("br");
  }

  virtual void RunForwardPass(const std::vector<Input> &input_sequence) {
    RunForwardLookupLayer(input_sequence);

    int hidden_size = Whh_.rows();
    int output_size = Why_.rows();

    z_.setZero(hidden_size, input_sequence.size());
    r_.setZero(hidden_size, input_sequence.size());
    hu_.setZero(hidden_size, input_sequence.size());
    h_.setZero(hidden_size, input_sequence.size());
    FloatVector hprev = FloatVector::Zero(h_.rows());
    if (use_hidden_start_) hprev = h0_;
    for (int t = 0; t < input_sequence.size(); ++t) {
      FloatMatrix result;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxz_ * x_.col(t) + bz_ + Whz_ * hprev,
                         &result);
      z_.col(t) = result;

      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxr_ * x_.col(t) + br_ + Whr_ * hprev,
                         &result);
      r_.col(t) = result;

      EvaluateActivation(activation_function_,
                         Wxh_ * x_.col(t) + bh_ + Whh_ * r_.col(t).cwiseProduct(hprev),
                         &result);
      hu_.col(t) = result;

      //h_.col(t) = z_.col(t) * hu_.col(t) + (1.0 - z_.col(t)) * hprev;
      h_.col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;

      //FloatVector hraw = Wxh_ * x_.col(t) + bh_;
      //if (t > 0) hraw += Whh_ * h_.col(t-1);
      //FloatMatrix result;
      //EvaluateActivation(activation_function_, hraw, &result);
      //h_.col(t) = result;

      hprev = h_.col(t);
    }

    y_.setZero(output_size, input_sequence.size());
    p_.setZero(output_size, input_sequence.size());
    //FloatMatrix y = (Why_ * h).colwise() + by_;
    //FloatVector logsums = LogSumExpColumnwise(y);
    //FloatMatrix p = (y.rowwise() - logsums.transpose()).array().exp();
    for (int t = 0; t < input_sequence.size(); ++t) {
      y_.col(t) = Why_ * h_.col(t) + by_;
      double logsum = LogSumExp(y_.col(t));
      p_.col(t) = (y_.col(t).array() - logsum).exp();
    }
  }

  virtual void RunBackwardPass(const std::vector<Input> &input_sequence,
                       const std::vector<int> &output_sequence,
                       double learning_rate) {
    FloatMatrix dWhy = FloatMatrix::Zero(Why_.rows(), Why_.cols());
    FloatMatrix dWhh = FloatMatrix::Zero(Whh_.rows(), Whh_.cols());
    FloatMatrix dWxh = FloatMatrix::Zero(Wxh_.rows(), Wxh_.cols());
    FloatMatrix dWhz = FloatMatrix::Zero(Whz_.rows(), Whz_.cols());
    FloatMatrix dWxz = FloatMatrix::Zero(Wxz_.rows(), Wxz_.cols());
    FloatMatrix dWhr = FloatMatrix::Zero(Whr_.rows(), Whr_.cols());
    FloatMatrix dWxr = FloatMatrix::Zero(Wxr_.rows(), Wxr_.cols());
    FloatVector dby = FloatVector::Zero(Why_.rows());
    FloatVector dbh = FloatVector::Zero(Whh_.rows());
    FloatVector dbz = FloatVector::Zero(Whz_.rows());
    FloatVector dbr = FloatVector::Zero(Whr_.rows());
    FloatVector dhnext = FloatVector::Zero(Whh_.rows());

    FloatMatrix dx = FloatMatrix::Zero(GetInputSize(), input_sequence.size());

    for (int t = input_sequence.size() - 1; t >= 0; --t) {
      FloatVector dy = p_.col(t);
      dy[output_sequence[t]] -= 1.0; // Backprop into y (softmax grad).

      dWhy += dy * h_.col(t).transpose();
      dby += dy;

      FloatVector dh = Why_.transpose() * dy + dhnext; // Backprop into h.
      FloatVector dhu = z_.col(t).cwiseProduct(dh);
      FloatMatrix dhuraw;
      DerivateActivation(activation_function_, hu_.col(t), &dhuraw);
      dhuraw = dhuraw.cwiseProduct(dhu);
      FloatVector hprev;
      if (t == 0) {
        hprev = FloatVector::Zero(h_.rows());
      } else {
        hprev = h_.col(t-1);
      }

      FloatVector dq = Whh_.transpose() * dhuraw;
      FloatVector dz = (hu_.col(t) - hprev).cwiseProduct(dh);
      FloatVector dr = hprev.cwiseProduct(dq);
      FloatMatrix dzraw;
      DerivateActivation(ActivationFunctions::LOGISTIC, z_.col(t), &dzraw);
      dzraw = dzraw.cwiseProduct(dz);
      FloatMatrix drraw;
      DerivateActivation(ActivationFunctions::LOGISTIC, r_.col(t), &drraw);
      drraw = drraw.cwiseProduct(dr);

      dWxz += dzraw * x_.col(t).transpose();
      dbz += dzraw;
      dWxr += drraw * x_.col(t).transpose();
      dbr += drraw;
      dWxh += dhuraw * x_.col(t).transpose();
      dbh += dhuraw;

      dWhz += dzraw * hprev.transpose();
      dWhr += drraw * hprev.transpose();
      dWhh += dhuraw * (r_.col(t).cwiseProduct(hprev)).transpose();

      dhnext.noalias() = Whz_.transpose() * dzraw + Whr_.transpose() * drraw +
        r_.col(t).cwiseProduct(dq) + (1.0 - z_.col(t).array()).matrix().cwiseProduct(dh);

      dx.col(t) = Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
                  Wxh_.transpose() * dhuraw; // Backprop into x.
    }

    Why_ -= learning_rate * dWhy;
    by_ -= learning_rate * dby;
    Wxz_ -= learning_rate * dWxz;
    bz_ -= learning_rate * dbz;
    Wxr_ -= learning_rate * dWxr;
    br_ -= learning_rate * dbr;
    Wxh_ -= learning_rate * dWxh;
    bh_ -= learning_rate * dbh;
    Whz_ -= learning_rate * dWhz;
    Whr_ -= learning_rate * dWhr;
    Whh_ -= learning_rate * dWhh;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dhnext;
    }

    RunBackwardLookupLayer(input_sequence, dx, learning_rate);
  }

 protected:
  FloatMatrix Wxz_;
  FloatMatrix Whz_;
  FloatMatrix Wxr_;
  FloatMatrix Whr_;
  FloatVector bz_;
  FloatVector br_;

  FloatMatrix z_;
  FloatMatrix r_;
  FloatMatrix hu_;

};

class BiRNN_GRU : public RNN_GRU {
 public:
  BiRNN_GRU(Dictionary *dictionary,
            int window_size, int embedding_dimension,
            int affix_embedding_dimension,
            int hidden_size, int output_size) {
    dictionary_ = dictionary;
    activation_function_ = ActivationFunctions::LOGISTIC;
    window_size_ = window_size;
    embedding_dimension_ = embedding_dimension;
    affix_embedding_dimension_ = affix_embedding_dimension;
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    use_hidden_start_ = true;
  }
  ~BiRNN_GRU() {}

  void CollectAllParameters(std::vector<FloatMatrix*> *weights,
                            std::vector<FloatVector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    RNN_GRU::CollectAllParameters(weights, biases, weight_names, bias_names);

    Wxz_r_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Wlz_r_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    bz_r_ = FloatVector::Zero(hidden_size_, 1);

    Wxr_r_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Wlr_r_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    br_r_ = FloatVector::Zero(hidden_size_, 1);

    Wxl_ = FloatMatrix::Zero(hidden_size_, GetInputSize());
    Wll_ = FloatMatrix::Zero(hidden_size_, hidden_size_);
    Wly_ = FloatMatrix::Zero(output_size_, hidden_size_);
    bl_ = FloatVector::Zero(hidden_size_, 1);
    if (use_hidden_start_) {
      l0_ = FloatVector::Zero(hidden_size_);
    }

    weights->push_back(&Wxz_r_);
    weights->push_back(&Wlz_r_);
    weights->push_back(&Wxr_r_);
    weights->push_back(&Wlr_r_);

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);
    weights->push_back(&Wly_);

    biases->push_back(&bz_r_);
    biases->push_back(&br_r_);
    biases->push_back(&bl_);
    if (use_hidden_start_) {
      biases->push_back(&l0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxz_r");
    weight_names->push_back("Wlz");
    weight_names->push_back("Wxr_r");
    weight_names->push_back("Wlr");

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");
    weight_names->push_back("Wly");

    bias_names->push_back("bz_r");
    bias_names->push_back("br_r");
    bias_names->push_back("bl");
    if (use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void RunForwardPass(const std::vector<Input> &input_sequence) {
    RunForwardLookupLayer(input_sequence);

    int hidden_size = Whh_.rows();
    int output_size = Why_.rows();

    z_.setZero(hidden_size, input_sequence.size());
    r_.setZero(hidden_size, input_sequence.size());
    hu_.setZero(hidden_size, input_sequence.size());
    h_.setZero(hidden_size, input_sequence.size());
    FloatVector hprev = FloatVector::Zero(h_.rows());
    if (use_hidden_start_) hprev = h0_;
    for (int t = 0; t < input_sequence.size(); ++t) {
      FloatMatrix result;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxz_ * x_.col(t) + bz_ + Whz_ * hprev,
                         &result);
      z_.col(t) = result;

      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxr_ * x_.col(t) + br_ + Whr_ * hprev,
                         &result);
      r_.col(t) = result;

      EvaluateActivation(activation_function_,
                         Wxh_ * x_.col(t) + bh_ + Whh_ * r_.col(t).cwiseProduct(hprev),
                         &result);
      hu_.col(t) = result;

      //h_.col(t) = z_.col(t) * hu_.col(t) + (1.0 - z_.col(t)) * hprev;
      h_.col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;

      //FloatVector hraw = Wxh_ * x_.col(t) + bh_;
      //if (t > 0) hraw += Whh_ * h_.col(t-1);
      //FloatMatrix result;
      //EvaluateActivation(activation_function_, hraw, &result);
      //h_.col(t) = result;

      hprev = h_.col(t);
    }

    z_r_.setZero(hidden_size, input_sequence.size());
    r_r_.setZero(hidden_size, input_sequence.size());
    lu_.setZero(hidden_size, input_sequence.size());
    l_.setZero(hidden_size, input_sequence.size());
    FloatVector lnext = FloatVector::Zero(l_.rows());
    if (use_hidden_start_) lnext = l0_;
    for (int t = input_sequence.size() - 1; t >= 0; --t) {
      FloatMatrix result;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxz_r_ * x_.col(t) + bz_r_ + Wlz_r_ * lnext,
                         &result);
      z_r_.col(t) = result;

      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         Wxr_r_ * x_.col(t) + br_r_ + Wlr_r_ * lnext,
                         &result);
      r_r_.col(t) = result;

      EvaluateActivation(activation_function_,
                         Wxl_ * x_.col(t) + bl_ + Wll_ * r_r_.col(t).cwiseProduct(lnext),
                         &result);
      lu_.col(t) = result;

      l_.col(t) = z_r_.col(t).cwiseProduct(lu_.col(t) - lnext) + lnext;
      lnext = l_.col(t);
    }

    y_.setZero(output_size, input_sequence.size());
    p_.setZero(output_size, input_sequence.size());
    for (int t = 0; t < input_sequence.size(); ++t) {
      y_.col(t) = Why_ * h_.col(t) +  Wly_ * l_.col(t) + by_;
      double logsum = LogSumExp(y_.col(t));
      p_.col(t) = (y_.col(t).array() - logsum).exp();
    }
  }

  void RunBackwardPass(const std::vector<Input> &input_sequence,
                       const std::vector<int> &output_sequence,
                       double learning_rate) {
    FloatMatrix dWhy = FloatMatrix::Zero(Why_.rows(), Why_.cols());
    FloatMatrix dWhh = FloatMatrix::Zero(Whh_.rows(), Whh_.cols());
    FloatMatrix dWxh = FloatMatrix::Zero(Wxh_.rows(), Wxh_.cols());
    FloatMatrix dWhz = FloatMatrix::Zero(Whz_.rows(), Whz_.cols());
    FloatMatrix dWxz = FloatMatrix::Zero(Wxz_.rows(), Wxz_.cols());
    FloatMatrix dWhr = FloatMatrix::Zero(Whr_.rows(), Whr_.cols());
    FloatMatrix dWxr = FloatMatrix::Zero(Wxr_.rows(), Wxr_.cols());
    FloatVector dby = FloatVector::Zero(Why_.rows());
    FloatVector dbh = FloatVector::Zero(Whh_.rows());
    FloatVector dbz = FloatVector::Zero(Whz_.rows());
    FloatVector dbr = FloatVector::Zero(Whr_.rows());
    FloatVector dhnext = FloatVector::Zero(Whh_.rows());

    FloatMatrix dWly = FloatMatrix::Zero(Wly_.rows(), Wly_.cols());
    FloatMatrix dWll = FloatMatrix::Zero(Wll_.rows(), Wll_.cols());
    FloatMatrix dWxl = FloatMatrix::Zero(Wxl_.rows(), Wxl_.cols());
    FloatMatrix dWlz_r = FloatMatrix::Zero(Wlz_r_.rows(), Wlz_r_.cols());
    FloatMatrix dWxz_r = FloatMatrix::Zero(Wxz_r_.rows(), Wxz_r_.cols());
    FloatMatrix dWlr_r = FloatMatrix::Zero(Wlr_r_.rows(), Wlr_r_.cols());
    FloatMatrix dWxr_r = FloatMatrix::Zero(Wxr_r_.rows(), Wxr_r_.cols());
    FloatVector dbl = FloatVector::Zero(Wll_.rows());
    FloatVector dbz_r = FloatVector::Zero(Wlz_r_.rows());
    FloatVector dbr_r = FloatVector::Zero(Wlr_r_.rows());
    FloatVector dlprev = FloatVector::Zero(Wll_.rows());

    FloatMatrix dx = FloatMatrix::Zero(GetInputSize(), input_sequence.size());

    // TODO(atm): here.

    for (int t = 0; t < input_sequence.size(); ++t) {
      FloatVector dy = p_.col(t);
      dy[output_sequence[t]] -= 1.0; // Backprop into y (softmax grad).

      dWly += dy * l_.col(t).transpose();

      FloatVector dl = Wly_.transpose() * dy + dlprev; // Backprop into l.
      FloatVector dlu = z_r_.col(t).cwiseProduct(dl);
      FloatMatrix dluraw;
      DerivateActivation(activation_function_, lu_.col(t), &dluraw);
      dluraw = dluraw.cwiseProduct(dlu);
      FloatVector lnext;
      if (t == input_sequence.size() - 1) {
        lnext = FloatVector::Zero(l_.rows());
      } else {
        lnext = l_.col(t+1);
      }

      FloatVector dq_r = Wll_.transpose() * dluraw;
      FloatVector dz_r = (lu_.col(t) - lnext).cwiseProduct(dl);
      FloatVector dr_r = lnext.cwiseProduct(dq_r);
      FloatMatrix dzraw_r;
      DerivateActivation(ActivationFunctions::LOGISTIC, z_r_.col(t), &dzraw_r);
      dzraw_r = dzraw_r.cwiseProduct(dz_r);
      FloatMatrix drraw_r;
      DerivateActivation(ActivationFunctions::LOGISTIC, r_r_.col(t), &drraw_r);
      drraw_r = drraw_r.cwiseProduct(dr_r);

      dWxz_r += dzraw_r * x_.col(t).transpose();
      dbz_r += dzraw_r;
      dWxr_r += drraw_r * x_.col(t).transpose();
      dbr_r += drraw_r;
      dWxl += dluraw * x_.col(t).transpose();
      dbl += dluraw;

      dWlz_r += dzraw_r * lnext.transpose();
      dWlr_r += drraw_r * lnext.transpose();
      dWll += dluraw * (r_r_.col(t).cwiseProduct(lnext)).transpose();

      dlprev.noalias() = Wlz_r_.transpose() * dzraw_r + Wlr_r_.transpose() * drraw_r +
        r_r_.col(t).cwiseProduct(dq_r) + (1.0 - z_r_.col(t).array()).matrix().cwiseProduct(dl);

      dx.col(t) = Wxz_r_.transpose() * dzraw_r + Wxr_r_.transpose() * drraw_r +
                  Wxl_.transpose() * dluraw; // Backprop into x.
    }

    for (int t = input_sequence.size() - 1; t >= 0; --t) {
      FloatVector dy = p_.col(t);
      dy[output_sequence[t]] -= 1.0; // Backprop into y (softmax grad).

      dWhy += dy * h_.col(t).transpose();
      dby += dy;

      FloatVector dh = Why_.transpose() * dy + dhnext; // Backprop into h.
      FloatVector dhu = z_.col(t).cwiseProduct(dh);
      FloatMatrix dhuraw;
      DerivateActivation(activation_function_, hu_.col(t), &dhuraw);
      dhuraw = dhuraw.cwiseProduct(dhu);
      FloatVector hprev;
      if (t == 0) {
        hprev = FloatVector::Zero(h_.rows());
      } else {
        hprev = h_.col(t-1);
      }

      FloatVector dq = Whh_.transpose() * dhuraw;
      FloatVector dz = (hu_.col(t) - hprev).cwiseProduct(dh);
      FloatVector dr = hprev.cwiseProduct(dq);
      FloatMatrix dzraw;
      DerivateActivation(ActivationFunctions::LOGISTIC, z_.col(t), &dzraw);
      dzraw = dzraw.cwiseProduct(dz);
      FloatMatrix drraw;
      DerivateActivation(ActivationFunctions::LOGISTIC, r_.col(t), &drraw);
      drraw = drraw.cwiseProduct(dr);

      dWxz += dzraw * x_.col(t).transpose();
      dbz += dzraw;
      dWxr += drraw * x_.col(t).transpose();
      dbr += drraw;
      dWxh += dhuraw * x_.col(t).transpose();
      dbh += dhuraw;

      dWhz += dzraw * hprev.transpose();
      dWhr += drraw * hprev.transpose();
      dWhh += dhuraw * (r_.col(t).cwiseProduct(hprev)).transpose();

      dhnext.noalias() = Whz_.transpose() * dzraw + Whr_.transpose() * drraw +
        r_.col(t).cwiseProduct(dq) + (1.0 - z_.col(t).array()).matrix().cwiseProduct(dh);

      dx.col(t) += Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
                   Wxh_.transpose() * dhuraw; // Backprop into x.
    }

    Why_ -= learning_rate * dWhy;
    by_ -= learning_rate * dby;
    Wxz_ -= learning_rate * dWxz;
    bz_ -= learning_rate * dbz;
    Wxr_ -= learning_rate * dWxr;
    br_ -= learning_rate * dbr;
    Wxh_ -= learning_rate * dWxh;
    bh_ -= learning_rate * dbh;
    Whz_ -= learning_rate * dWhz;
    Whr_ -= learning_rate * dWhr;
    Whh_ -= learning_rate * dWhh;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dhnext;
    }

    Wly_ -= learning_rate * dWly;
    Wxz_r_ -= learning_rate * dWxz_r;
    bz_r_ -= learning_rate * dbz_r;
    Wxr_r_ -= learning_rate * dWxr_r;
    br_r_ -= learning_rate * dbr_r;
    Wxl_ -= learning_rate * dWxl;
    bl_ -= learning_rate * dbl;
    Wlz_r_ -= learning_rate * dWlz_r;
    Wlr_r_ -= learning_rate * dWlr_r;
    Wll_ -= learning_rate * dWll;
    if (use_hidden_start_) {
      l0_ -= learning_rate * dlprev;
    }

    RunBackwardLookupLayer(input_sequence, dx, learning_rate);
  }

 protected:
  FloatMatrix Wxl_;
  FloatMatrix Wll_;
  FloatMatrix Wly_;
  FloatMatrix Wxz_r_;
  FloatMatrix Wlz_r_;
  FloatMatrix Wxr_r_;
  FloatMatrix Wlr_r_;
  FloatVector bl_;
  FloatVector bz_r_;
  FloatVector br_r_;
  FloatVector l0_;

  FloatMatrix l_;
  FloatMatrix z_r_;
  FloatMatrix r_r_;
  FloatMatrix lu_;
};

#endif

#endif /* RNN_H_ */

