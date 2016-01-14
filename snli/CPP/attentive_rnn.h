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
    use_bidirectional_rnns_ = false;
    use_linear_layer_after_rnn_ = true;
    use_average_layer_ = true;
    input_size_ = hidden_size; // Size of the projected embedded words.
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    activation_function_ = ActivationFunctions::TANH; //LOGISTIC;
    lookup_layer_ = new LookupLayer<float>(dictionary->GetNumWords(),
                                           embedding_dimension);
    linear_layer_ = new LinearLayer<float>(embedding_dimension, input_size_);

    //rnn_layer_ = new RNNLayer(input_size_, hidden_size);
    int state_size;
    if (use_bidirectional_rnns_) {
      rnn_layer_ = new BiGRULayer<float>(input_size_, hidden_size);
      state_size = 2*hidden_size;
    } else {
      rnn_layer_ = new GRULayer<float>(input_size_, hidden_size);
      state_size = hidden_size;
    }

    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_ = new LinearLayer<float>(state_size, state_size);
    } else {
      linear_layer_after_rnn_ = NULL;
    }

    if (use_attention_) {
      attention_layer_ = new AttentionLayer<float>(state_size, state_size,
                                                   hidden_size,
                                                   sparse_attention_);
    } else {
      attention_layer_ = NULL;
    }

    if (use_average_layer_) {
      average_layer_ = new AverageLayer<float>;
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
    feedforward_layer_ = new FeedforwardLayer<float>(2*state_size,
                                                     hidden_size);
    output_layer_ = new SoftmaxOutputLayer<float>(hidden_size, output_size);
  }
  virtual ~RNN() {
    delete lookup_layer_;
    delete linear_layer_;
    delete rnn_layer_;
    delete linear_layer_after_rnn_;
    delete attention_layer_;
    delete average_layer_;
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

    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->CollectAllParameters(weights, biases,
                                                    weight_names,
                                                    bias_names);
    }

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
    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->InitializeParameters();
    }
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
      if (use_linear_layer_after_rnn_) {
        linear_layer_after_rnn_->InitializeADAM(beta1, beta2, epsilon);
      }
      if (use_attention_) {
        attention_layer_->InitializeADAM(beta1, beta2, epsilon);
      }
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
      os_attention_.open("sparse_attention_bi.txt", std::ifstream::out);
    } else {
      os_attention_.open("soft_attention_bi.txt", std::ifstream::out);
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

    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->SetNumInputs(1);
      linear_layer_after_rnn_->SetInput(0, rnn_layer_->GetOutput());
      linear_layer_after_rnn_->RunForward();
    }

    int t = input_sequence.size() - 1;

    int state_size;
    if (use_bidirectional_rnns_) {
      state_size = 2*hidden_size_;
    } else {
      state_size = hidden_size_;
    }

    if (use_attention_) {

      premise_extractor_layer_->SetNumInputs(1);
      if (use_linear_layer_after_rnn_) {
        premise_extractor_layer_->
          SetInput(0, linear_layer_after_rnn_->GetOutput());
      } else {
        premise_extractor_layer_->SetInput(0, rnn_layer_->GetOutput());
      }
      premise_extractor_layer_->DefineBlock(0, 0, state_size, separator);
      premise_extractor_layer_->RunForward();

      if (use_average_layer_) {
        average_layer_->SetNumInputs(1);
        if (use_linear_layer_after_rnn_) {
          average_layer_->SetInput(0, linear_layer_after_rnn_->GetOutput());
        } else {
          average_layer_->SetInput(0, rnn_layer_->GetOutput());
        }
        average_layer_->RunForward();
      } else {
        hypothesis_selector_layer_->SetNumInputs(1);
        if (use_linear_layer_after_rnn_) {
          hypothesis_selector_layer_->
            SetInput(0, linear_layer_after_rnn_->GetOutput());
        } else {
          hypothesis_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
        }
        if (use_bidirectional_rnns_) {
          hypothesis_selector_layer_->DefineBlock(0, (t+separator)/2, state_size, 1); // CHANGE THIS!!!
        } else {
          hypothesis_selector_layer_->DefineBlock(0, t, state_size, 1);
        }
        hypothesis_selector_layer_->RunForward();
      }

      attention_layer_->SetNumInputs(2);
      attention_layer_->SetInput(0, premise_extractor_layer_->GetOutput());
      if (use_average_layer_) {
        attention_layer_->SetInput(1, average_layer_->GetOutput());
      } else {
        attention_layer_->SetInput(1, hypothesis_selector_layer_->GetOutput());
      }
      attention_layer_->RunForward();

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetInput(0, attention_layer_->GetOutput());
      if (use_average_layer_) {
        concatenator_layer_->SetInput(1, average_layer_->GetOutput());
      } else {
        concatenator_layer_->
          SetInput(1, hypothesis_selector_layer_->GetOutput());
      }
      concatenator_layer_->RunForward();

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetInput(0, concatenator_layer_->GetOutput());
      feedforward_layer_->RunForward();

    } else {

      premise_selector_layer_->SetNumInputs(1);
      if (use_linear_layer_after_rnn_) {
        premise_selector_layer_->
          SetInput(0, linear_layer_after_rnn_->GetOutput());
      } else {
        premise_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
      }
      premise_selector_layer_->DefineBlock(0, separator, state_size, 1);
      premise_selector_layer_->RunForward();

      hypothesis_selector_layer_->SetNumInputs(1);
      if (use_linear_layer_after_rnn_) {
        hypothesis_selector_layer_->
          SetInput(0, linear_layer_after_rnn_->GetOutput());
      } else {
        hypothesis_selector_layer_->SetInput(0, rnn_layer_->GetOutput());
      }
      hypothesis_selector_layer_->DefineBlock(0, t, state_size, 1);
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

    int state_size;
    if (use_bidirectional_rnns_) {
      state_size = 2*hidden_size_;
    } else {
      state_size = hidden_size_;
    }

    // Reset parameter gradients.
    output_layer_->ResetGradients();
    feedforward_layer_->ResetGradients();
    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->ResetGradients();
    }
    rnn_layer_->ResetGradients();
    linear_layer_->ResetGradients();
    lookup_layer_->ResetGradients();
    if (use_attention_) {
      attention_layer_->ResetGradients();
    }

    // Reset variable derivatives.
    feedforward_layer_->GetOutputDerivative()->setZero(hidden_size_, 1);
    concatenator_layer_->GetOutputDerivative()->setZero(2*state_size, 1);
    if (use_attention_) {
      attention_layer_->GetOutputDerivative()->setZero(state_size, 1);
      premise_extractor_layer_->GetOutputDerivative()->setZero(state_size,
                                                               separator);
      hypothesis_selector_layer_->GetOutputDerivative()->setZero(state_size,
                                                                 1);
    } else {
      premise_selector_layer_->GetOutputDerivative()->setZero(state_size, 1);
      hypothesis_selector_layer_->GetOutputDerivative()->setZero(state_size,
                                                                 1);
    }
    if (use_average_layer_) {
      average_layer_->GetOutputDerivative()->setZero(state_size, 1);
    }
    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->GetOutputDerivative()->
        setZero(state_size, input_sequence.size());
    }
    rnn_layer_->GetOutputDerivative()->setZero(state_size,
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
      if (use_average_layer_) {
        concatenator_layer_->
          SetInputDerivative(1, average_layer_->GetOutputDerivative());
      } else {
        concatenator_layer_->
          SetInputDerivative(1,
                             hypothesis_selector_layer_->GetOutputDerivative());
      }
      concatenator_layer_->RunBackward();

      attention_layer_->
        SetInputDerivative(0, premise_extractor_layer_->GetOutputDerivative());
      attention_layer_->
        SetInputDerivative(1,
                           hypothesis_selector_layer_->GetOutputDerivative());
      attention_layer_->RunBackward();

      if (use_linear_layer_after_rnn_) {
        premise_extractor_layer_->
          SetInputDerivative(0, linear_layer_after_rnn_->GetOutputDerivative());
      } else {
        premise_extractor_layer_->
          SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      }
      premise_extractor_layer_->DefineBlock(0, 0, state_size, separator);
      premise_extractor_layer_->RunBackward();

      if (use_average_layer_) {
        if (use_linear_layer_after_rnn_) {
          average_layer_->
            SetInputDerivative(0, linear_layer_after_rnn_->GetOutputDerivative());
        } else {
          average_layer_->
            SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
        }
        average_layer_->RunBackward();
      } else {
        if (use_linear_layer_after_rnn_) {
          hypothesis_selector_layer_->
            SetInputDerivative(0, linear_layer_after_rnn_->GetOutputDerivative());
        } else {
          hypothesis_selector_layer_->
            SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
        }
        if (use_bidirectional_rnns_) {
          hypothesis_selector_layer_->DefineBlock(0, (t+separator)/2, state_size, 1); // CHANGE THIS!!!
        } else {
          hypothesis_selector_layer_->DefineBlock(0, t, state_size, 1);
        }
        hypothesis_selector_layer_->RunBackward();
      }

    } else {

      concatenator_layer_->
        SetInputDerivative(0, premise_selector_layer_->GetOutputDerivative());
      concatenator_layer_->
        SetInputDerivative(1,
                           hypothesis_selector_layer_->GetOutputDerivative());
      concatenator_layer_->RunBackward();

      if (use_linear_layer_after_rnn_) {
        premise_selector_layer_->
          SetInputDerivative(0, linear_layer_after_rnn_->GetOutputDerivative());
      } else {
        premise_selector_layer_->
          SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      }
      premise_selector_layer_->DefineBlock(0, separator, state_size, 1);
      premise_selector_layer_->RunBackward();

      if (use_linear_layer_after_rnn_) {
        hypothesis_selector_layer_->
          SetInputDerivative(0, linear_layer_after_rnn_->GetOutputDerivative());
      } else {
        hypothesis_selector_layer_->
          SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      }
      hypothesis_selector_layer_->DefineBlock(0, t, state_size, 1);
      hypothesis_selector_layer_->RunBackward();

    }

    if (use_linear_layer_after_rnn_) {
      linear_layer_after_rnn_->
        SetInputDerivative(0, rnn_layer_->GetOutputDerivative());
      linear_layer_after_rnn_->RunBackward();
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
      if (use_linear_layer_after_rnn_) {
        linear_layer_after_rnn_->UpdateParametersADAM(learning_rate,
                                                      regularization_constant);
      }
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
      if (use_linear_layer_after_rnn_) {
        linear_layer_after_rnn_->UpdateParameters(learning_rate,
                                                  regularization_constant);
      }
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
  LinearLayer<float> *linear_layer_after_rnn_;
  AttentionLayer<float> *attention_layer_;
  FeedforwardLayer<float> *feedforward_layer_;
  AverageLayer<float> *average_layer_;
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
  bool use_bidirectional_rnns_;
  bool use_linear_layer_after_rnn_;
  bool use_average_layer_;
  bool use_ADAM_;
  bool write_attention_probabilities_;
  std::ofstream os_attention_;

  //FloatMatrix x_;
  //FloatMatrix h_;
};

#endif /* RNN_H_ */

