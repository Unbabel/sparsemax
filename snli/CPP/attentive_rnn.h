#ifndef ATTENTIVE_RNN_H_
#define ATTENTIVE_RNN_H_

#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "snli_data.h"
#include "layer.h"
#include "neural_network.h"

template<typename Real> class AttentiveRNN : public NeuralNetwork<Real> {
 public:
  AttentiveRNN() {}
  AttentiveRNN(Dictionary *dictionary,
               int embedding_dimension,
               int hidden_size,
               int output_size,
               bool use_attention,
               int attention_type) : dictionary_(dictionary) {
    write_attention_probabilities_ = false;
    use_ADAM_ = true; //false; //true;
    use_attention_ = use_attention;
    attention_type_ = attention_type;
    use_lstms_ = false; //true; //false;
    use_bidirectional_rnns_ = false;
    apply_dropout_ = true; // false;
    dropout_probability_ = 0.1;
    test_ = false;
    input_size_ = hidden_size; // Size of the projected embedded words.
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    embedding_size_ = embedding_dimension;

    CreateNetwork();
  }

  virtual ~AttentiveRNN() {
    delete lookup_layer_;
    delete linear_layer_;
    delete rnn_layer_;
    delete hypothesis_rnn_layer_;
    delete attention_layer_;
    delete hypothesis_extractor_layer_;
    delete premise_extractor_layer_;
    delete hypothesis_selector_layer_;
    delete premise_selector_layer_;
    delete concatenator_layer_;
    delete feedforward_layer_;
    delete output_layer_;
  }

  void CreateNetwork() {
    // Create the layers.
    lookup_layer_ = new LookupLayer<Real>(dictionary_->GetNumWords(),
                                          embedding_size_);
    NeuralNetwork<Real>::AddLayer(lookup_layer_);

    linear_layer_ = new LinearLayer<Real>(embedding_size_, input_size_);
    NeuralNetwork<Real>::AddLayer(linear_layer_);

    int state_size;
    if (use_bidirectional_rnns_) {
      rnn_layer_ = new BiGRULayer<Real>(input_size_, hidden_size_);
      NeuralNetwork<Real>::AddLayer(rnn_layer_);
      hypothesis_rnn_layer_ = new BiGRULayer<Real>(input_size_, hidden_size_);
      NeuralNetwork<Real>::AddLayer(hypothesis_rnn_layer_);
      state_size = 2*hidden_size_;
    } else {
      if (use_lstms_) {
        rnn_layer_ = new LSTMLayer<Real>(input_size_, hidden_size_);
        NeuralNetwork<Real>::AddLayer(rnn_layer_);
        hypothesis_rnn_layer_ = new LSTMLayer<Real>(input_size_,
                                                    hidden_size_);
        NeuralNetwork<Real>::AddLayer(hypothesis_rnn_layer_);
      } else {
        rnn_layer_ = new GRULayer<Real>(input_size_, hidden_size_);
        NeuralNetwork<Real>::AddLayer(rnn_layer_);
        hypothesis_rnn_layer_ = new GRULayer<Real>(input_size_, hidden_size_);
        NeuralNetwork<Real>::AddLayer(hypothesis_rnn_layer_);
      }
      state_size = hidden_size_;
    }

    attention_layer_ = NULL;
    premise_selector_layer_ = NULL;
    if (use_attention_) {
      attention_layer_ = new AttentionLayer<Real>(state_size, state_size,
                                                  hidden_size_,
                                                  attention_type_);
      NeuralNetwork<Real>::AddLayer(attention_layer_);
    }

    premise_extractor_layer_ = new SelectorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(premise_extractor_layer_);

    hypothesis_extractor_layer_ = new SelectorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(hypothesis_extractor_layer_);

    premise_selector_layer_ = new SelectorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(premise_selector_layer_);

    hypothesis_selector_layer_ = new SelectorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(hypothesis_selector_layer_);

    concatenator_layer_ = new ConcatenatorLayer<Real>;
    NeuralNetwork<Real>::AddLayer(concatenator_layer_);

    feedforward_layer_ = new FeedforwardLayer<Real>(2*state_size,
                                                    hidden_size_);
    NeuralNetwork<Real>::AddLayer(feedforward_layer_);

    output_layer_ = new SoftmaxOutputLayer<Real>(hidden_size_, output_size_);
    NeuralNetwork<Real>::AddLayer(output_layer_);

    // Connect the layers.
    lookup_layer_->SetNumInputs(1);
    lookup_layer_->SetNumOutputs(1);

    linear_layer_->SetNumInputs(1);
    linear_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(lookup_layer_, linear_layer_, 0, 0);

    premise_extractor_layer_->SetNumInputs(1);
    premise_extractor_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(linear_layer_, premise_extractor_layer_, 0, 0);

    hypothesis_extractor_layer_->SetNumInputs(1);
    hypothesis_extractor_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(linear_layer_, hypothesis_extractor_layer_, 0, 0);

    // Premise RNN.
    rnn_layer_->SetNumInputs(1);
    if (use_lstms_) {
      rnn_layer_->SetNumOutputs(2); // Cell states are an additional output.
    } else {
      rnn_layer_->SetNumOutputs(1);
    }
    NeuralNetwork<Real>::ConnectLayers(premise_extractor_layer_, rnn_layer_, 0, 0);

    // Selector of the last state of the RNN layer.
    premise_selector_layer_->SetNumInputs(1);
    premise_selector_layer_->SetNumOutputs(1);
    if (use_lstms_) {
      // Cell states.
      NeuralNetwork<Real>::ConnectLayers(rnn_layer_, premise_selector_layer_, 1, 0);
    } else {
      // Hidden states.
      NeuralNetwork<Real>::ConnectLayers(rnn_layer_, premise_selector_layer_, 0, 0);
    }

    hypothesis_rnn_layer_->set_use_control(true);
    hypothesis_rnn_layer_->SetNumInputs(2);
    if (use_lstms_) {
      hypothesis_rnn_layer_->SetNumOutputs(2);
    } else {
      hypothesis_rnn_layer_->SetNumOutputs(1);
    }
    NeuralNetwork<Real>::ConnectLayers(hypothesis_extractor_layer_, hypothesis_rnn_layer_,
                  0, 0);
    NeuralNetwork<Real>::ConnectLayers(premise_selector_layer_, hypothesis_rnn_layer_,
                  0, 1);

    if (use_attention_) {
      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(hypothesis_rnn_layer_, hypothesis_selector_layer_,
                    0, 0);

      attention_layer_->SetNumInputs(2);
      attention_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(rnn_layer_, attention_layer_, 0, 0);
      NeuralNetwork<Real>::ConnectLayers(hypothesis_selector_layer_, attention_layer_,
                    0, 1);

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(attention_layer_, concatenator_layer_, 0, 0);
      NeuralNetwork<Real>::ConnectLayers(hypothesis_selector_layer_, concatenator_layer_,
                    0, 1);

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(concatenator_layer_, feedforward_layer_, 0, 0);
    } else {
      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(hypothesis_rnn_layer_, hypothesis_selector_layer_,
                    0, 0);

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetNumOutputs(1);
      // Note: this will be weird if using lstms, since there premise_selector
      // is selecting the last cell state.
      NeuralNetwork<Real>::ConnectLayers(premise_selector_layer_, concatenator_layer_,
                    0, 0);
      NeuralNetwork<Real>::ConnectLayers(hypothesis_selector_layer_, concatenator_layer_,
                    0, 1);

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetNumOutputs(1);
      NeuralNetwork<Real>::ConnectLayers(concatenator_layer_, feedforward_layer_, 0, 0);
    }

    output_layer_->SetNumInputs(1);
    output_layer_->SetNumOutputs(1);
    NeuralNetwork<Real>::ConnectLayers(feedforward_layer_, output_layer_, 0, 0);

    // Sort layers by topological order.
    NeuralNetwork<Real>::SortLayersByTopologicalOrder();
  }

  void SetModelPrefix(const std::string &model_prefix) {
    model_prefix_ = model_prefix;
  }

  int GetEmbeddingSize() { return lookup_layer_->embedding_dimension(); }

  int GetInputSize() { return linear_layer_->output_size(); }

  void InitializeParameters() {
    srand(1234);

    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->InitializeParameters();
    }

    if (use_ADAM_) {
      double beta1 = 0.9;
      double beta2 = 0.999;
      double epsilon = 1e-8;
      for (int k = 0; k < layers.size(); ++k) {
        layers[k]->InitializeADAM(beta1, beta2, epsilon);
      }
    }
  }

  void LoadModel(const std::string &prefix) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      std::ostringstream ss;
      ss << "Layer" << k << "_" << layers[k]->name() + "_";
      layers[k]->LoadParameters(prefix + ss.str(), true);
      if (use_ADAM_) {
        layers[k]->LoadADAMParameters(prefix + ss.str());
      }
    }
  }

  void SaveModel(const std::string &prefix) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      std::ostringstream ss;
      ss << "Layer" << k << "_" << layers[k]->name() + "_";
      layers[k]->SaveParameters(prefix + ss.str(), true);
      if (use_ADAM_) {
        layers[k]->SaveADAMParameters(prefix + ss.str());
      }
    }
  }

  void SetFixedEmbeddings(const Matrix<Real> &fixed_embeddings) {
    lookup_layer_->SetFixedEmbeddings(fixed_embeddings);
  }

  void Train(const std::vector<std::vector<Input> > &input_sequences,
             const std::vector<int> &output_labels,
             const std::vector<std::vector<Input> > &input_sequences_dev,
             const std::vector<int> &output_labels_dev,
             const std::vector<std::vector<Input> > &input_sequences_test,
             const std::vector<int> &output_labels_test,
             int warm_start_on_epoch, // 0 for no warm-starting.
             int num_epochs,
             int batch_size,
             double learning_rate,
             double regularization_constant) {

    if (warm_start_on_epoch == 0) {
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

      SaveModel(model_prefix_ + "Epoch0_");
    } else {
      std::cout << "Warm-starting on epoch " << warm_start_on_epoch
                << "..." << std::endl;
      std::ostringstream ss;
      ss << "Epoch" << warm_start_on_epoch << "_";
      LoadModel(model_prefix_ + ss.str());
    }

    for (int epoch = warm_start_on_epoch; epoch < num_epochs; ++epoch) {
      std::ostringstream ss;
      TrainEpoch(input_sequences, output_labels,
                 input_sequences_dev, output_labels_dev,
                 input_sequences_test, output_labels_test,
                 epoch, batch_size, learning_rate, regularization_constant);
      ss << "Epoch" << epoch+1 << "_";
      SaveModel(model_prefix_ + ss.str());
    }

    SaveModel(model_prefix_);
  }

  void TrainEpoch(const std::vector<std::vector<Input> > &input_sequences,
                  const std::vector<int> &output_labels,
                  const std::vector<std::vector<Input> > &input_sequences_dev,
                  const std::vector<int> &output_labels_dev,
                  const std::vector<std::vector<Input> > &input_sequences_test,
                  const std::vector<int> &output_labels_test,
                  int epoch,
                  int batch_size,
                  double learning_rate,
                  double regularization_constant) {
    timeval start, end;
    gettimeofday(&start, NULL);
    double total_loss = 0.0;
    double accuracy = 0.0;
    int num_sentences = input_sequences.size();
    int actual_batch_size = 0;
    for (int i = 0; i < input_sequences.size(); ++i) {
      if (i % batch_size == 0) {
        ResetParameterGradients();
        actual_batch_size = 0;
      }
      RunForwardPass(input_sequences[i]);
      Vector<Real> p = output_layer_->GetOutput(0);
      double loss = -log(p(output_labels[i]));
      int prediction;
      p.maxCoeff(&prediction);
      if (prediction == output_labels[i]) {
        accuracy += 1.0;
      }
      total_loss += loss;
      RunBackwardPass(input_sequences[i], output_labels[i], learning_rate,
                      regularization_constant);
      ++actual_batch_size;
      if (((i+1) % batch_size == 0) || (i == input_sequences.size()-1)) {
        UpdateParameters(actual_batch_size, learning_rate,
                         regularization_constant);
      }
    }
    accuracy /= num_sentences;
    total_loss /= num_sentences;
    double total_reg = 0.0;
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      total_reg += 0.5 * regularization_constant *
        layers[k]->ComputeSquaredNormOfParameters();
    }

    write_attention_probabilities_ = true;
    if (attention_type_ == AttentionTypes::SPARSEMAX) {
      os_attention_.open("sparse_attention.txt", std::ifstream::out);
    } else if (attention_type_ == AttentionTypes::SOFTMAX) {
      os_attention_.open("soft_attention.txt", std::ifstream::out);
    } else { // LOGISTIC.
      os_attention_.open("logistic_attention.txt", std::ifstream::out);
    }

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

    write_attention_probabilities_ = false;

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

    os_attention_.flush();
    os_attention_.clear();
    os_attention_.close();

    gettimeofday(&end, NULL);
    std::cout << "Epoch: " << epoch+1
              << " Total loss: " << total_loss
              << " Total reg: " << total_reg
              << " Total loss+reg: " << total_loss + total_reg
              << " Accuracy train: " << accuracy
              << " Accuracy dev: " << accuracy_dev
              << " Accuracy test: " << accuracy_test
              << " Time: " << diff_ms(end,start)
              << std::endl;
  }

  void Test(const std::vector<std::vector<Input> > &input_sequences_dev,
            const std::vector<int> &output_labels_dev,
            const std::vector<std::vector<Input> > &input_sequences_test,
            const std::vector<int> &output_labels_test) {
    timeval start, end;
    gettimeofday(&start, NULL);

    write_attention_probabilities_ = true;
    if (attention_type_ == AttentionTypes::SPARSEMAX) {
      os_attention_.open("sparse_attention_test.txt", std::ifstream::out);
    } else if (attention_type_ == AttentionTypes::SOFTMAX) {
      os_attention_.open("soft_attention_test.txt", std::ifstream::out);
    } else { // LOGISTIC.
      os_attention_.open("logistic_attention_test.txt", std::ifstream::out);
    }

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

    write_attention_probabilities_ = false;

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

    os_attention_.flush();
    os_attention_.clear();
    os_attention_.close();

    gettimeofday(&end, NULL);
    std::cout << " Accuracy dev: " << accuracy_dev
              << " Accuracy test: " << accuracy_test
              << " Time: " << diff_ms(end,start)
              << std::endl;
  }

  void Run(const std::vector<Input> &input_sequence,
           int *predicted_label) {
    bool apply_dropout = apply_dropout_;
    test_ = true;
    //apply_dropout_ = false; // TODO: Remove this line to have correct dropout at test time.
    RunForwardPass(input_sequence);
    test_ = false;
    apply_dropout_ = apply_dropout;
    int prediction;
    Vector<Real> p = output_layer_->GetOutput(0);
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
    int t = input_sequence.size() - 1;

    lookup_layer_->set_input_sequence(input_sequence);
    premise_extractor_layer_->DefineBlock(0, 0, hidden_size_, separator);
    hypothesis_extractor_layer_->DefineBlock(0, separator,
                                             hidden_size_, t-separator+1);

    int state_size;
    if (use_bidirectional_rnns_) {
      state_size = 2*hidden_size_;
    } else {
      state_size = hidden_size_;
    }

    // Selector of the last state of the RNN layer.
    premise_selector_layer_->DefineBlock(0, separator-1, state_size, 1);
    if (use_attention_) {
      if (use_bidirectional_rnns_) {
        hypothesis_selector_layer_->
          DefineBlock(0, (t-separator)/2, state_size, 1); // CHANGE THIS!!!
      } else {
        hypothesis_selector_layer_->DefineBlock(0, t-separator, state_size, 1);
      }
    } else {
      hypothesis_selector_layer_->DefineBlock(0, t-separator, state_size, 1);
    }

    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->RunForward();
      if (apply_dropout_) {
        if (layers[k] == lookup_layer_) {
          if (test_) {
            lookup_layer_->ScaleOutput(0, 1.0 - dropout_probability_);
          } else {
            lookup_layer_->ApplyDropout(0, dropout_probability_);
          }
        } else if (layers[k] == hypothesis_rnn_layer_) {
          if (test_) {
            hypothesis_selector_layer_->ScaleOutput(0,
                                                    1.0 - dropout_probability_);
          } else {
            hypothesis_selector_layer_->ApplyDropout(0, dropout_probability_);
          }
        }
      }
    }

    if (use_attention_ && write_attention_probabilities_) {
      int prediction;
      Vector<Real> p = output_layer_->GetOutput(0);
      p.maxCoeff(&prediction);
      os_attention_ << dictionary_->GetLabelName(prediction) << "\t"
                    << attention_layer_->GetAttentionProbabilities().transpose()
                    << std::endl;
    }
  }

  void RunBackwardPass(const std::vector<Input> &input_sequence,
                       int output_label,
                       double learning_rate,
                       double regularization_constant) {
    // Reset variable derivatives.
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      if (layers[k] == output_layer_) continue;
      layers[k]->ResetOutputDerivatives();
    }

    // Backprop.
    output_layer_->set_output_label(output_label);
    for (int k = layers.size() - 1; k >= 0; --k) {
      layers[k]->RunBackward();
    }
  }

  void ResetParameterGradients() {
    // Reset parameter gradients.
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->ResetGradients();
    }
  }

  void UpdateParameters(int batch_size,
                        double learning_rate,
                        double regularization_constant) {
    const std::vector<Layer<Real>*> &layers = NeuralNetwork<Real>::GetLayers();
    for (int k = layers.size() - 1; k >= 0; --k) {
      // Update parameters.
      if (use_ADAM_) {
        layers[k]->UpdateParametersADAM(batch_size,
                                        learning_rate,
                                        regularization_constant);
      } else {
        layers[k]->UpdateParameters(batch_size,
                                    learning_rate,
                                    regularization_constant);
      }
    }
  }

 protected:
  Dictionary *dictionary_;
  int activation_function_;
  LookupLayer<Real> *lookup_layer_;
  LinearLayer<Real> *linear_layer_;
  RNNLayer<Real> *rnn_layer_;
  RNNLayer<Real> *hypothesis_rnn_layer_;
  AttentionLayer<Real> *attention_layer_;
  FeedforwardLayer<Real> *feedforward_layer_;
  SelectorLayer<Real> *hypothesis_selector_layer_;
  SelectorLayer<Real> *premise_selector_layer_;
  SelectorLayer<Real> *hypothesis_extractor_layer_;
  SelectorLayer<Real> *premise_extractor_layer_;
  ConcatenatorLayer<Real> *concatenator_layer_;
  SoftmaxOutputLayer<Real> *output_layer_;
  int embedding_size_;
  int input_size_;
  int hidden_size_;
  int output_size_;
  bool use_attention_;
  int attention_type_;
  bool use_lstms_;
  bool use_bidirectional_rnns_;
  bool apply_dropout_;
  double dropout_probability_;
  bool use_ADAM_;
  bool write_attention_probabilities_;
  bool test_;
  std::string model_prefix_;
  std::ofstream os_attention_;
};

#endif /* ATTENTIVE_RNN_H_ */

