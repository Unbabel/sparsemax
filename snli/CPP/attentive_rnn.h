#ifndef RNN_H_
#define RNN_H_

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

//template<typename Real>
//using Arc = std::pair<Layer<Real>*, Layer<Real>*>;
//arcs_.push_back(Arc<Real>(origin, target));
//std::vector<Arc<Real> > arcs_;

#include <map>
#include <set>
#include <queue>

template<typename Real> class NeuralNetwork {
 public:
  NeuralNetwork() {}
  virtual ~NeuralNetwork() {}

  const std::vector<Layer<Real>*> &GetLayers() { return layers_; }

  void AddLayer(Layer<Real> *layer) {
    layers_.push_back(layer);
    children_layers_[layer] = std::set<Layer<Real>*>();
    parent_layers_[layer] = std::set<Layer<Real>*>();
  }

  void ConnectLayers(Layer<Real> *origin, Layer<Real> *target,
                     int origin_output, int target_input) {
    target->SetInput(target_input, origin->GetOutput(origin_output));
    target->SetInputDerivative(target_input, origin->
                               GetMutableOutputDerivative(origin_output));
    const std::set<Layer<Real> *> &children = children_layers_[target];
    for (auto it = children.begin(); it != children.end(); ++it) {
      if (*it == origin) return;
    }
    children_layers_[target].insert(origin);
    parent_layers_[origin].insert(target);
  }

  void SortLayersByTopologicalOrder() {
    // Use Kahn's algorithm.
    layers_.clear();
    std::queue<Layer<Real>*> roots;
    auto active_children = children_layers_;
    for (auto it = children_layers_.begin();
         it != children_layers_.end();
         ++it) {
      if (it->second.empty()) {
        roots.push(it->first);
        active_children.erase(it->first);
      }
    }
    while (!roots.empty()) {
      Layer<Real> *layer = roots.front();
      roots.pop();
      layers_.push_back(layer);
      const std::set<Layer<Real> *> &parents = parent_layers_[layer];
      for (auto it = parents.begin(); it != parents.end(); ++it) {
        active_children[*it].erase(layer);
        if (active_children[*it].empty()) {
          roots.push(*it);
          active_children.erase(*it);
        }
      }
    }
    // If the graph is non-empty, then there is a cycle.
    assert(active_children.empty());

    for (int k = 0; k < layers_.size(); ++k) {
      std::cout << layers_[k]->name() << std::endl;
    }
  }

 protected:
  std::vector<Layer<Real>*> layers_;
  std::map<Layer<Real>*, std::set<Layer<Real>*> > children_layers_;
  std::map<Layer<Real>*, std::set<Layer<Real>*> > parent_layers_;
};

#if 0
typedef NeuralNetwork<float> FloatNeuralNetwork;
typedef Layer<float> FloatLayer;
typedef LookupLayer<float> FloatLookupLayer;
typedef RNNLayer<float> FloatRNNLayer;
typedef GRULayer<float> FloatGRULayer;
typedef BiGRULayer<float> FloatBiGRULayer;
typedef LSTMLayer<float> FloatLSTMLayer;
typedef LinearLayer<float> FloatLinearLayer;
typedef AttentionLayer<float> FloatAttentionLayer;
typedef FeedforwardLayer<float> FloatFeedforwardLayer;
typedef AverageLayer<float> FloatAverageLayer;
typedef SelectorLayer<float> FloatSelectorLayer;
typedef ConcatenatorLayer<float> FloatConcatenatorLayer;
typedef SoftmaxOutputLayer<float> FloatSoftmaxOutputLayer;
#else
typedef NeuralNetwork<double> FloatNeuralNetwork;
typedef Layer<double> FloatLayer;
typedef LookupLayer<double> FloatLookupLayer;
typedef RNNLayer<double> FloatRNNLayer;
typedef GRULayer<double> FloatGRULayer;
typedef BiGRULayer<double> FloatBiGRULayer;
typedef LSTMLayer<double> FloatLSTMLayer;
typedef LinearLayer<double> FloatLinearLayer;
typedef AttentionLayer<double> FloatAttentionLayer;
typedef FeedforwardLayer<double> FloatFeedforwardLayer;
typedef AverageLayer<double> FloatAverageLayer;
typedef SelectorLayer<double> FloatSelectorLayer;
typedef ConcatenatorLayer<double> FloatConcatenatorLayer;
typedef SoftmaxOutputLayer<double> FloatSoftmaxOutputLayer;
#endif

class RNN {
 public:
  RNN() {}
  RNN(Dictionary *dictionary,
      int embedding_dimension,
      int hidden_size,
      int output_size,
      bool use_attention,
      int attention_type) : dictionary_(dictionary) {
    write_attention_probabilities_ = false;
    use_ADAM_ = true; //false; //true;
    use_attention_ = use_attention;
    attention_type_ = attention_type;
    use_separate_rnns_ = true; // false;
    connect_rnns_ = true; // false;
    use_lstms_ = false; //true; //false;
    use_bidirectional_rnns_ = false;
    use_linear_layer_after_rnn_ = false; //true;
    use_average_layer_ = false; //true;
    apply_dropout_ = true; // false;
    dropout_probability_ = 0.1;
    test_ = false;
    input_size_ = hidden_size; // Size of the projected embedded words.
    hidden_size_ = hidden_size;
    output_size_ = output_size;
    embedding_size_ = embedding_dimension;
    activation_function_ = ActivationFunctions::TANH; //LOGISTIC;

    CreateNetwork();
  }

  virtual ~RNN() {
    delete lookup_layer_;
    delete linear_layer_;
    delete rnn_layer_;
    delete hypothesis_rnn_layer_;
    delete linear_layer_after_rnn_;
    delete attention_layer_;
    delete average_layer_;
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
    lookup_layer_ = new FloatLookupLayer(dictionary_->GetNumWords(),
                                         embedding_size_);
    network_.AddLayer(lookup_layer_);

    linear_layer_ = new FloatLinearLayer(embedding_size_, input_size_);
    network_.AddLayer(linear_layer_);

    int state_size;
    if (use_bidirectional_rnns_) {
      rnn_layer_ = new FloatBiGRULayer(input_size_, hidden_size_);
      network_.AddLayer(rnn_layer_);
      if (use_separate_rnns_) {
        hypothesis_rnn_layer_ = new FloatBiGRULayer(input_size_, hidden_size_);
        network_.AddLayer(hypothesis_rnn_layer_);
      }
      state_size = 2*hidden_size_;
    } else {
      if (use_lstms_) {
        rnn_layer_ = new FloatLSTMLayer(input_size_, hidden_size_);
        network_.AddLayer(rnn_layer_);
        if (use_separate_rnns_) {
          hypothesis_rnn_layer_ = new FloatLSTMLayer(input_size_,
                                                     hidden_size_);
          network_.AddLayer(hypothesis_rnn_layer_);
        }
      } else {
        rnn_layer_ = new FloatGRULayer(input_size_, hidden_size_);
        network_.AddLayer(rnn_layer_);
        if (use_separate_rnns_) {
          hypothesis_rnn_layer_ = new FloatGRULayer(input_size_, hidden_size_);
          network_.AddLayer(hypothesis_rnn_layer_);
        }
      }
      state_size = hidden_size_;
    }

    linear_layer_after_rnn_ = NULL;
    attention_layer_ = NULL;
    premise_selector_layer_ = NULL;
    if (use_attention_) {
      attention_layer_ = new FloatAttentionLayer(state_size, state_size,
                                                 hidden_size_,
                                                 attention_type_);
      network_.AddLayer(attention_layer_);
    }

    premise_extractor_layer_ = new FloatSelectorLayer;
    network_.AddLayer(premise_extractor_layer_);

    hypothesis_extractor_layer_ = new FloatSelectorLayer;
    network_.AddLayer(hypothesis_extractor_layer_);

    hypothesis_selector_layer_ = new FloatSelectorLayer;
    network_.AddLayer(hypothesis_selector_layer_);

    if (!use_attention_ || connect_rnns_) {
      premise_selector_layer_ = new FloatSelectorLayer;
      network_.AddLayer(premise_selector_layer_);
    }

    concatenator_layer_ = new FloatConcatenatorLayer;
    network_.AddLayer(concatenator_layer_);

    feedforward_layer_ = new FloatFeedforwardLayer(2*state_size,
                                                   hidden_size_);
    network_.AddLayer(feedforward_layer_);

    output_layer_ = new FloatSoftmaxOutputLayer(hidden_size_, output_size_);
    network_.AddLayer(output_layer_);

    // Connect the layers.
    lookup_layer_->SetNumInputs(1);
    lookup_layer_->SetNumOutputs(1);

    linear_layer_->SetNumInputs(1);
    linear_layer_->SetNumOutputs(1);
    network_.ConnectLayers(lookup_layer_, linear_layer_, 0, 0);

    premise_extractor_layer_->SetNumInputs(1);
    premise_extractor_layer_->SetNumOutputs(1);
    network_.ConnectLayers(linear_layer_, premise_extractor_layer_, 0, 0);

    hypothesis_extractor_layer_->SetNumInputs(1);
    hypothesis_extractor_layer_->SetNumOutputs(1);
    network_.ConnectLayers(linear_layer_, hypothesis_extractor_layer_, 0, 0);

    // Premise RNN.
    rnn_layer_->SetNumInputs(1);
    if (use_lstms_) {
      rnn_layer_->SetNumOutputs(2); // Cell states are an additional output.
    } else {
      rnn_layer_->SetNumOutputs(1);
    }
    network_.ConnectLayers(premise_extractor_layer_, rnn_layer_, 0, 0);

    // Selector of the last state of the RNN layer.
    if (connect_rnns_ || !use_attention_) {
      premise_selector_layer_->SetNumInputs(1);
      premise_selector_layer_->SetNumOutputs(1);
      if (connect_rnns_ && use_lstms_) {
        // Cell states.
        network_.ConnectLayers(rnn_layer_, premise_selector_layer_, 1, 0);
      } else {
        // Hidden states.
        network_.ConnectLayers(rnn_layer_, premise_selector_layer_, 0, 0);
      }
    }

    if (connect_rnns_) {
      hypothesis_rnn_layer_->set_use_control(true);
      hypothesis_rnn_layer_->SetNumInputs(2);
      if (use_lstms_) {
        hypothesis_rnn_layer_->SetNumOutputs(2);
      } else {
        hypothesis_rnn_layer_->SetNumOutputs(1);
      }
      network_.ConnectLayers(hypothesis_extractor_layer_, hypothesis_rnn_layer_,
                             0, 0);
      network_.ConnectLayers(premise_selector_layer_, hypothesis_rnn_layer_,
                             0, 1);
    } else {
      hypothesis_rnn_layer_->SetNumInputs(1);
      if (use_lstms_) {
        hypothesis_rnn_layer_->SetNumOutputs(2);
      } else {
        hypothesis_rnn_layer_->SetNumOutputs(1);
      }
      network_.ConnectLayers(hypothesis_extractor_layer_, hypothesis_rnn_layer_,
                             0, 0);
    }

    if (use_attention_) {
      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetNumOutputs(1);
      network_.ConnectLayers(hypothesis_rnn_layer_, hypothesis_selector_layer_,
                             0, 0);

      attention_layer_->SetNumInputs(2);
      attention_layer_->SetNumOutputs(1);
      network_.ConnectLayers(rnn_layer_, attention_layer_, 0, 0);
      network_.ConnectLayers(hypothesis_selector_layer_, attention_layer_,
                             0, 1);

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetNumOutputs(1);
      network_.ConnectLayers(attention_layer_, concatenator_layer_, 0, 0);
      network_.ConnectLayers(hypothesis_selector_layer_, concatenator_layer_,
                             0, 1);

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetNumOutputs(1);
      network_.ConnectLayers(concatenator_layer_, feedforward_layer_, 0, 0);
    } else {
      hypothesis_selector_layer_->SetNumInputs(1);
      hypothesis_selector_layer_->SetNumOutputs(1);
      network_.ConnectLayers(hypothesis_rnn_layer_, hypothesis_selector_layer_,
                             0, 0);

      concatenator_layer_->SetNumInputs(2);
      concatenator_layer_->SetNumOutputs(1);
      // Note: this will be weird if using lstms, since there premise_selector
      // is selecting the last cell state.
      network_.ConnectLayers(premise_selector_layer_, concatenator_layer_,
                             0, 0);
      network_.ConnectLayers(hypothesis_selector_layer_, concatenator_layer_,
                             0, 1);

      feedforward_layer_->SetNumInputs(1);
      feedforward_layer_->SetNumOutputs(1);
      network_.ConnectLayers(concatenator_layer_, feedforward_layer_, 0, 0);
    }

    output_layer_->SetNumInputs(1);
    output_layer_->SetNumOutputs(1);
    network_.ConnectLayers(feedforward_layer_, output_layer_, 0, 0);

    // Sort layers by topological order.
    network_.SortLayersByTopologicalOrder();
  }

  void SetModelPrefix(const std::string &model_prefix) {
    model_prefix_ = model_prefix;
  }

  int GetEmbeddingSize() { return lookup_layer_->embedding_dimension(); }

  int GetInputSize() { return linear_layer_->output_size(); }

  void InitializeParameters() {
    srand(1234);

    const std::vector<FloatLayer*> &layers = network_.GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->InitializeParameters();
    }

    if (use_ADAM_) {
      double beta1 = 0.9;
      double beta2 = 0.999;
      double epsilon = 1e-8; //1e-5; // 1e-8;

      for (int k = 0; k < layers.size(); ++k) {
        layers[k]->InitializeADAM(beta1, beta2, epsilon);
      }
    }
  }

  void LoadModel(const std::string &prefix) {
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
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
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      std::ostringstream ss;
      ss << "Layer" << k << "_" << layers[k]->name() + "_";
      //layers[k]->SaveParameters(prefix + ss.str(), false);
      layers[k]->SaveParameters(prefix + ss.str(), true);
      if (use_ADAM_) {
        layers[k]->SaveADAMParameters(prefix + ss.str());
      }
    }
  }

  void SetFixedEmbeddings(const FloatMatrix &fixed_embeddings) {
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

      //assert(false);
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
      FloatVector p = output_layer_->GetOutput(0);
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
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
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
    apply_dropout_ = false;
    RunForwardPass(input_sequence);
    test_ = false;
    apply_dropout_ = apply_dropout;
    int prediction;
    FloatVector p = output_layer_->GetOutput(0);
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
    if (connect_rnns_ || !use_attention_) {
      premise_selector_layer_->DefineBlock(0, separator-1, state_size, 1);
    }

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

    const std::vector<FloatLayer*> &layers = network_.GetLayers();
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
      FloatVector p = output_layer_->GetOutput(0);
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
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
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
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
    for (int k = 0; k < layers.size(); ++k) {
      layers[k]->ResetGradients();
    }
  }

  void UpdateParameters(int batch_size,
                        double learning_rate,
                        double regularization_constant) {
    const std::vector<FloatLayer*> &layers = network_.GetLayers();
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
  FloatNeuralNetwork network_;
  FloatLookupLayer *lookup_layer_;
  FloatLinearLayer *linear_layer_;
  FloatRNNLayer *rnn_layer_;
  FloatRNNLayer *hypothesis_rnn_layer_;
  FloatLinearLayer *linear_layer_after_rnn_;
  FloatAttentionLayer *attention_layer_;
  FloatFeedforwardLayer *feedforward_layer_;
  FloatAverageLayer *average_layer_ = NULL; // Remove this variable?
  FloatSelectorLayer *hypothesis_selector_layer_;
  FloatSelectorLayer *premise_selector_layer_;
  FloatSelectorLayer *hypothesis_extractor_layer_;
  FloatSelectorLayer *premise_extractor_layer_;
  FloatConcatenatorLayer *concatenator_layer_;
  FloatSoftmaxOutputLayer *output_layer_;
  int embedding_size_;
  int input_size_;
  int hidden_size_;
  int output_size_;
  bool use_attention_;
  int attention_type_;
  bool use_separate_rnns_;
  bool connect_rnns_;
  bool use_lstms_;
  bool use_bidirectional_rnns_;
  bool use_linear_layer_after_rnn_;
  bool use_average_layer_;
  bool apply_dropout_;
  double dropout_probability_;
  bool use_ADAM_;
  bool write_attention_probabilities_;
  bool test_;
  std::string model_prefix_;
  std::ofstream os_attention_;
  //FloatMatrix x_;
  //FloatMatrix h_;
};

#endif /* RNN_H_ */

