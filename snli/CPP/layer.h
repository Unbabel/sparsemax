#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "snli_data.h"

class Layer {
 public:
  Layer() {};
  ~Layer() {};

  virtual void ResetParameters() = 0;

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) = 0;

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix*> *weight_derivatives,
      std::vector<Vector*> *bias_derivatives) = 0;

  virtual double GetUniformInitializationLimit(Matrix *W) = 0;

  void InitializeParameters() {
    std::vector<Matrix*> weights;
    std::vector<Vector*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    ResetParameters();
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    for (auto b: biases) {
      b->setZero();
    }
    for (auto W: weights) {
      double max = GetUniformInitializationLimit(W);
      for (int i = 0; i < W->rows(); ++i) {
        for (int j = 0; j < W->cols(); ++j) {
          double t = max *
            (2.0*static_cast<double>(rand()) / RAND_MAX - 1.0);
          (*W)(i, j) = t;
          //std::cout << t/max << std::endl;
        }
      }
    }
  }

  void ReadParameters() {
    std::vector<Matrix*> weights;
    std::vector<Vector*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto name = bias_names[i];
      std::cout << "Loading " << name << "..." << std::endl;
      LoadVectorParameter(name, b);
    }
    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto name = weight_names[i];
      std::cout << "Loading " << name << "..." << std::endl;
      LoadMatrixParameter(name, W);
    }
  }

  void CheckGradient(int num_checks, double delta) {
    std::vector<Matrix*> weights, weight_derivatives;
    std::vector<Vector*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    ResetGradients();
    RunForward();
    RunBackward();

    std::cout << name_ << " " << output_.size() << " "
              << output_derivative_.size()
              << std::endl;
    assert(output_.size() == output_derivative_.size());

    //float delta = 1e-3; //1e-5; //1e-5;
    for (int check = 0; check < num_checks; ++check) {
      for (int i = 0; i < biases.size(); ++i) {
        auto name = bias_names[i];
        auto b = biases[i];
        auto db = bias_derivatives[i];
        int r = static_cast<int>(b->size() *
                                 static_cast<double>(rand()) / RAND_MAX);
        double value = (*b)[r];
        (*b)[r] = value + delta;
        RunForward();
        double out0 = (output_.array() * output_derivative_.array()).sum();
        (*b)[r] = value - delta;
        RunForward();
        double out1 = (output_.array() * output_derivative_.array()).sum();
        (*b)[r] = value; // Put the value back.
        RunForward();
        double numeric_gradient = (out0 - out1) / (2 * delta);
        double analytic_gradient = (*db)[r];
        double relative_error = 0.0;
        if (numeric_gradient + analytic_gradient != 0.0) {
          relative_error = fabs(numeric_gradient - analytic_gradient) /
            fabs(numeric_gradient + analytic_gradient);
        }
        std::cout << name << ": "
                  << numeric_gradient << ", " << analytic_gradient
                  << " => " << relative_error
                  << std::endl;
      }
      for (int i = 0; i < weights.size(); ++i) {
        auto name = weight_names[i];
        auto W = weights[i];
        auto dW = weight_derivatives[i];
        int r = static_cast<int>(W->size() *
                                 static_cast<double>(rand()) / RAND_MAX);
        double value = (*W)(r);
        assert(output_.size() == output_derivative_.size());
        (*W)(r) = value + delta;
        RunForward();
        double out0 = (output_.array() * output_derivative_.array()).sum();
        (*W)(r) = value - delta;
        RunForward();
        double out1 = (output_.array() * output_derivative_.array()).sum();
        (*W)(r) = value; // Put the value back.
        RunForward();
        double numeric_gradient = (out0 - out1) / (2 * delta);
        double analytic_gradient = (*dW)(r);
        double relative_error = 0.0;
        if (numeric_gradient + analytic_gradient != 0.0) {
          relative_error = fabs(numeric_gradient - analytic_gradient) /
            fabs(numeric_gradient + analytic_gradient);
        }
        std::cout << name << ": "
                  << numeric_gradient << ", " << analytic_gradient
                  << " => " << relative_error
                  << std::endl;
      }
    }

    std::cout << std::endl;
  }

  virtual void ResetGradients() = 0;
  virtual void RunForward() = 0;
  virtual void RunBackward() = 0;
  virtual void UpdateParameters(double learning_rate) = 0;

  int GetNumInputs() const { return inputs_.size(); }
  void SetNumInputs(int n) {
    inputs_.resize(n);
    input_derivatives_.resize(n);
  }
  void SetInput(int i, const Matrix &input) { inputs_[i] = &input; }
  const Matrix &GetOutput() const { return output_; }
  void SetInputDerivative(int i, Matrix *input_derivative) {
    input_derivatives_[i] = input_derivative;
  }
  Matrix *GetOutputDerivative() { return &output_derivative_; }

 protected:
  const Matrix &GetInput() {
    assert(GetNumInputs() == 1);
    return *inputs_[0];
  }
  Matrix *GetInputDerivative() {
    assert(GetNumInputs() == 1);
    return input_derivatives_[0];
  }

 protected:
  std::string name_;
  std::vector<const Matrix*> inputs_;
  Matrix output_;
  std::vector<Matrix*> input_derivatives_;
  Matrix output_derivative_;
};

class SelectorLayer : public Layer {
 public:
  SelectorLayer() { name_ = "Selector"; }
  virtual ~SelectorLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    const Matrix &x = GetInput();
    output_ = x.block(first_row_, first_column_, num_rows_, num_columns_);
  }

  void RunBackward() {
    Matrix *dx = GetInputDerivative();
    (*dx).block(first_row_, first_column_, num_rows_, num_columns_) +=
      output_derivative_;
  }

  void UpdateParameters(double learning_rate) {}

  void DefineBlock(int first_row, int first_column,
                   int num_rows, int num_columns) {
    first_row_ = first_row;
    first_column_ = first_column;
    num_rows_ = num_rows;
    num_columns_ = num_columns;
  }

 protected:
  int first_row_;
  int first_column_;
  int num_rows_;
  int num_columns_;
};

class ConcatenatorLayer : public Layer {
 public:
  ConcatenatorLayer() { name_ = "Concatenator"; }
  virtual ~ConcatenatorLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    int num_rows = 0;
    int num_columns = inputs_[0]->cols();
    for (int i = 0; i < GetNumInputs(); ++i) {
      assert(inputs_[i]->cols() == num_columns);
      num_rows += inputs_[i]->rows();
    }
    output_.setZero(num_rows, num_columns);
    int start = 0;
    for (int i = 0; i < GetNumInputs(); ++i) {
      output_.block(start, 0, inputs_[i]->rows(), num_columns) = *inputs_[i];
      start += inputs_[i]->rows();
    }
  }

  void RunBackward() {
    int num_columns = output_derivative_.cols();
    int start = 0;
    for (int i = 0; i < GetNumInputs(); ++i) {
      *input_derivatives_[i] +=
        output_derivative_.block(start, 0, input_derivatives_[i]->rows(),
                                 num_columns);
      start += inputs_[i]->rows();
    }
  }

  void UpdateParameters(double learning_rate) {}
};

class LookupLayer : public Layer {
 public:
  LookupLayer(int num_words, int embedding_dimension) {
    name_ = "Lookup";
    num_words_ = num_words;
    embedding_dimension_ = embedding_dimension;
    updatable_.assign(num_words, true);
    //std::cout << "Num words = " << num_words_ << std::endl;
  }

  virtual ~LookupLayer() {}

  int embedding_dimension() { return embedding_dimension_; }

  void ResetParameters() {
    E_ = Matrix::Zero(embedding_dimension_, num_words_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&E_);
    weight_names->push_back("embeddings");
  }

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    // This layer does not store full derivatives of the embeddings.
    assert(false);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    return 0.05;
  }

  void SetFixedEmbeddings(const Matrix &fixed_embeddings,
                          const std::vector<int> &word_ids) {
    for (int i = 0; i < fixed_embeddings.cols(); i++) {
      int wid = word_ids[i];
      E_.col(wid) = fixed_embeddings.col(i);
      updatable_[wid] = false;
    }
  }

  void ResetGradients() {
    // TODO(afm): decide how to handle mini-batch gradients in this layer.
  }

  void RunForward() {
    output_.setZero(embedding_dimension_, input_sequence_->size());
    assert(output_.rows() == embedding_dimension_ &&
           output_.cols() == input_sequence_->size());
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      assert(wid >= 0 && wid < E_.cols());
      output_.col(t) = E_.col(wid);
    }
  }

  void RunBackward() {}

  // NOTE: this is not supporting mini-batch!!!
  void UpdateParameters(double learning_rate) {
    // Update word embeddings.
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      if (!updatable_[wid]) continue;
      E_.col(wid) -= learning_rate * output_derivative_.col(t);

      //std::cout << "param emb layer" << std::endl;
      //std::cout << E_(0,wid) << std::endl;
    }
  }

  void set_input_sequence(const std::vector<Input> &input_sequence) {
    input_sequence_ = &input_sequence;
  }

 public:
  int num_words_;
  int embedding_dimension_;
  Matrix E_;
  std::vector<bool> updatable_;

  const std::vector<Input> *input_sequence_; // Input.
};

class LinearLayer : public Layer {
 public:
  LinearLayer(int input_size, int output_size) {
    name_ = "Linear";
    input_size_ = input_size;
    output_size_ = output_size;
  }

  virtual ~LinearLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Wxy_ = Matrix::Zero(output_size_, input_size_);
    by_ = Vector::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Wxy_);
    weight_names->push_back("linear_weights");

    biases->push_back(&by_);
    bias_names->push_back("linear_bias");
  }

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxy_);
    bias_derivatives->push_back(&dby_);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff = 1.0; // Like in TANH.
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    dWxy_.setZero(output_size_, input_size_);
    dby_.setZero(output_size_);
  }

  void RunForward() {
    const Matrix &x = GetInput();
    output_ = (Wxy_ * x).colwise() + by_;
  }

  void RunBackward() {
    const Matrix &x = GetInput();
    Matrix *dx = GetInputDerivative();
    (*dx).noalias() += Wxy_.transpose() * output_derivative_;
    dWxy_.noalias() += output_derivative_ * x.transpose();
    dby_.noalias() += output_derivative_.rowwise().sum();
  }

  void UpdateParameters(double learning_rate) {
    Wxy_ -= learning_rate * dWxy_;
    by_ -= learning_rate * dby_;
  }

 protected:
  int input_size_;
  int output_size_;
  Matrix Wxy_;
  Vector by_;

  Matrix dWxy_;
  Vector dby_;
};

class SoftmaxOutputLayer : public Layer {
 public:
  SoftmaxOutputLayer() {}
  SoftmaxOutputLayer(int input_size,
                     int output_size) {
    name_ = "SoftmaxOutput";
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~SoftmaxOutputLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Why_ = Matrix::Zero(output_size_, input_size_);
    by_ = Vector::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Why_);
    biases->push_back(&by_);

    weight_names->push_back("Why");
    bias_names->push_back("by");
  }

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    weight_derivatives->push_back(&dWhy_);
    bias_derivatives->push_back(&dby_);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff = 4.0; // Like in LOGISTIC.
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    dWhy_.setZero(output_size_, input_size_);
    dby_.setZero(output_size_);
  }

  void RunForward() {
    const Matrix &h = GetInput();
    assert(h.cols() == 1);
    Matrix y = Why_ * h + by_;
    float logsum = LogSumExp(y);
    output_ = (y.array() - logsum).exp(); // This is the probability vector.
  }

  void RunBackward() {
    const Matrix &h = GetInput();
    assert(h.cols() == 1);
    Matrix *dh = GetInputDerivative();
    assert(dh->cols() == 1);

    Vector dy = output_;
    dy[output_label_] -= 1.0; // Backprop into y (softmax grad).
    dWhy_.noalias() += dy * h.transpose();
    dby_.noalias() += dy;
    (*dh).noalias() += Why_.transpose() * dy; // Backprop into h.
  }

  void UpdateParameters(double learning_rate) {
    Why_ -= learning_rate * dWhy_;
    by_ -= learning_rate * dby_;
  }

  int output_label() { return output_label_; }
  void set_output_label(int output_label) {
    output_label_ = output_label;
  }

 protected:
  int input_size_;
  int output_size_;

  Matrix Why_;
  Vector by_;

  Matrix dWhy_;
  Vector dby_;

  int output_label_; // Output.
};

class FeedforwardLayer : public Layer {
 public:
  FeedforwardLayer() {}
  FeedforwardLayer(int input_size,
                   int output_size) {
    name_ = "Feedforward";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~FeedforwardLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Wxh_ = Matrix::Zero(output_size_, input_size_);
    bh_ = Vector::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Wxh_);
    biases->push_back(&bh_);

    weight_names->push_back("Wxh");
    bias_names->push_back("bh");
  }

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxh_);
    bias_derivatives->push_back(&dbh_);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    if (activation_function_ == ActivationFunctions::LOGISTIC) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    dWxh_.setZero(output_size_, input_size_);
    dbh_.setZero(output_size_);
  }

  void RunForward() {
    // In the normal case, x is a column vector.
    // If x has several columns, one assumes a feedforward net
    // for every column with shared parameters.
    const Matrix &x = GetInput();
    EvaluateActivation(activation_function_,
                       (Wxh_ * x).colwise() + bh_,
                       &output_);
  }

  void RunBackward() {
    const Matrix &x = GetInput();
    Matrix *dx = GetInputDerivative();
    Matrix dhraw;
    DerivateActivation(activation_function_, output_, &dhraw);
    dhraw = dhraw.array() * output_derivative_.array();
    dWxh_.noalias() += dhraw * x.transpose();
    std::cout << dbh_.size() << " " << dhraw.size() << std::endl;
    dbh_.noalias() += dhraw.rowwise().sum();
    *dx += Wxh_.transpose() * dhraw; // Backprop into x.
  }

  void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
  }

 protected:
  int activation_function_;
  int output_size_;
  int input_size_;

  Matrix Wxh_;
  Vector bh_;

  Matrix dWxh_;
  Vector dbh_;
};

class RNNLayer : public Layer {
 public:
  RNNLayer() {}
  RNNLayer(int input_size,
           int hidden_size) {
    name_ = "RNN";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
  }
  virtual ~RNNLayer() {}

  int input_size() const { return input_size_; }
  int hidden_size() const { return hidden_size_; }

  virtual void ResetParameters() {
    Wxh_ = Matrix::Zero(hidden_size_, input_size_);
    Whh_ = Matrix::Zero(hidden_size_, hidden_size_);
    bh_ = Vector::Zero(hidden_size_);
    if (use_hidden_start_) {
      h0_ = Vector::Zero(hidden_size_);
    }
  }

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    weights->push_back(&Wxh_);
    weights->push_back(&Whh_);

    biases->push_back(&bh_);
    if (use_hidden_start_) {
      biases->push_back(&h0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxh");
    weight_names->push_back("Whh");

    bias_names->push_back("bh");
    if (use_hidden_start_) {
      bias_names->push_back("h0");
    }
  }

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix*> *weight_derivatives,
      std::vector<Vector*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxh_);
    weight_derivatives->push_back(&dWhh_);
    bias_derivatives->push_back(&dbh_);
    if (use_hidden_start_) {
      bias_derivatives->push_back(&dh0_); // Not really a bias, but goes here.
    }
  }

  virtual double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    if (activation_function_ == ActivationFunctions::LOGISTIC) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  virtual void ResetGradients() {
    dWxh_.setZero(hidden_size_, input_size_);
    dbh_.setZero(hidden_size_);
    dWhh_.setZero(hidden_size_, hidden_size_);
    if (use_hidden_start_) {
      dh0_.setZero(hidden_size_);
    }
  }

  virtual void RunForward() {
    const Matrix &x = GetInput();
    int length = x.cols();
    output_.setZero(hidden_size_, length);
    Matrix hraw = (Wxh_ * x).colwise() + bh_;
    Vector hprev = Vector::Zero(output_.rows());
    if (use_hidden_start_) hprev = h0_;
    Vector result;
    for (int t = 0; t < length; ++t) {
      EvaluateActivation(activation_function_,
                         hraw.col(t) + Whh_ * hprev,
                         &result);
      output_.col(t) = result;
      hprev = output_.col(t);
    }
  }

  virtual void RunBackward() {
    const Matrix &x = GetInput();
    Matrix *dx = GetInputDerivative();

    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = output_derivative_;

    Matrix dhraw;
    DerivateActivation(activation_function_, output_, &dhraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = Whh_.transpose() * dhraw.col(t);
    }

    *dx += Wxh_.transpose() * dhraw; // Backprop into x.

    dWxh_.noalias() += dhraw * x.transpose();
    dbh_.noalias() += dhraw.rowwise().sum();
    dWhh_.noalias() += dhraw.rightCols(length-1) *
      output_.leftCols(length-1).transpose();
    dh0_.noalias() += dhnext;
  }

  virtual void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
    Whh_ -= learning_rate * dWhh_;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dh0_;
    }
  }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  bool use_hidden_start_;

  Matrix Wxh_;
  Matrix Whh_;
  Vector bh_;
  Vector h0_;

  Matrix dWxh_;
  Matrix dWhh_;
  Vector dbh_;
  Vector dh0_;
};

class GRULayer : public RNNLayer {
 public:
  GRULayer() {}
  GRULayer(int input_size,
           int hidden_size) {
    name_ = "GRU";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
  }
  virtual ~GRULayer() {}

  void ResetParameters() {
    RNNLayer::ResetParameters();

    Wxz_ = Matrix::Zero(hidden_size_, input_size_);
    Whz_ = Matrix::Zero(hidden_size_, hidden_size_);
    Wxr_ = Matrix::Zero(hidden_size_, input_size_);
    Whr_ = Matrix::Zero(hidden_size_, hidden_size_);
    bz_ = Vector::Zero(hidden_size_);
    br_ = Vector::Zero(hidden_size_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    RNNLayer::CollectAllParameters(weights, biases, weight_names, bias_names);

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

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    RNNLayer::CollectAllParameterDerivatives(weight_derivatives,
                                             bias_derivatives);
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWhz_);
    weight_derivatives->push_back(&dWxr_);
    weight_derivatives->push_back(&dWhr_);
    bias_derivatives->push_back(&dbz_);
    bias_derivatives->push_back(&dbr_);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &Wxz_ || W == &Whz_ || W == &Wxr_ || W == &Whr_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    RNNLayer::ResetGradients();
    dWxz_.setZero(hidden_size_, input_size_);
    dWhz_.setZero(hidden_size_, hidden_size_);
    dWxr_.setZero(hidden_size_, input_size_);
    dWhr_.setZero(hidden_size_, hidden_size_);
    dbz_.setZero(hidden_size_);
    dbr_.setZero(hidden_size_);
  }

  void RunForward() {
    const Matrix &x = GetInput();

    int length = x.cols();
    z_.setZero(hidden_size_, length);
    r_.setZero(hidden_size_, length);
    hu_.setZero(hidden_size_, length);
    output_.setZero(hidden_size_, length);
    Matrix zraw = (Wxz_ * x).colwise() + bz_;
    Matrix rraw = (Wxr_ * x).colwise() + br_;
    Matrix hraw = (Wxh_ * x).colwise() + bh_;
    Vector hprev = Vector::Zero(output_.rows());
    if (use_hidden_start_) hprev = h0_;
    Vector result;
    for (int t = 0; t < length; ++t) {
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         zraw.col(t) + Whz_ * hprev,
                         &result);
      z_.col(t) = result;

      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         rraw.col(t) + Whr_ * hprev,
                         &result);
      r_.col(t) = result;

      EvaluateActivation(activation_function_,
                         hraw.col(t) + Whh_ * r_.col(t).cwiseProduct(hprev),
                         &result);
      hu_.col(t) = result;
      output_.col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;
      hprev = output_.col(t);
    }
  }

  void RunBackward() {
    const Matrix &x = GetInput();
    Matrix *dx = GetInputDerivative();

    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = output_derivative_;

    Matrix dhuraw;
    DerivateActivation(activation_function_, hu_, &dhuraw);
    Matrix dzraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, z_, &dzraw);
    Matrix drraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, r_, &drraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t) + dhnext; // Backprop into h.
      Vector dhu = z_.col(t).cwiseProduct(dh);

      dhuraw.col(t) = dhuraw.col(t).cwiseProduct(dhu);
      Vector hprev;
      if (t == 0) {
        hprev = Vector::Zero(output_.rows());
      } else {
        hprev = output_.col(t-1);
      }

      Vector dq = Whh_.transpose() * dhuraw.col(t);
      Vector dz = (hu_.col(t) - hprev).cwiseProduct(dh);
      Vector dr = hprev.cwiseProduct(dq);

      dzraw.col(t) = dzraw.col(t).cwiseProduct(dz);
      drraw.col(t) = drraw.col(t).cwiseProduct(dr);

      dhnext.noalias() =
        Whz_.transpose() * dzraw.col(t) +
        Whr_.transpose() * drraw.col(t) +
        r_.col(t).cwiseProduct(dq) +
        (1.0 - z_.col(t).array()).matrix().cwiseProduct(dh);
    }

    *dx += Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
      Wxh_.transpose() * dhuraw; // Backprop into x.

    dWxz_.noalias() += dzraw * x.transpose();
    dbz_.noalias() += dzraw.rowwise().sum();
    dWxr_.noalias() += drraw * x.transpose();
    dbr_.noalias() += drraw.rowwise().sum();
    dWxh_.noalias() += dhuraw * x.transpose();
    dbh_.noalias() += dhuraw.rowwise().sum();

    dWhz_.noalias() += dzraw.rightCols(length-1) *
      output_.leftCols(length-1).transpose();
    dWhr_.noalias() += drraw.rightCols(length-1) *
      output_.leftCols(length-1).transpose();
    dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((r_.rightCols(length-1)).cwiseProduct(output_.leftCols(length-1))).
      transpose();

    dh0_.noalias() += dhnext;
  }

  void UpdateParameters(double learning_rate) {
    RNNLayer::UpdateParameters(learning_rate);
    Wxz_ -= learning_rate * dWxz_;
    Whz_ -= learning_rate * dWhz_;
    Wxr_ -= learning_rate * dWxr_;
    Whr_ -= learning_rate * dWhr_;
    bz_ -= learning_rate * dbz_;
    br_ -= learning_rate * dbr_;
  }

 protected:
  Matrix Wxz_;
  Matrix Whz_;
  Matrix Wxr_;
  Matrix Whr_;
  Vector bz_;
  Vector br_;

  Matrix dWxz_;
  Matrix dWhz_;
  Matrix dWxr_;
  Matrix dWhr_;
  Vector dbz_;
  Vector dbr_;

  Matrix z_;
  Matrix r_;
  Matrix hu_;
};

class AttentionLayer : public Layer {
 public:
  AttentionLayer() {}
  AttentionLayer(int input_size,
                 int control_size,
                 int hidden_size,
                 bool use_sparsemax) {
    name_ = "Attention";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    control_size_ = control_size;
    hidden_size_ = hidden_size;
    use_sparsemax_ = use_sparsemax;
  }
  virtual ~AttentionLayer() {}

  int input_size() const { return input_size_; }
  int control_size() const { return control_size_; }
  int hidden_size() const { return hidden_size_; }

  void ResetParameters() {
    Wxz_ = Matrix::Zero(hidden_size_, input_size_);
    Wyz_ = Matrix::Zero(hidden_size_, control_size_);
    wzp_ = Matrix::Zero(hidden_size_, 1);
    bz_ = Vector::Zero(hidden_size_);
  }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Wxz_);
    weights->push_back(&Wyz_);
    weights->push_back(&wzp_);

    biases->push_back(&bz_);

    weight_names->push_back("Wxz");
    weight_names->push_back("Wyz");
    weight_names->push_back("wzp");

    bias_names->push_back("bz");
  }

  void CollectAllParameterDerivatives(std::vector<Matrix*> *weight_derivatives,
                                      std::vector<Vector*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWyz_);
    weight_derivatives->push_back(&dwzp_);
    bias_derivatives->push_back(&dbz_);
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    if (activation_function_ == ActivationFunctions::LOGISTIC) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    dWxz_.setZero(hidden_size_, input_size_);
    dWyz_.setZero(hidden_size_, control_size_);
    dwzp_.setZero(hidden_size_, 1);
    dbz_.setZero(hidden_size_);
  }

  void RunForward() {
    assert(GetNumInputs() == 2);
    const Matrix &X = *(inputs_[0]); // Input subject to attention.
    const Matrix &y = *(inputs_[1]); // Control input.
    assert(y.cols() == 1);

#if 0
    std::cout << "X_att=" << X << std::endl;
    std::cout << "y_att=" << y << std::endl;
#endif

    int length = X.cols();
    z_.setZero(hidden_size_, length);

    for (int t = 0; t < length; ++t) {
      Matrix result;
      EvaluateActivation(activation_function_,
                         Wxz_ * X.col(t) + bz_ + Wyz_ * y,
                         &result);
      z_.col(t) = result;
    }

#if 0
    std::cout << "z_att=" << z_ << std::endl;
#endif

    Vector v = z_.transpose() * wzp_;
    if (use_sparsemax_) {
      float tau;
      ProjectOntoSimplex(v, 1.0, &p_, &tau);
    } else {
      float logsum = LogSumExp(v);
      p_ = (v.array() - logsum).exp();
    }

#if 0
    std::cout << "p_att=" << p_ << std::endl;
#endif

    output_.noalias() = X * p_;
  }

  void RunBackward() {
    assert(GetNumInputs() == 2);
    const Matrix &X = *(inputs_[0]); // Input subject to attention.
    const Matrix &y = *(inputs_[1]); // Control input.
    Matrix *dX = input_derivatives_[0];
    Matrix *dy = input_derivatives_[1];
    assert(y.cols() == 1);
    assert(output_derivative_.cols() == 1);

    int length = X.cols();
    dp_.noalias() = X.transpose() * output_derivative_;

    Vector Jdp;
    if (use_sparsemax_) {
      // Compute Jparsemax * dp_.
      // Let s = supp(p_) and k = sum(s).
      // Jsparsemax = diag(s) - s*s.transpose() / k.
      // Jsparsemax * dp_ = s.array() * dp_.array() - s*s.transpose() * dp_ / k
      //                  = s.array() * dp_.array() - val * s,
      // where val = s.transpose() * dp_ / k.
      //
      // With array-indexing this would be:
      //
      // float val = dp_[mask].sum() / mask.size();
      // Jdp[mask] = dp_[mask] - val;
      int nnz = 0;
      float val = 0.0;
      for (int i = 0; i < p_.size(); ++i) {
        if (p_[i] > 0.0) {
          val += dp_[i];
          ++nnz;
        }
      }
      val /= static_cast<float>(nnz);
      Jdp.setZero(p_.size());
      for (int i = 0; i < p_.size(); ++i) {
        if (p_[i] > 0.0) {
          Jdp[i] = dp_[i] - val;
        }
      }
    } else {
      // Compute Jsoftmax * dp_.
      // Jsoftmax = diag(p_) - p_*p_.transpose().
      // Jsoftmax * dp_ = p_.array() * dp_.array() - p_* (p_.transpose() * dp_).
      //                = p_.array() * (dp_ - val).array(),
      // where val = p_.transpose() * dp_.
      float val = p_.transpose() * dp_;
      Jdp = p_.array() * (dp_.array() - val);
    }
    dz_ = wzp_ * Jdp.transpose();

    Matrix dzraw = Matrix::Zero(hidden_size_, length);
    for (int t = 0; t < length; ++t) {
      Matrix result; // TODO: perform this in a matrix-level way to be more efficient.
      DerivateActivation(activation_function_, z_.col(t), &result);
      dzraw.col(t).noalias() = result;
    }
    dzraw = dzraw.array() * dz_.array();
    Vector dzraw_sum = dzraw.rowwise().sum();

    *dX += Wxz_.transpose() * dzraw;
    *dX += output_derivative_ * p_.transpose();
    *dy += Wyz_.transpose() * dzraw_sum;

    dwzp_ += z_ * Jdp;
    dWxz_.noalias() += dzraw * X.transpose();
    dWyz_.noalias() += dzraw_sum * y.transpose();
    dbz_.noalias() += dzraw_sum;
  }

  void UpdateParameters(double learning_rate) {
    Wxz_ -= learning_rate * dWxz_;
    Wyz_ -= learning_rate * dWyz_;
    dbz_ -= learning_rate * dbz_;
    wzp_ -= learning_rate * dwzp_;
  }

#if 0
  void set_x(const Matrix &x) { x_ = &x; }
  void set_y(const Vector &y) { y_ = &y; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  void set_dy(Vector *dy) { dy_ = dy; }
  const Vector &get_u() { return u_; }
  Vector *get_mutable_du() { return &du_; }
#endif

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  int control_size_;
  bool use_sparsemax_;

  Matrix Wxz_;
  Matrix Wyz_;
  Matrix wzp_; // Column vector.
  Vector bz_;

  Matrix dWxz_;
  Matrix dWyz_;
  Matrix dwzp_;
  Vector dbz_;

#if 0
  const Matrix *x_; // Input.
  const Vector *y_; // Input (control).
  Vector u_; // Output.
#endif

  Matrix z_;
  Vector p_;

#if 0
  Matrix *dx_;
  Vector *dy_;
  Vector du_;
#endif

  Matrix dz_;
  Vector dp_;
};

#endif /* LAYER_H_ */
