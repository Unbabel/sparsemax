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

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) = 0;

  virtual double GetUniformInitializationLimit(Matrix *W) = 0;

  void InitializeParameters() {
    std::vector<Matrix*> weights;
    std::vector<Vector*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
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

  virtual void ResetGradients() = 0;
  virtual void RunForward() = 0;
  virtual void RunBackward() = 0;
  virtual void UpdateParameters(double learning_rate) = 0;
};

class SelectorLayer {
 public:
  SelectorLayer() {};
  ~SelectorLayer() {};

  void CollectAllParameters(std::vector<Matrix*> *weights,
			    std::vector<Vector*> *biases,
			    std::vector<std::string> *weight_names,
			    std::vector<std::string> *bias_names) {}

  double GetUniformInitializationLimit(Matrix *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    assert(num_rows_ == 1 || num_columns_ == 1);
    y_ = (*x_).block(first_row_, first_column_,
		     num_rows_, num_columns_);
  }

  void RunBackward() {
    (*dx_).block(first_row_, first_column_, num_rows_, num_columns_) += dy_;
  }

  void UpdateParameters(double learning_rate) {}

  void DefineBlock(int first_row, int first_column,
		   int num_rows, int num_columns) {
    first_row_ = first_row;
    first_column_ = first_column;
    num_rows_ = num_rows;
    num_columns_ = num_columns;
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  const Vector &get_y() { return y_; }
  Vector *get_mutable_dy() { return &dy_; }

 protected:
  int first_row_;
  int first_column_;
  int num_rows_;
  int num_columns_;

  const Matrix *x_; // Input.
  Vector y_; // Output.
  Matrix *dx_;
  Vector dy_;
};

class MatrixSelectorLayer {
 public:
  MatrixSelectorLayer() {};
  ~MatrixSelectorLayer() {};

  void CollectAllParameters(std::vector<Matrix*> *weights,
			    std::vector<Vector*> *biases,
			    std::vector<std::string> *weight_names,
			    std::vector<std::string> *bias_names) {}

  double GetUniformInitializationLimit(Matrix *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    y_ = (*x_).block(first_row_, first_column_,
		     num_rows_, num_columns_);
  }

  void RunBackward() {
    (*dx_).block(first_row_, first_column_, num_rows_, num_columns_) += dy_;
  }

  void UpdateParameters(double learning_rate) {}

  void DefineBlock(int first_row, int first_column,
		   int num_rows, int num_columns) {
    first_row_ = first_row;
    first_column_ = first_column;
    num_rows_ = num_rows;
    num_columns_ = num_columns;
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  const Matrix &get_y() { return y_; }
  Matrix *get_mutable_dy() { return &dy_; }

 protected:
  int first_row_;
  int first_column_;
  int num_rows_;
  int num_columns_;

  const Matrix *x_; // Input.
  Matrix y_; // Output.
  Matrix *dx_;
  Matrix dy_;
};

class ConcatenatorLayer {
 public:
  ConcatenatorLayer() {};
  ~ConcatenatorLayer() {};

  void CollectAllParameters(std::vector<Matrix*> *weights,
			    std::vector<Vector*> *biases,
			    std::vector<std::string> *weight_names,
			    std::vector<std::string> *bias_names) {}

  double GetUniformInitializationLimit(Matrix *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    //assert(y_.size() == x1_->size() + x2_->size());
    y_.setZero(x1_->size() + x2_->size());
    y_.head(x1_->size()) = *x1_;
    y_.tail(x2_->size()) = *x2_;
  }

  void RunBackward() {
    assert(dy_.size() == dx1_->size() + dx2_->size());
    *dx1_ += dy_.head(dx1_->size());
    *dx2_ += dy_.tail(dx2_->size());
  }

  void UpdateParameters(double learning_rate) {}

  void set_x1(const Vector &x1) { x1_ = &x1; }
  void set_x2(const Vector &x2) { x2_ = &x2; }
  void set_dx1(Vector *dx1) { dx1_ = dx1; }
  void set_dx2(Vector *dx2) { dx2_ = dx2; }
  const Vector &get_y() { return y_; }
  Vector *get_mutable_dy() { return &dy_; }

 protected:
  int first_row_;
  int first_column_;
  int num_rows_;
  int num_columns_;

  const Vector *x1_; // Input.
  const Vector *x2_; // Input.
  Vector y_; // Output.
  Vector *dx1_;
  Vector *dx2_;
  Vector dy_;
};

class LookupLayer : public Layer {
 public:
  LookupLayer(int num_words, int embedding_dimension) {
    num_words_ = num_words;
    embedding_dimension_ = embedding_dimension;
    updatable_.assign(num_words, true);
    //std::cout << "Num words = " << num_words_ << std::endl;
  }

  virtual ~LookupLayer() {}

  int embedding_dimension() { return embedding_dimension_; }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    E_ = Matrix::Zero(embedding_dimension_, num_words_);

    weights->push_back(&E_);
    weight_names->push_back("embeddings");
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
    x_.setZero(embedding_dimension_, input_sequence_->size());
    assert(x_.rows() == embedding_dimension_ &&
           x_.cols() == input_sequence_->size());
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      assert(wid >= 0 && wid < E_.cols());
      x_.col(t) = E_.col(wid);
    }
  }

  void RunBackward() {}

  void UpdateParameters(double learning_rate) {
    // Update word embeddings.
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      if (!updatable_[wid]) continue;
      E_.col(wid) -= learning_rate * dx_.col(t);

      //std::cout << "param emb layer" << std::endl;
      //std::cout << E_(0,wid) << std::endl;
    }
  }

  void set_input_sequence(const std::vector<Input> &input_sequence) {
    input_sequence_ = &input_sequence;
  }
  const Matrix &get_x() { return x_; }
  Matrix *get_mutable_dx() { return &dx_; }

 public:
  int num_words_;
  int embedding_dimension_;
  Matrix E_;
  std::vector<bool> updatable_;

  const std::vector<Input> *input_sequence_; // Input.
  Matrix x_; // Output.
  Matrix dx_;
};

class LinearLayer : public Layer {
 public:
  LinearLayer(int input_size, int output_size) {
    input_size_ = input_size;
    output_size_ = output_size;
  }

  virtual ~LinearLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    Wxy_ = Matrix::Zero(output_size_, input_size_);
    by_ = Vector::Zero(output_size_);

    weights->push_back(&Wxy_);
    weight_names->push_back("linear_weights");

    biases->push_back(&by_);
    bias_names->push_back("linear_bias");
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
    y_ = (Wxy_ * (*x_)).colwise() + by_;
  }

  void RunBackward() {
    (*dx_).noalias() += Wxy_.transpose() * dy_;
    dWxy_.noalias() += dy_ * (*x_).transpose();
    dby_.noalias() += dy_.rowwise().sum();
  }

  void UpdateParameters(double learning_rate) {
    Wxy_ -= learning_rate * dWxy_;
    by_ -= learning_rate * dby_;

    //std::cout << "params linear layer" << std::endl;
    //std::cout << Wxy_(0,0) << std::endl;
    //std::cout << by_(0,0) << std::endl;
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  const Matrix &get_y() { return y_; }
  Matrix *get_mutable_dy() { return &dy_; }

 protected:
  int input_size_;
  int output_size_;
  Matrix Wxy_;
  Vector by_;

  Matrix dWxy_;
  Vector dby_;

  const Matrix *x_; // Input.
  Matrix y_; // Output.
  Matrix *dx_;
  Matrix dy_;
};

class SoftmaxOutputLayer : public Layer {
 public:
  SoftmaxOutputLayer() {}
  SoftmaxOutputLayer(int hidden_size,
                     int output_size) {
    hidden_size_ = hidden_size;
    output_size_ = output_size;
  }
  virtual ~SoftmaxOutputLayer() {}

  void CollectAllParameters(std::vector<Matrix*> *weights,
                            std::vector<Vector*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    Why_ = Matrix::Zero(output_size_, hidden_size_);
    by_ = Vector::Zero(output_size_);

    weights->push_back(&Why_);
    biases->push_back(&by_);

    weight_names->push_back("Why");
    bias_names->push_back("by");
  }

  double GetUniformInitializationLimit(Matrix *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff = 4.0; // Like in LOGISTIC.
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    dWhy_.setZero(output_size_, hidden_size_);
    dby_.setZero(output_size_);
  }

  void RunForward() {
    y_ = Why_ * (*h_) + by_;
    float logsum = LogSumExp(y_);
    p_ = (y_.array() - logsum).exp();
  }

  void RunBackward() {
    Vector dy = p_;
    dy[output_label_] -= 1.0; // Backprop into y (softmax grad).
    dWhy_.noalias() += dy * h_->transpose();
    dby_.noalias() += dy;
    (*dh_).noalias() += Why_.transpose() * dy; // Backprop into h. // CHECK +=!!!
  }

  void UpdateParameters(double learning_rate) {
    Why_ -= learning_rate * dWhy_;
    by_ -= learning_rate * dby_;
  }

  void set_output_label(int output_label) {
    output_label_ = output_label;
  }
  void set_h(const Vector &h) { h_ = &h; }
  void set_dh(Vector *dh) { dh_ = dh; }
  const Vector &get_y() { return y_; } // remove.
  const Vector &GetProbabilities() { return p_; }

 protected:
  int hidden_size_;
  int output_size_;

  Matrix Why_;
  Vector by_;

  Matrix dWhy_;
  Vector dby_;

  const Vector *h_; // Input.
  Vector y_; // Output.
  Vector p_; // Output.
  int output_label_; // Output.
  Vector *dh_;
};

class FeedforwardLayer : public Layer {
 public:
  FeedforwardLayer() {}
  FeedforwardLayer(int input_size,
                   int hidden_size) {
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
  }
  virtual ~FeedforwardLayer() {}

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    Wxh_ = Matrix::Zero(hidden_size_, input_size_);
    bh_ = Vector::Zero(hidden_size_);

    weights->push_back(&Wxh_);
    biases->push_back(&bh_);

    weight_names->push_back("Wxh");
    bias_names->push_back("bh");
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
    dWxh_.setZero(hidden_size_, input_size_);
    dbh_.setZero(hidden_size_);
  }

  virtual void RunForward() {
    EvaluateActivation(activation_function_,
		       Wxh_ * (*x_) + bh_,
		       &h_);
  }

  virtual void RunBackward() {
    Vector dhraw;
    DerivateActivation(activation_function_, h_, &dhraw);
    dhraw = dhraw.array() * dh_.array();
    dWxh_.noalias() += dhraw * (*x_).transpose();
    dbh_.noalias() += dhraw;
    *dx_ += Wxh_.transpose() * dhraw; // Backprop into x.
  }

  void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
  }

  void set_x(const Vector &x) { x_ = &x; }
  void set_dx(Vector *dx) { dx_ = dx; }
  const Vector &get_h() { return h_; }
  Vector *get_mutable_dh() { return &dh_; }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;

  Matrix Wxh_;
  Vector bh_;

  Matrix dWxh_;
  Vector dbh_;

  const Vector *x_; // Input.
  Vector h_; // Output.
  Vector *dx_;
  Vector dh_;
};

class MatrixFeedforwardLayer : public Layer {
 public:
  MatrixFeedforwardLayer() {}
  MatrixFeedforwardLayer(int input_size,
                   int hidden_size) {
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
  }
  virtual ~MatrixFeedforwardLayer() {}

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    Wxh_ = Matrix::Zero(hidden_size_, input_size_);
    bh_ = Vector::Zero(hidden_size_);

    weights->push_back(&Wxh_);
    biases->push_back(&bh_);

    weight_names->push_back("Wxh");
    bias_names->push_back("bh");
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
    dWxh_.setZero(hidden_size_, input_size_);
    dbh_.setZero(hidden_size_);
  }

  virtual void RunForward() {
    int length = (*x_).cols();
    h_.setZero(hidden_size_, length);
    for (int t = 0; t < length; ++t) {
      Matrix result;
      EvaluateActivation(activation_function_,
                         Wxh_ * (*x_).col(t) + bh_,
                         &result);
      h_.col(t) = result;
    }
  }

  virtual void RunBackward() {
    const Matrix &dy = dh_; // Just to avoid messing up the names.

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t); // Backprop into h.
      Matrix dhraw;
      DerivateActivation(activation_function_, h_.col(t), &dhraw);
      dhraw = dhraw.array() * dh.array();

      dWxh_.noalias() += dhraw * x_->col(t).transpose();
      dbh_.noalias() += dhraw;

      dx_->col(t) += Wxh_.transpose() * dhraw; // Backprop into x.
    }
  }

  void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  const Matrix &get_h() { return h_; }
  Matrix *get_mutable_dh() { return &dh_; }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;

  Matrix Wxh_;
  Vector bh_;

  Matrix dWxh_;
  Vector dbh_;

  const Matrix *x_; // Input.
  Matrix h_; // Output.
  Matrix *dx_;
  Matrix dh_;
};

class RNNLayer : public Layer {
 public:
  RNNLayer() {}
  RNNLayer(int input_size,
           int hidden_size) {
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
  }
  virtual ~RNNLayer() {}

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    Wxh_ = Matrix::Zero(hidden_size_, input_size_);
    Whh_ = Matrix::Zero(hidden_size_, hidden_size_);
    bh_ = Vector::Zero(hidden_size_);
    if (use_hidden_start_) {
      h0_ = Vector::Zero(hidden_size_);
    }

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
#if 0
    int length = (*x_).cols();
    h_.setZero(hidden_size_, length);
    Vector hprev = Vector::Zero(h_.rows());
    if (use_hidden_start_) hprev = h0_;
    for (int t = 0; t < length; ++t) {
      Matrix result;
      EvaluateActivation(activation_function_,
                         Wxh_ * (*x_).col(t) + bh_ + Whh_ * hprev,
                         &result);
      h_.col(t) = result;
      hprev = h_.col(t);
    }
#else
    int length = (*x_).cols();
    h_.setZero(hidden_size_, length);
    Matrix hraw = (Wxh_ * (*x_)).colwise() + bh_;
    Vector hprev = Vector::Zero(h_.rows());
    if (use_hidden_start_) hprev = h0_;
    Vector result;
    for (int t = 0; t < length; ++t) {
      EvaluateActivation(activation_function_,
                         hraw.col(t) + Whh_ * hprev,
                         &result);
      h_.col(t) = result;
      hprev = h_.col(t);
    }
#endif
  }

  virtual void RunBackward() {
#if 0
    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = dh_; // Just to avoid messing up the names.

    int length = dy.cols();
    // dx_->setZero(input_size_, length); // CHANGE THIS!!!
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t) + dhnext; // Backprop into h.
      Matrix dhraw;
      DerivateActivation(activation_function_, h_.col(t), &dhraw);
      dhraw = dhraw.array() * dh.array();

      dWxh_.noalias() += dhraw * x_->col(t).transpose();
      dbh_.noalias() += dhraw;
      if (t > 0) {
        dWhh_.noalias() += dhraw * h_.col(t-1).transpose();
      }
      dhnext.noalias() = Whh_.transpose() * dhraw;

      dx_->col(t) += Wxh_.transpose() * dhraw; // Backprop into x.
    }

    dh0_.noalias() += dhnext;
#else
    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = dh_; // Just to avoid messing up the names.

    Matrix dhraw;
    DerivateActivation(activation_function_, h_, &dhraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = Whh_.transpose() * dhraw.col(t);
    }

    (*dx_) += Wxh_.transpose() * dhraw; // Backprop into x.

    dWxh_.noalias() += dhraw * x_->transpose();
    dbh_.noalias() += dhraw.rowwise().sum();
    dWhh_.noalias() += dhraw.rightCols(length-1) *
      h_.leftCols(length-1).transpose();
    dh0_.noalias() += dhnext;
#endif
  }

  virtual void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
    Whh_ -= learning_rate * dWhh_;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dh0_;
    }
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  const Matrix &get_h() { return h_; }
  Matrix *get_mutable_dh() { return &dh_; }

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

  const Matrix *x_; // Input.
  Matrix h_; // Output.
  Matrix *dx_;
  Matrix dh_;
};

class GRULayer : public RNNLayer {
 public:
  GRULayer() {}
  GRULayer(int input_size,
           int hidden_size) {
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
  }
  virtual ~GRULayer() {}

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    RNNLayer::CollectAllParameters(weights, biases, weight_names, bias_names);

    Wxz_ = Matrix::Zero(hidden_size_, input_size_);
    Whz_ = Matrix::Zero(hidden_size_, hidden_size_);
    Wxr_ = Matrix::Zero(hidden_size_, input_size_);
    Whr_ = Matrix::Zero(hidden_size_, hidden_size_);
    bz_ = Vector::Zero(hidden_size_);
    br_ = Vector::Zero(hidden_size_);

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
    int length = (*x_).cols();
    z_.setZero(hidden_size_, length);
    r_.setZero(hidden_size_, length);
    hu_.setZero(hidden_size_, length);
    h_.setZero(hidden_size_, length);
    Matrix zraw = (Wxz_ * (*x_)).colwise() + bz_;
    Matrix rraw = (Wxr_ * (*x_)).colwise() + br_;
    Matrix hraw = (Wxh_ * (*x_)).colwise() + bh_;
    Vector hprev = Vector::Zero(h_.rows());
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
      h_.col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;
      hprev = h_.col(t);
    }
  }

  void RunBackward() {
    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = dh_; // Just to avoid messing up the names.

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
        hprev = Vector::Zero(h_.rows());
      } else {
        hprev = h_.col(t-1);
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

    (*dx_) += Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
      Wxh_.transpose() * dhuraw; // Backprop into x.

    dWxz_.noalias() += dzraw * x_->transpose();
    dbz_.noalias() += dzraw.rowwise().sum();
    dWxr_.noalias() += drraw * x_->transpose();
    dbr_.noalias() += drraw.rowwise().sum();
    dWxh_.noalias() += dhuraw * x_->transpose();
    dbh_.noalias() += dhuraw.rowwise().sum();

    dWhz_.noalias() += dzraw.rightCols(length-1) *
      h_.leftCols(length-1).transpose();
    dWhr_.noalias() += drraw.rightCols(length-1) *
      h_.leftCols(length-1).transpose();
    dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((r_.rightCols(length-1)).cwiseProduct(h_.leftCols(length-1))).transpose();

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
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    control_size_ = control_size;
    hidden_size_ = hidden_size;
    use_sparsemax_ = use_sparsemax;
  }
  virtual ~AttentionLayer() {}

  virtual void CollectAllParameters(std::vector<Matrix*> *weights,
                                    std::vector<Vector*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) {
    Wxz_ = Matrix::Zero(hidden_size_, input_size_);
    Wyz_ = Matrix::Zero(hidden_size_, control_size_);
    bz_ = Vector::Zero(hidden_size_);
    wzp_ = Vector::Zero(hidden_size_);

    weights->push_back(&Wxz_);
    weights->push_back(&Wyz_);

    biases->push_back(&bz_);
    biases->push_back(&wzp_); // Not really a bias, but it goes here.

    weight_names->push_back("Wxz");
    weight_names->push_back("Wyz");

    bias_names->push_back("bz");
    bias_names->push_back("wzp");
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
    dbz_.setZero(hidden_size_);
    dwzp_.setZero(hidden_size_);
  }

  virtual void RunForward() {
    int length = (*x_).cols();
    z_.setZero(hidden_size_, length);

    for (int t = 0; t < length; ++t) {
      Matrix result;
      EvaluateActivation(activation_function_,
                         Wxz_ * (*x_).col(t) + bz_ + Wyz_ * (*y_),
                         &result);
      z_.col(t) = result;
    }

    Vector v = z_.transpose() * wzp_;
    if (use_sparsemax_) {
      float tau;
      ProjectOntoSimplex(v, 1.0, &p_, &tau);
    } else {
      float logsum = LogSumExp(v);
      p_ = (v.array() - logsum).exp();
    }
    u_.noalias() = (*x_) * p_;
  }

  virtual void RunBackward() {
    int length = (*x_).cols();
    dp_.noalias() = (*x_).transpose() * du_;

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

    *dx_ += Wxz_.transpose() * dzraw;
    *dx_ += du_ * p_.transpose();
    *dy_ += Wyz_.transpose() * dzraw_sum;

    dwzp_ += z_ * Jdp;
    dWxz_.noalias() += dzraw * (*x_).transpose();
    dWyz_.noalias() += dzraw_sum * (*y_).transpose();
    dbz_.noalias() += dzraw_sum;
  }

  void UpdateParameters(double learning_rate) {
    Wxz_ -= learning_rate * dWxz_;
    Wyz_ -= learning_rate * dWyz_;
    dbz_ -= learning_rate * dbz_;
    wzp_ -= learning_rate * dwzp_;
  }

  void set_x(const Matrix &x) { x_ = &x; }
  void set_y(const Vector &y) { y_ = &y; }
  void set_dx(Matrix *dx) { dx_ = dx; }
  void set_dy(Vector *dy) { dy_ = dy; }
  const Vector &get_u() { return u_; }
  Vector *get_mutable_du() { return &du_; }

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  int control_size_;
  bool use_sparsemax_;

  Matrix Wxz_;
  Matrix Wyz_;
  Vector bz_;
  Vector wzp_;

  Matrix dWxz_;
  Matrix dWyz_;
  Vector dbz_;
  Vector dwzp_;

  const Matrix *x_; // Input.
  const Vector *y_; // Input (control).
  Vector u_; // Output.
  Matrix z_;
  Vector p_;

  Matrix *dx_;
  Vector *dy_;
  Matrix dz_;
  Vector dp_;
  Vector du_;
};

#endif /* LAYER_H_ */
