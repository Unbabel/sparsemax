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

  virtual void RunForward() = 0;
  virtual void RunBackward() = 0;
  virtual void UpdateParameters(double learning_rate) = 0;
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

  void SetFixedEmbeddings(const Matrix &fixed_embeddings,
                          const std::vector<int> &word_ids) {
    for (int i = 0; i < fixed_embeddings.cols(); i++) {
      int wid = word_ids[i];
      E_.col(wid) = fixed_embeddings.col(i);
      updatable_[wid] = false;
    }
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

  void RunForward() {
    y_ = (Wxy_ * (*x_)).colwise() + by_;
  }

  void RunBackward() {
    *dx_ += Wxy_.transpose() * dy_; // check +=
    dWxy_ = dy_ * (*x_).transpose();
    dby_ = dy_.rowwise().sum();
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

  void RunForward() {
    y_ = Why_ * (*h_) + by_;
    double logsum = LogSumExp(y_);
    p_ = (y_.array() - logsum).exp();
  }

  void RunBackward() {
    dWhy_ = Matrix::Zero(Why_.rows(), Why_.cols()); // CHANGE THIS!!!
    dby_ = Vector::Zero(Why_.rows()); // CHANGE THIS!!!

    //std::cout << p_[0] << " " << p_[1] << " " << p_[2] << std::endl;

    Vector dy = p_;
    dy[output_label_] -= 1.0; // Backprop into y (softmax grad).
    dWhy_ += dy * h_->transpose();
    dby_ += dy;
    *dh_ += Why_.transpose() * dy; // Backprop into h. // CHECK +=!!!
  }

  void UpdateParameters(double learning_rate) {
    Why_ -= learning_rate * dWhy_;
    by_ -= learning_rate * dby_;

    //std::cout << "params output layer" << std::endl;
    //std::cout << Why_(0,0) << std::endl; // WRONG!!!
    //std::cout << by_(0,0) << std::endl;
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

class RNNLayer : public Layer {
 public:
  RNNLayer() {}
  RNNLayer(int input_size,
           int hidden_size) {
    activation_function_ = ActivationFunctions::LOGISTIC; // Change to TANH?
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

  virtual void RunForward() {
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
  }

  virtual void RunBackward() {
    dWhh_ = Matrix::Zero(Whh_.rows(), Whh_.cols()); // CHANGE THIS!!!
    dWxh_ = Matrix::Zero(Wxh_.rows(), Wxh_.cols()); // CHANGE THIS!!!
    dbh_ = Vector::Zero(Whh_.rows()); // CHANGE THIS!!!
    Vector dhnext = Vector::Zero(Whh_.rows());
    const Matrix &dy = dh_;

    int length = dy.cols();
    dx_->setZero(input_size_, length); // CHANGE THIS!!!
    for (int t = length - 1; t >= 0; --t) {
      Vector dh = dy.col(t) + dhnext; // Backprop into h.
      Matrix dhraw;
      DerivateActivation(activation_function_, h_.col(t), &dhraw);
      dhraw = dhraw.array() * dh.array();

      dWxh_ += dhraw * x_->col(t).transpose();
      dbh_ += dhraw;
      if (t > 0) {
        dWhh_ += dhraw * h_.col(t-1).transpose();
      }
      dhnext.noalias() = Whh_.transpose() * dhraw;

      dx_->col(t) = Wxh_.transpose() * dhraw; // Backprop into x.
    }

    dh0_ = dhnext;
  }

  void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
    Whh_ -= learning_rate * dWhh_;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dh0_;
    }
    //std::cout << "params RNN layer" << std::endl;
    //std::cout << Wxh_(0,0) << std::endl;
    //std::cout << Whh_(0,0) << std::endl;
    //std::cout << bh_(0,0) << std::endl;
    //std::cout << h0_(0,0) << std::endl;
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

#endif /* LAYER_H_ */
