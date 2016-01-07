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

    bool read_from_file = false;
    if (read_from_file) {
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
      return;
    }

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

  virtual void ResetGradients() = 0;
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
    double logsum = LogSumExp(y_);
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
    dWhh_.setZero(hidden_size_, hidden_size_);
    if (use_hidden_start_) {
      dh0_.setZero(hidden_size_);
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
  }

  void UpdateParameters(double learning_rate) {
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

class AttentionLayer : public Layer {
 public:
  AttentionLayer() {}
  AttentionLayer(int input_size,
                 int control_size,
                 int hidden_size) {
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    control_size_ = control_size;
    hidden_size_ = hidden_size;
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
    double logsum = LogSumExp(v);
    //p_.noalias() = (v.array() - logsum).exp();
    p_ = (v.array() - logsum).exp();
    u_.noalias() = (*x_) * p_;
  }

  virtual void RunBackward() {
    int length = (*x_).cols();
    dp_.noalias() = (*x_).transpose() * du_;

    // Compute Jsoftmax * dp_.
    // Jsoftmax = diag(p_) - p_*p_.transpose().
    // Jsoftmax * dp_ = p_.array() * dp_.array() - p_* (p_.transpose() * dp_).
    //                = p_.array() * (dp_ - val).array(),
    // where val = p_.transpose() * dp_.
    float val = p_.transpose() * dp_;
    Vector Jdp = p_.array() * (dp_.array() - val);
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
