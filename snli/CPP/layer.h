#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"
#include "nn_utils.h"
#include "snli_data.h"

//typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixTpl<Real>;
//typedef Eigen::Vector<Real, Eigen::Dynamic, 1> VectorTpl<Real>;

template<typename Real> class Layer {
 public:
  Layer() {}
  ~Layer() {
    for (int i = 0; i < first_bias_moments_.size(); ++i) {
      delete first_bias_moments_[i];
      delete second_bias_moments_[i];
    }
    for (int i = 0; i < first_weight_moments_.size(); ++i) {
      delete first_weight_moments_[i];
      delete second_weight_moments_[i];
    }
  }

  virtual void ResetParameters() = 0;

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
                                    std::vector<std::string> *weight_names,
                                    std::vector<std::string> *bias_names) = 0;

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) = 0;

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) = 0;

  virtual void UpdateParameters(double learning_rate,
                                double regularization_constant) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];
      (*db) += (regularization_constant * (*b));
      *b -= learning_rate * (*db);
    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];
      (*dW) += (regularization_constant * (*W));
      *W -= learning_rate * (*dW);
    }
  }

  void InitializeADAM(double beta1, double beta2, double epsilon) {
    beta1_ = beta1;
    beta2_ = beta2;
    epsilon_ = epsilon;
    iteration_number_ = 0;

    std::vector<Matrix<Real>*> weights;
    std::vector<Vector<Real>*> biases;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);

    first_bias_moments_.resize(biases.size());
    second_bias_moments_.resize(biases.size());
    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      first_bias_moments_[i] = new Vector<Real>;
      first_bias_moments_[i]->setZero(b->size());
      second_bias_moments_[i] = new Vector<Real>;
      second_bias_moments_[i]->setZero(b->size());
    }

    first_weight_moments_.resize(weights.size());
    second_weight_moments_.resize(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      first_weight_moments_[i] = new Matrix<Real>;
      first_weight_moments_[i]->setZero(W->rows(), W->cols());
      second_weight_moments_[i] = new Matrix<Real>;
      second_weight_moments_[i]->setZero(W->rows(), W->cols());
    }
  }

  void UpdateParametersADAM(double learning_rate,
                            double regularization_constant) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    bool adagrad = false;
    double stepsize = learning_rate *
      sqrt(1.0 - pow(beta2_, iteration_number_ + 1)) /
      (1.0 - pow(beta1_, iteration_number_ + 1));

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];

      (*db) += (regularization_constant * (*b));

      auto mb = first_bias_moments_[i];
      auto vb = second_bias_moments_[i];

      if (adagrad) {
        *vb = vb->array() + (db->array() * db->array());
        *b = b->array() - learning_rate * db->array() /
          (epsilon_ + vb->array().sqrt());
      } else {
        *mb = beta1_ * (*mb) + (1.0 - beta1_) * (*db);
        *vb = beta2_ * vb->array() +
          (1.0 - beta2_) * (db->array() * db->array());
        *b = b->array() - stepsize * mb->array()
          / (epsilon_ + vb->array().sqrt());
      }
    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];

      (*dW) += (regularization_constant * (*W));

      auto mW = first_weight_moments_[i];
      auto vW = second_weight_moments_[i];

      if (adagrad) {
        *vW = vW->array() + (dW->array() * dW->array());
        *W = W->array() - learning_rate * dW->array() /
          (epsilon_ + vW->array().sqrt());
      } else {
        *mW = beta1_ * (*mW) + (1.0 - beta1_) * (*dW);
        *vW = beta2_ * vW->array() +
          (1.0 - beta2_) * (dW->array() * dW->array());
        *W = W->array() - stepsize * mW->array()
          / (epsilon_ + vW->array().sqrt());
      }
    }

    ++iteration_number_;
  }

  void InitializeParameters() {
    std::vector<Matrix<Real>*> weights;
    std::vector<Vector<Real>*> biases;
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
    std::vector<Matrix<Real>*> weights;
    std::vector<Vector<Real>*> biases;
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
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
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
        std::cout << name << "(" << r << "/" << W->size() << ")" << ": "
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
  //virtual void UpdateParameters(double learning_rate) = 0;

  int GetNumInputs() const { return inputs_.size(); }
  void SetNumInputs(int n) {
    inputs_.resize(n);
    input_derivatives_.resize(n);
  }
  void SetInput(int i, const Matrix<Real> &input) { inputs_[i] = &input; }
  const Matrix<Real> &GetOutput() const { return output_; }
  void SetInputDerivative(int i, Matrix<Real> *input_derivative) {
    input_derivatives_[i] = input_derivative;
  }
  Matrix<Real> *GetOutputDerivative() { return &output_derivative_; }

 protected:
  const Matrix<Real> &GetInput() {
    assert(GetNumInputs() == 1);
    return *inputs_[0];
  }
  Matrix<Real> *GetInputDerivative() {
    assert(GetNumInputs() == 1);
    return input_derivatives_[0];
  }

 protected:
  std::string name_;
  std::vector<const Matrix<Real>*> inputs_;
  Matrix<Real> output_;
  std::vector<Matrix<Real>*> input_derivatives_;
  Matrix<Real> output_derivative_;

  // ADAM parameters.
  double beta1_;
  double beta2_;
  double epsilon_;
  int iteration_number_;
  std::vector<Matrix<Real>*> first_weight_moments_;
  std::vector<Vector<Real>*> first_bias_moments_;
  std::vector<Matrix<Real>*> second_weight_moments_;
  std::vector<Vector<Real>*> second_bias_moments_;
};

template<typename Real> class AverageLayer : public Layer<Real> {
 public:
  AverageLayer() { this->name_ = "Average"; }
  virtual ~AverageLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix<Real> *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    this->output_ = x.rowwise().sum() / static_cast<double>(x.cols());
  }

  void RunBackward() {
    //std::cout << "tmp1" << std::endl;
    Matrix<Real> *dx = this->GetInputDerivative();
    assert(this->output_derivative_.cols() == 1);
    dx->colwise() += this->output_derivative_.col(0) /
      static_cast<double>(dx->cols());
    //std::cout << "tmp2" << std::endl;
  }
};

template<typename Real> class SelectorLayer : public Layer<Real> {
 public:
  SelectorLayer() { this->name_ = "Selector"; }
  virtual ~SelectorLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix<Real> *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    this->output_ = x.block(first_row_, first_column_, num_rows_, num_columns_);
  }

  void RunBackward() {
    Matrix<Real> *dx = this->GetInputDerivative();
    (*dx).block(first_row_, first_column_, num_rows_, num_columns_) +=
      this->output_derivative_;
  }

#if 0
  void UpdateParameters(double learning_rate) {}
#endif

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

template<typename Real> class ConcatenatorLayer : public Layer<Real> {
 public:
  ConcatenatorLayer() { this->name_ = "Concatenator"; }
  virtual ~ConcatenatorLayer() {}

  void ResetParameters() {}

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {}

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {}

  double GetUniformInitializationLimit(Matrix<Real> *W) { return 0.0; }

  void ResetGradients() {}
  void RunForward() {
    int num_rows = 0;
    int num_columns = this->inputs_[0]->cols();
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      assert(this->inputs_[i]->cols() == num_columns);
      num_rows += this->inputs_[i]->rows();
    }
    this->output_.setZero(num_rows, num_columns);
    int start = 0;
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      this->output_.block(start, 0, this->inputs_[i]->rows(), num_columns) =
        *(this->inputs_[i]);
      start += this->inputs_[i]->rows();
    }
  }

  void RunBackward() {
    int num_columns = this->output_derivative_.cols();
    int start = 0;
    for (int i = 0; i < this->GetNumInputs(); ++i) {
      *(this->input_derivatives_[i]) +=
        this->output_derivative_.block(start, 0,
                                       this->input_derivatives_[i]->rows(),
                                       num_columns);
      start += this->inputs_[i]->rows();
    }
  }

#if 0
  void UpdateParameters(double learning_rate) {}
#endif
};

template<typename Real> class LookupLayer : public Layer<Real> {
 public:
  LookupLayer(int num_words, int embedding_dimension) {
    this->name_ = "Lookup";
    num_words_ = num_words;
    embedding_dimension_ = embedding_dimension;
    updatable_.assign(num_words, true);
    //std::cout << "Num words = " << num_words_ << std::endl;
  }

  virtual ~LookupLayer() {}

  int embedding_dimension() { return embedding_dimension_; }

  void ResetParameters() {
    E_ = Matrix<Real>::Zero(embedding_dimension_, num_words_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&E_);
    weight_names->push_back("embeddings");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    // This layer does not store full derivatives of the embeddings.
    assert(false);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
    return 0.05;
  }

  void SetFixedEmbeddings(const Matrix<Real> &fixed_embeddings,
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
    this->output_.setZero(embedding_dimension_, input_sequence_->size());
    assert(this->output_.rows() == embedding_dimension_ &&
           this->output_.cols() == input_sequence_->size());
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      assert(wid >= 0 && wid < E_.cols());
      this->output_.col(t) = E_.col(wid);
    }
  }

  void RunBackward() {}

  // NOTE: this is not supporting mini-batch!!!
  void UpdateParameters(double learning_rate, double regularization_constant) {
    // Update word embeddings.
    assert(regularization_constant == 0.0);
    for (int t = 0; t < input_sequence_->size(); ++t) {
      int wid = (*input_sequence_)[t].wid();
      if (!updatable_[wid]) continue;
      E_.col(wid) -= learning_rate * this->output_derivative_.col(t);
    }
  }

  void set_input_sequence(const std::vector<Input> &input_sequence) {
    input_sequence_ = &input_sequence;
  }

 public:
  int num_words_;
  int embedding_dimension_;
  Matrix<Real> E_;
  std::vector<bool> updatable_;

  const std::vector<Input> *input_sequence_; // Input.
};

template<typename Real> class LinearLayer : public Layer<Real> {
 public:
  LinearLayer(int input_size, int output_size) {
    this->name_ = "Linear";
    input_size_ = input_size;
    output_size_ = output_size;
  }

  virtual ~LinearLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Wxy_ = Matrix<Real>::Zero(output_size_, input_size_);
    by_ = Vector<Real>::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Wxy_);
    weight_names->push_back("linear_weights");

    biases->push_back(&by_);
    bias_names->push_back("linear_bias");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxy_);
    bias_derivatives->push_back(&dby_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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
    const Matrix<Real> &x = this->GetInput();
    this->output_ = (Wxy_ * x).colwise() + by_;
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();
    (*dx).noalias() += Wxy_.transpose() * this->output_derivative_;
    dWxy_.noalias() += this->output_derivative_ * x.transpose();
    dby_.noalias() += this->output_derivative_.rowwise().sum();
  }

#if 0
  void UpdateParameters(double learning_rate) {
    Wxy_ -= learning_rate * dWxy_;
    by_ -= learning_rate * dby_;
  }
#endif

 protected:
  int input_size_;
  int output_size_;
  Matrix<Real> Wxy_;
  Vector<Real> by_;

  Matrix<Real> dWxy_;
  Vector<Real> dby_;
};

template<typename Real> class SoftmaxOutputLayer : public Layer<Real> {
 public:
  SoftmaxOutputLayer() {}
  SoftmaxOutputLayer(int input_size,
                     int output_size) {
    this->name_ = "SoftmaxOutput";
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~SoftmaxOutputLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Why_ = Matrix<Real>::Zero(output_size_, input_size_);
    by_ = Vector<Real>::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Why_);
    biases->push_back(&by_);

    weight_names->push_back("Why");
    bias_names->push_back("by");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWhy_);
    bias_derivatives->push_back(&dby_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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
    const Matrix<Real> &h = this->GetInput();
    assert(h.cols() == 1);
    Vector<Real> y = Why_ * h + by_;
    Real logsum = LogSumExp(y);
    this->output_ = (y.array() - logsum).exp(); // This is the probability vector.
  }

  void RunBackward() {
    const Matrix<Real> &h = this->GetInput();
    assert(h.cols() == 1);
    Matrix<Real> *dh = this->GetInputDerivative();
    assert(dh->cols() == 1);

    Vector<Real> dy = this->output_;
    dy[output_label_] -= 1.0; // Backprop into y (softmax grad).
    dWhy_.noalias() += dy * h.transpose();
    dby_.noalias() += dy;
    (*dh).noalias() += Why_.transpose() * dy; // Backprop into h.
  }

#if 0
  void UpdateParameters(double learning_rate) {
    Why_ -= learning_rate * dWhy_;
    by_ -= learning_rate * dby_;
  }
#endif

  int output_label() { return output_label_; }
  void set_output_label(int output_label) {
    output_label_ = output_label;
  }

 protected:
  int input_size_;
  int output_size_;

  Matrix<Real> Why_;
  Vector<Real> by_;

  Matrix<Real> dWhy_;
  Vector<Real> dby_;

  int output_label_; // Output.
};

template<typename Real> class FeedforwardLayer : public Layer<Real> {
 public:
  FeedforwardLayer() {}
  FeedforwardLayer(int input_size,
                   int output_size) {
    this->name_ = "Feedforward";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    output_size_ = output_size;
  }
  virtual ~FeedforwardLayer() {}

  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }

  void ResetParameters() {
    Wxh_ = Matrix<Real>::Zero(output_size_, input_size_);
    bh_ = Vector<Real>::Zero(output_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    weights->push_back(&Wxh_);
    biases->push_back(&bh_);

    weight_names->push_back("Wxh");
    bias_names->push_back("bh");
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxh_);
    bias_derivatives->push_back(&dbh_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> tmp = (Wxh_ * x).colwise() + bh_;
    EvaluateActivation(activation_function_,
                       tmp,
                       &(this->output_));
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();
    Matrix<Real> dhraw;
    DerivateActivation(activation_function_, this->output_, &dhraw);
    dhraw = dhraw.array() * this->output_derivative_.array();
    dWxh_.noalias() += dhraw * x.transpose();
    //std::cout << dbh_.size() << " " << dhraw.size() << std::endl;
    dbh_.noalias() += dhraw.rowwise().sum();
    *dx += Wxh_.transpose() * dhraw; // Backprop into x.
  }

#if 0
  void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
  }
#endif

 protected:
  int activation_function_;
  int output_size_;
  int input_size_;

  Matrix<Real> Wxh_;
  Vector<Real> bh_;

  Matrix<Real> dWxh_;
  Vector<Real> dbh_;
};

template<typename Real> class RNNLayer : public Layer<Real> {
 public:
  RNNLayer() {}
  RNNLayer(int input_size,
           int hidden_size) {
    this->name_ = "RNN";
    activation_function_ = ActivationFunctions::TANH;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    use_hidden_start_ = true;
  }
  virtual ~RNNLayer() {}

  int input_size() const { return input_size_; }
  int hidden_size() const { return hidden_size_; }

  virtual void ResetParameters() {
    Wxh_ = Matrix<Real>::Zero(hidden_size_, input_size_);
    Whh_ = Matrix<Real>::Zero(hidden_size_, hidden_size_);
    bh_ = Vector<Real>::Zero(hidden_size_);
    if (use_hidden_start_) {
      h0_ = Vector<Real>::Zero(hidden_size_);
    }
  }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                                    std::vector<Vector<Real>*> *biases,
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
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxh_);
    weight_derivatives->push_back(&dWhh_);
    bias_derivatives->push_back(&dbh_);
    if (use_hidden_start_) {
      bias_derivatives->push_back(&dh0_); // Not really a bias, but goes here.
    }
  }

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) {
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
    const Matrix<Real> &x = this->GetInput();
    int length = x.cols();
    this->output_.setZero(hidden_size_, length);
    Matrix<Real> hraw = (Wxh_ * x).colwise() + bh_;
    Vector<Real> hprev = Vector<Real>::Zero(this->output_.rows());
    if (use_hidden_start_) hprev = h0_;
    Vector<Real> result;
    for (int t = 0; t < length; ++t) {
      Vector<Real> tmp = hraw.col(t) + Whh_ * hprev;
      EvaluateActivation(activation_function_,
                         tmp,
                         &result);
      this->output_.col(t) = result;
      hprev = this->output_.col(t);
    }
  }

  virtual void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(Whh_.rows());
    const Matrix<Real> &dy = this->output_derivative_;

    Matrix<Real> dhraw;
    DerivateActivation(activation_function_, this->output_, &dhraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = dy.col(t) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = Whh_.transpose() * dhraw.col(t);
    }

    *dx += Wxh_.transpose() * dhraw; // Backprop into x.

    dWxh_.noalias() += dhraw * x.transpose();
    dbh_.noalias() += dhraw.rowwise().sum();
    dWhh_.noalias() += dhraw.rightCols(length-1) *
      this->output_.leftCols(length-1).transpose();
    dh0_.noalias() += dhnext;
  }

#if 0
  virtual void UpdateParameters(double learning_rate) {
    Wxh_ -= learning_rate * dWxh_;
    bh_ -= learning_rate * dbh_;
    Whh_ -= learning_rate * dWhh_;
    if (use_hidden_start_) {
      h0_ -= learning_rate * dh0_;
    }
  }
#endif

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  bool use_hidden_start_;

  Matrix<Real> Wxh_;
  Matrix<Real> Whh_;
  Vector<Real> bh_;
  Vector<Real> h0_;

  Matrix<Real> dWxh_;
  Matrix<Real> dWhh_;
  Vector<Real> dbh_;
  Vector<Real> dh0_;
};

template<typename Real> class BiRNNLayer : public RNNLayer<Real> {
 public:
  BiRNNLayer() {}
  BiRNNLayer(int input_size,
           int hidden_size) {
    this->name_ = "BiRNN";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
  }
  virtual ~BiRNNLayer() {}

  void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxl_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wll_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bl_ = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      l0_ = Vector<Real>::Zero(this->hidden_size_);
    }
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
			    std::vector<Vector<Real>*> *biases,
			    std::vector<std::string> *weight_names,
			    std::vector<std::string> *bias_names) {
    RNNLayer<Real>::CollectAllParameters(weights, biases, weight_names,
					 bias_names);

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);

    biases->push_back(&bl_);
    if (this->use_hidden_start_) {
      biases->push_back(&l0_); // Not really a bias, but it goes here.
    }

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");

    bias_names->push_back("bl");
    if (this->use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    RNNLayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxl_);
    weight_derivatives->push_back(&dWll_);
    bias_derivatives->push_back(&dbl_);
    if (this->use_hidden_start_) {
      bias_derivatives->push_back(&dl0_); // Not really a bias, but goes here.
    }
  }

  void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxl_.setZero(this->hidden_size_, this->input_size_);
    dbl_.setZero(this->hidden_size_);
    dWll_.setZero(this->hidden_size_, this->hidden_size_);
    if (this->use_hidden_start_) {
      dl0_.setZero(this->hidden_size_);
    }
  }

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();
    int length = x.cols();
    this->output_.setZero(2*this->hidden_size_, length);
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Matrix<Real> lraw = (Wxl_ * x).colwise() + bl_;
    Vector<Real> hprev = Vector<Real>::Zero(this->hidden_size_);
    Vector<Real> lnext = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      hprev = this->h0_;
      lnext = l0_;
    }
    Vector<Real> result;
    for (int t = 0; t < length; ++t) {
      Vector<Real> tmp = hraw.col(t) + this->Whh_ * hprev;
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->output_.block(0, t, this->hidden_size_, 1) = result;
      hprev = result;
    }
    for (int t = length-1; t >= 0; --t) {
      Vector<Real> tmp = lraw.col(t) + Wll_ * lnext;
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->output_.block(this->hidden_size_, t, this->hidden_size_, 1) = result;
      lnext = result;
    }
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    Vector<Real> dlprev = Vector<Real>::Zero(Wll_.rows());

    int length = this->output_.cols();
    Matrix<Real> result;
    DerivateActivation(this->activation_function_, this->output_, &result);
    Matrix<Real> dhraw = result.block(0, 0, this->hidden_size_, length);
    Matrix<Real> dlraw = result.block(this->hidden_size_, 0, this->hidden_size_, length);

    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = this->output_derivative_.block(0, t, this->hidden_size_, 1) + dhnext; // Backprop into h.
      dhraw.col(t) = dhraw.col(t).array() * dh.array();
      dhnext.noalias() = this->Whh_.transpose() * dhraw.col(t);
    }

    for (int t = 0; t < length; ++t) {
      Vector<Real> dl = this->output_derivative_.block(this->hidden_size_, t, this->hidden_size_, 1) + dlprev; // Backprop into h.
      dlraw.col(t) = dlraw.col(t).array() * dl.array();
      dlprev.noalias() = Wll_.transpose() * dlraw.col(t);
    }

    *dx += this->Wxh_.transpose() * dhraw; // Backprop into x.
    *dx += Wxl_.transpose() * dlraw; // Backprop into x.

    this->dWxh_.noalias() += dhraw * x.transpose();
    this->dbh_.noalias() += dhraw.rowwise().sum();
    this->dWhh_.noalias() += dhraw.rightCols(length-1) *
      this->output_.block(0, 0, this->hidden_size_, length-1).transpose();
    this->dh0_.noalias() += dhnext;

    dWxl_.noalias() += dlraw * x.transpose();
    dbl_.noalias() += dlraw.rowwise().sum();
    dWll_.noalias() += dlraw.leftCols(length-1) *
      this->output_.block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dl0_.noalias() += dlprev;
  }

 protected:
  Matrix<Real> Wxl_;
  Matrix<Real> Wll_;
  Vector<Real> bl_;
  Vector<Real> l0_;

  Matrix<Real> dWxl_;
  Matrix<Real> dWll_;
  Vector<Real> dbl_;
  Vector<Real> dl0_;
};

template<typename Real> class GRULayer : public RNNLayer<Real> {
 public:
  GRULayer() {}
  GRULayer(int input_size,
           int hidden_size) {
    this->name_ = "GRU";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
  }
  virtual ~GRULayer() {}

  virtual void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxz_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whz_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxr_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whr_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bz_ = Vector<Real>::Zero(this->hidden_size_);
    br_ = Vector<Real>::Zero(this->hidden_size_);
  }

  virtual void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
				    std::vector<Vector<Real>*> *biases,
				    std::vector<std::string> *weight_names,
				    std::vector<std::string> *bias_names) {
    RNNLayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);

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

  virtual void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    RNNLayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWhz_);
    weight_derivatives->push_back(&dWxr_);
    weight_derivatives->push_back(&dWhr_);
    bias_derivatives->push_back(&dbz_);
    bias_derivatives->push_back(&dbr_);
  }

  virtual double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (this->activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &Wxz_ || W == &Whz_ || W == &Wxr_ || W == &Whr_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  virtual void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxz_.setZero(this->hidden_size_, this->input_size_);
    dWhz_.setZero(this->hidden_size_, this->hidden_size_);
    dWxr_.setZero(this->hidden_size_, this->input_size_);
    dWhr_.setZero(this->hidden_size_, this->hidden_size_);
    dbz_.setZero(this->hidden_size_);
    dbr_.setZero(this->hidden_size_);
  }

  virtual void RunForward() {
    const Matrix<Real> &x = this->GetInput();

    int length = x.cols();
    z_.setZero(this->hidden_size_, length);
    r_.setZero(this->hidden_size_, length);
    hu_.setZero(this->hidden_size_, length);
    this->output_.setZero(this->hidden_size_, length);
    Matrix<Real> zraw = (Wxz_ * x).colwise() + bz_;
    Matrix<Real> rraw = (Wxr_ * x).colwise() + br_;
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Vector<Real> hprev = Vector<Real>::Zero(this->output_.rows());
    if (this->use_hidden_start_) hprev = this->h0_;
    Vector<Real> result;
    Vector<Real> tmp;
    for (int t = 0; t < length; ++t) {
      tmp = zraw.col(t) + Whz_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      z_.col(t) = result;

      tmp = rraw.col(t) + Whr_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      r_.col(t) = result;

      tmp = hraw.col(t) + this->Whh_ * r_.col(t).cwiseProduct(hprev);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      hu_.col(t) = result;
      this->output_.col(t) = z_.col(t).cwiseProduct(hu_.col(t) - hprev) + hprev;
      hprev = this->output_.col(t);
    }
  }

  virtual void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    const Matrix<Real> &dy = this->output_derivative_;

    Matrix<Real> dhuraw;
    DerivateActivation(this->activation_function_, hu_, &dhuraw);
    Matrix<Real> dzraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, z_, &dzraw);
    Matrix<Real> drraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, r_, &drraw);

    int length = dy.cols();
    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = dy.col(t) + dhnext; // Backprop into h.
      Vector<Real> dhu = z_.col(t).cwiseProduct(dh);

      dhuraw.col(t) = dhuraw.col(t).cwiseProduct(dhu);
      Vector<Real> hprev;
      if (t == 0) {
        hprev = Vector<Real>::Zero(this->output_.rows());
      } else {
        hprev = this->output_.col(t-1);
      }

      Vector<Real> dq = this->Whh_.transpose() * dhuraw.col(t);
      Vector<Real> dz = (hu_.col(t) - hprev).cwiseProduct(dh);
      Vector<Real> dr = hprev.cwiseProduct(dq);

      dzraw.col(t) = dzraw.col(t).cwiseProduct(dz);
      drraw.col(t) = drraw.col(t).cwiseProduct(dr);

      dhnext.noalias() =
        Whz_.transpose() * dzraw.col(t) +
        Whr_.transpose() * drraw.col(t) +
        r_.col(t).cwiseProduct(dq) +
        (1.0 - z_.col(t).array()).matrix().cwiseProduct(dh);
    }

    *dx += Wxz_.transpose() * dzraw + Wxr_.transpose() * drraw +
      this->Wxh_.transpose() * dhuraw; // Backprop into x.

    dWxz_.noalias() += dzraw * x.transpose();
    dbz_.noalias() += dzraw.rowwise().sum();
    dWxr_.noalias() += drraw * x.transpose();
    dbr_.noalias() += drraw.rowwise().sum();
    this->dWxh_.noalias() += dhuraw * x.transpose();
    this->dbh_.noalias() += dhuraw.rowwise().sum();

    dWhz_.noalias() += dzraw.rightCols(length-1) *
      this->output_.leftCols(length-1).transpose();
    dWhr_.noalias() += drraw.rightCols(length-1) *
      this->output_.leftCols(length-1).transpose();
    this->dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((r_.rightCols(length-1)).cwiseProduct(this->output_.leftCols(length-1))).
      transpose();

    this->dh0_.noalias() += dhnext;
  }

#if 0
  void UpdateParameters(double learning_rate) {
    RNNLayer<Real>::UpdateParameters(learning_rate);
    Wxz_ -= learning_rate * dWxz_;
    Whz_ -= learning_rate * dWhz_;
    Wxr_ -= learning_rate * dWxr_;
    Whr_ -= learning_rate * dWhr_;
    bz_ -= learning_rate * dbz_;
    br_ -= learning_rate * dbr_;
  }
#endif

 protected:
  Matrix<Real> Wxz_;
  Matrix<Real> Whz_;
  Matrix<Real> Wxr_;
  Matrix<Real> Whr_;
  Vector<Real> bz_;
  Vector<Real> br_;

  Matrix<Real> dWxz_;
  Matrix<Real> dWhz_;
  Matrix<Real> dWxr_;
  Matrix<Real> dWhr_;
  Vector<Real> dbz_;
  Vector<Real> dbr_;

  Matrix<Real> z_;
  Matrix<Real> r_;
  Matrix<Real> hu_;
};

template<typename Real> class BiGRULayer : public GRULayer<Real> {
 public:
  BiGRULayer() {}
  BiGRULayer(int input_size,
           int hidden_size) {
    this->name_ = "BiGRU";
    this->activation_function_ = ActivationFunctions::TANH;
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->use_hidden_start_ = true;
  }
  virtual ~BiGRULayer() {}

  void ResetParameters() {
    GRULayer<Real>::ResetParameters();

    Wxl_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wll_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxz_r_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wlz_r_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxr_r_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Wlr_r_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bl_ = Vector<Real>::Zero(this->hidden_size_);
    bz_r_ = Vector<Real>::Zero(this->hidden_size_);
    br_r_ = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      l0_ = Vector<Real>::Zero(this->hidden_size_);
    }
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
                            std::vector<std::string> *weight_names,
                            std::vector<std::string> *bias_names) {
    GRULayer<Real>::CollectAllParameters(weights, biases, weight_names,
                                         bias_names);

    weights->push_back(&Wxl_);
    weights->push_back(&Wll_);
    weights->push_back(&Wxz_r_);
    weights->push_back(&Wlz_r_);
    weights->push_back(&Wxr_r_);
    weights->push_back(&Wlr_r_);

    biases->push_back(&bl_);
    biases->push_back(&bz_r_);
    biases->push_back(&br_r_);
    if (this->use_hidden_start_) {
      biases->push_back(&l0_);
    }

    weight_names->push_back("Wxl");
    weight_names->push_back("Wll");
    weight_names->push_back("Wxz_r");
    weight_names->push_back("Wlz_r");
    weight_names->push_back("Wxr_r");
    weight_names->push_back("Wlr_r");

    bias_names->push_back("bl");
    bias_names->push_back("bz_r");
    bias_names->push_back("br_r");
    if (this->use_hidden_start_) {
      bias_names->push_back("l0");
    }
  }

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    GRULayer<Real>::CollectAllParameterDerivatives(weight_derivatives,
                                                   bias_derivatives);
    weight_derivatives->push_back(&dWxl_);
    weight_derivatives->push_back(&dWll_);
    weight_derivatives->push_back(&dWxz_r_);
    weight_derivatives->push_back(&dWlz_r_);
    weight_derivatives->push_back(&dWxr_r_);
    weight_derivatives->push_back(&dWlr_r_);
    bias_derivatives->push_back(&dbl_);
    bias_derivatives->push_back(&dbz_r_);
    bias_derivatives->push_back(&dbr_r_);
    if (this->use_hidden_start_) {
      bias_derivatives->push_back(&dl0_);
    }
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
    int num_outputs = W->rows();
    int num_inputs = W->cols();
    double coeff;
    // Weights controlling gates have logistic activations.
    if (this->activation_function_ == ActivationFunctions::LOGISTIC ||
        W == &this->Wxz_ || W == &this->Whz_ || W == &this->Wxr_ || W == &this->Whr_ ||
        W == &this->Wxz_r_ || W == &this->Wlz_r_ || W == &this->Wxr_r_ || W == &this->Wlr_r_) {
      coeff = 4.0;
    } else {
      coeff = 1.0;
    }
    return coeff * sqrt(6.0 / (num_inputs + num_outputs));
  }

  void ResetGradients() {
    GRULayer<Real>::ResetGradients();
    dWxl_.setZero(this->hidden_size_, this->input_size_);
    dWll_.setZero(this->hidden_size_, this->hidden_size_);
    dWxz_r_.setZero(this->hidden_size_, this->input_size_);
    dWlz_r_.setZero(this->hidden_size_, this->hidden_size_);
    dWxr_r_.setZero(this->hidden_size_, this->input_size_);
    dWlr_r_.setZero(this->hidden_size_, this->hidden_size_);
    dbl_.setZero(this->hidden_size_);
    dbz_r_.setZero(this->hidden_size_);
    dbr_r_.setZero(this->hidden_size_);
    if (this->use_hidden_start_) {
      dl0_.setZero(this->hidden_size_);
    }
  }

  void RunForward() {
    const Matrix<Real> &x = this->GetInput();

    int length = x.cols();

    this->z_.setZero(this->hidden_size_, length);
    this->r_.setZero(this->hidden_size_, length);
    this->hu_.setZero(this->hidden_size_, length);
    z_r_.setZero(this->hidden_size_, length);
    r_r_.setZero(this->hidden_size_, length);
    lu_.setZero(this->hidden_size_, length);

    this->output_.setZero(2*this->hidden_size_, length);

    Matrix<Real> zraw = (this->Wxz_ * x).colwise() + this->bz_;
    Matrix<Real> rraw = (this->Wxr_ * x).colwise() + this->br_;
    Matrix<Real> hraw = (this->Wxh_ * x).colwise() + this->bh_;
    Matrix<Real> zraw_r = (Wxz_r_ * x).colwise() + bz_r_;
    Matrix<Real> rraw_r = (Wxr_r_ * x).colwise() + br_r_;
    Matrix<Real> lraw = (Wxl_ * x).colwise() + bl_;
    Vector<Real> hprev = Vector<Real>::Zero(this->hidden_size_);
    Vector<Real> lnext = Vector<Real>::Zero(this->hidden_size_);
    if (this->use_hidden_start_) {
      hprev = this->h0_;
      lnext = l0_;
    }
    Vector<Real> result;
    Vector<Real> tmp;
    for (int t = 0; t < length; ++t) {
      tmp = zraw.col(t) + this->Whz_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      this->z_.col(t) = result;

      tmp = rraw.col(t) + this->Whr_ * hprev;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      this->r_.col(t) = result;

      tmp = hraw.col(t) + this->Whh_ * this->r_.col(t).cwiseProduct(hprev);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      this->hu_.col(t) = result;
      this->output_.block(0, t, this->hidden_size_, 1) =
	this->z_.col(t).cwiseProduct(this->hu_.col(t) - hprev) + hprev;
      hprev = this->output_.block(0, t, this->hidden_size_, 1);
    }
    for (int t = length-1; t >= 0; --t) {
      tmp = zraw_r.col(t) + Wlz_r_ * lnext;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      z_r_.col(t) = result;

      tmp = rraw_r.col(t) + Wlr_r_ * lnext;
      EvaluateActivation(ActivationFunctions::LOGISTIC,
                         tmp,
                         &result);
      r_r_.col(t) = result;

      tmp = lraw.col(t) + Wll_ * r_r_.col(t).cwiseProduct(lnext);
      EvaluateActivation(this->activation_function_,
                         tmp,
                         &result);
      lu_.col(t) = result;
      this->output_.block(this->hidden_size_, t, this->hidden_size_, 1) =
	z_r_.col(t).cwiseProduct(lu_.col(t) - lnext) + lnext;
      lnext = this->output_.block(this->hidden_size_, t, this->hidden_size_, 1);
    }
  }

  void RunBackward() {
    const Matrix<Real> &x = this->GetInput();
    Matrix<Real> *dx = this->GetInputDerivative();

    Vector<Real> dhnext = Vector<Real>::Zero(this->Whh_.rows());
    Vector<Real> dlprev = Vector<Real>::Zero(Wll_.rows());
    //const Matrix<Real> &dy = this->output_derivative_;

    int length = this->output_.cols();
    Matrix<Real> dhuraw;
    DerivateActivation(this->activation_function_, this->hu_, &dhuraw);
    Matrix<Real> dzraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, this->z_, &dzraw);
    Matrix<Real> drraw;
    DerivateActivation(ActivationFunctions::LOGISTIC, this->r_, &drraw);
    Matrix<Real> dluraw;
    DerivateActivation(this->activation_function_, lu_, &dluraw);
    Matrix<Real> dzraw_r;
    DerivateActivation(ActivationFunctions::LOGISTIC, z_r_, &dzraw_r);
    Matrix<Real> drraw_r;
    DerivateActivation(ActivationFunctions::LOGISTIC, r_r_, &drraw_r);

    for (int t = length - 1; t >= 0; --t) {
      Vector<Real> dh = this->output_derivative_.block(0, t, this->hidden_size_, 1) + dhnext; // Backprop into h.
      Vector<Real> dhu = this->z_.col(t).cwiseProduct(dh);

      dhuraw.col(t) = dhuraw.col(t).cwiseProduct(dhu);
      Vector<Real> hprev;
      if (t == 0) {
        hprev = Vector<Real>::Zero(this->hidden_size_);
      } else {
        hprev = this->output_.block(0, t-1, this->hidden_size_, 1);
      }

      Vector<Real> dq = this->Whh_.transpose() * dhuraw.col(t);
      Vector<Real> dz = (this->hu_.col(t) - hprev).cwiseProduct(dh);
      Vector<Real> dr = hprev.cwiseProduct(dq);

      dzraw.col(t) = dzraw.col(t).cwiseProduct(dz);
      drraw.col(t) = drraw.col(t).cwiseProduct(dr);

      dhnext.noalias() =
        this->Whz_.transpose() * dzraw.col(t) +
        this->Whr_.transpose() * drraw.col(t) +
        this->r_.col(t).cwiseProduct(dq) +
        (1.0 - this->z_.col(t).array()).matrix().cwiseProduct(dh);
    }

    for (int t = 0; t < length; ++t) {
      Vector<Real> dl = this->output_derivative_.block(this->hidden_size_, t, this->hidden_size_, 1) + dlprev; // Backprop into h.
      Vector<Real> dlu = z_r_.col(t).cwiseProduct(dl);

      dluraw.col(t) = dluraw.col(t).cwiseProduct(dlu);
      Vector<Real> lnext;
      if (t == length-1) {
        lnext = Vector<Real>::Zero(this->hidden_size_);
      } else {
        lnext = this->output_.block(this->hidden_size_, t+1, this->hidden_size_, 1);
      }

      Vector<Real> dq_r = this->Wll_.transpose() * dluraw.col(t);
      Vector<Real> dz_r = (lu_.col(t) - lnext).cwiseProduct(dl);
      Vector<Real> dr_r = lnext.cwiseProduct(dq_r);

      dzraw_r.col(t) = dzraw_r.col(t).cwiseProduct(dz_r);
      drraw_r.col(t) = drraw_r.col(t).cwiseProduct(dr_r);

      dlprev.noalias() =
        Wlz_r_.transpose() * dzraw_r.col(t) +
        Wlr_r_.transpose() * drraw_r.col(t) +
        r_r_.col(t).cwiseProduct(dq_r) +
        (1.0 - z_r_.col(t).array()).matrix().cwiseProduct(dl);
    }

    *dx += this->Wxz_.transpose() * dzraw + this->Wxr_.transpose() * drraw +
      this->Wxh_.transpose() * dhuraw; // Backprop into x.
    *dx += Wxz_r_.transpose() * dzraw_r + Wxr_r_.transpose() * drraw_r +
      Wxl_.transpose() * dluraw; // Backprop into x.

    this->dWxz_.noalias() += dzraw * x.transpose();
    this->dbz_.noalias() += dzraw.rowwise().sum();
    this->dWxr_.noalias() += drraw * x.transpose();
    this->dbr_.noalias() += drraw.rowwise().sum();
    this->dWxh_.noalias() += dhuraw * x.transpose();
    this->dbh_.noalias() += dhuraw.rowwise().sum();

    dWxz_r_.noalias() += dzraw_r * x.transpose();
    dbz_r_.noalias() += dzraw_r.rowwise().sum();
    dWxr_r_.noalias() += drraw_r * x.transpose();
    dbr_r_.noalias() += drraw_r.rowwise().sum();
    dWxl_.noalias() += dluraw * x.transpose();
    dbl_.noalias() += dluraw.rowwise().sum();

    this->dWhz_.noalias() += dzraw.rightCols(length-1) *
      this->output_.block(0, 0, this->hidden_size_, length-1).transpose();
    this->dWhr_.noalias() += drraw.rightCols(length-1) *
      this->output_.block(0, 0, this->hidden_size_, length-1).transpose();
    this->dWhh_.noalias() += dhuraw.rightCols(length-1) *
      ((this->r_.rightCols(length-1)).cwiseProduct(this->output_.block(0, 0, this->hidden_size_, length-1))).
      transpose();

    dWlz_r_.noalias() += dzraw_r.leftCols(length-1) *
      this->output_.block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dWlr_r_.noalias() += drraw_r.leftCols(length-1) *
      this->output_.block(this->hidden_size_, 1, this->hidden_size_, length-1).transpose();
    dWll_.noalias() += dluraw.leftCols(length-1) *
      ((r_r_.leftCols(length-1)).cwiseProduct(this->output_.block(this->hidden_size_, 1, this->hidden_size_, length-1))).
      transpose();

    this->dh0_.noalias() += dhnext;
    this->dl0_.noalias() += dlprev;

    //std::cout << "dl0=" << this->dl0_ << std::endl;
  }

 protected:
  Matrix<Real> Wxl_;
  Matrix<Real> Wll_;
  Matrix<Real> Wxz_r_;
  Matrix<Real> Wlz_r_;
  Matrix<Real> Wxr_r_;
  Matrix<Real> Wlr_r_;
  Vector<Real> bl_;
  Vector<Real> bz_r_;
  Vector<Real> br_r_;
  Vector<Real> l0_;

  Matrix<Real> dWxl_;
  Matrix<Real> dWll_;
  Matrix<Real> dWxz_r_;
  Matrix<Real> dWlz_r_;
  Matrix<Real> dWxr_r_;
  Matrix<Real> dWlr_r_;
  Vector<Real> dbl_;
  Vector<Real> dbz_r_;
  Vector<Real> dbr_r_;
  Vector<Real> dl0_;

  Matrix<Real> z_r_;
  Matrix<Real> r_r_;
  Matrix<Real> lu_;
};

template<typename Real> class AttentionLayer : public Layer<Real> {
 public:
  AttentionLayer() {}
  AttentionLayer(int input_size,
                 int control_size,
                 int hidden_size,
                 bool use_sparsemax) {
    this->name_ = "Attention";
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
    Wxz_ = Matrix<Real>::Zero(hidden_size_, input_size_);
    Wyz_ = Matrix<Real>::Zero(hidden_size_, control_size_);
    wzp_ = Matrix<Real>::Zero(hidden_size_, 1);
    bz_ = Vector<Real>::Zero(hidden_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
                            std::vector<Vector<Real>*> *biases,
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

  void CollectAllParameterDerivatives(
      std::vector<Matrix<Real>*> *weight_derivatives,
      std::vector<Vector<Real>*> *bias_derivatives) {
    weight_derivatives->push_back(&dWxz_);
    weight_derivatives->push_back(&dWyz_);
    weight_derivatives->push_back(&dwzp_);
    bias_derivatives->push_back(&dbz_);
  }

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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
    assert(this->GetNumInputs() == 2);
    const Matrix<Real> &X = *(this->inputs_[0]); // Input subject to attention.
    const Matrix<Real> &y = *(this->inputs_[1]); // Control input.
    assert(y.cols() == 1);

#if 0
    std::cout << "X_att=" << X << std::endl;
    std::cout << "y_att=" << y << std::endl;
#endif

    int length = X.cols();
    z_.setZero(hidden_size_, length);

    Vector<Real> tmp, result;
    for (int t = 0; t < length; ++t) {
      tmp = Wxz_ * X.col(t) + bz_ + Wyz_ * y;
      EvaluateActivation(activation_function_,
                         tmp,
                         &result);
      z_.col(t) = result;
    }

#if 0
    std::cout << "z_att=" << z_ << std::endl;
#endif

    Vector<Real> v = z_.transpose() * wzp_;
    if (use_sparsemax_) {
      Real tau;
      Real r = 1.0;
      //std::cout << "v=" << v << std::endl;
      ProjectOntoSimplex(v, r, &p_, &tau);
      //std::cout << "p=" << p_ << std::endl;
      //std::cout << "tau=" << tau << std::endl;
    } else {
      Real logsum = LogSumExp(v);
      p_ = (v.array() - logsum).exp();
    }

#if 0
    std::cout << "p_att=" << p_ << std::endl;
#endif

    this->output_.noalias() = X * p_;
  }

  void RunBackward() {
    assert(this->GetNumInputs() == 2);
    const Matrix<Real> &X = *(this->inputs_[0]); // Input subject to attention.
    const Matrix<Real> &y = *(this->inputs_[1]); // Control input.
    Matrix<Real> *dX = this->input_derivatives_[0];
    Matrix<Real> *dy = this->input_derivatives_[1];
    assert(y.cols() == 1);
    assert(this->output_derivative_.cols() == 1);

    int length = X.cols();
    dp_.noalias() = X.transpose() * this->output_derivative_;

    Vector<Real> Jdp;
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

    Matrix<Real> dzraw = Matrix<Real>::Zero(hidden_size_, length);
    for (int t = 0; t < length; ++t) {
      Vector<Real> result; // TODO: perform this in a matrix-level way to be more efficient.
      Vector<Real> tmp = z_.col(t);
      DerivateActivation(activation_function_, tmp, &result);
      dzraw.col(t).noalias() = result;
    }
    dzraw = dzraw.array() * dz_.array();
    Vector<Real> dzraw_sum = dzraw.rowwise().sum();

    *dX += Wxz_.transpose() * dzraw;
    *dX += this->output_derivative_ * p_.transpose();
    *dy += Wyz_.transpose() * dzraw_sum;

    dwzp_ += z_ * Jdp;
    dWxz_.noalias() += dzraw * X.transpose();
    dWyz_.noalias() += dzraw_sum * y.transpose();
    dbz_.noalias() += dzraw_sum;
  }

  const Vector<Real> &GetAttentionProbabilities() {
    return p_;
  }

#if 0
  void UpdateParameters(double learning_rate) {
    Wxz_ -= learning_rate * dWxz_;
    Wyz_ -= learning_rate * dWyz_;
    dbz_ -= learning_rate * dbz_;
    wzp_ -= learning_rate * dwzp_;
  }
#endif

 protected:
  int activation_function_;
  int hidden_size_;
  int input_size_;
  int control_size_;
  bool use_sparsemax_;

  Matrix<Real> Wxz_;
  Matrix<Real> Wyz_;
  Matrix<Real> wzp_; // Column vector.
  Vector<Real> bz_;

  Matrix<Real> dWxz_;
  Matrix<Real> dWyz_;
  Matrix<Real> dwzp_;
  Vector<Real> dbz_;

  Matrix<Real> z_;
  Vector<Real> p_;

  Matrix<Real> dz_;
  Vector<Real> dp_;
};

#endif /* LAYER_H_ */
