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

  virtual void UpdateParameters(double learning_rate) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];
      *b -= learning_rate * (*db);
    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];
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

  void UpdateParametersADAM(double learning_rate) {
    std::vector<Matrix<Real>*> weights, weight_derivatives;
    std::vector<Vector<Real>*> biases, bias_derivatives;
    std::vector<std::string> weight_names;
    std::vector<std::string> bias_names;
    CollectAllParameters(&weights, &biases, &weight_names, &bias_names);
    CollectAllParameterDerivatives(&weight_derivatives, &bias_derivatives);

    for (int i = 0; i < biases.size(); ++i) {
      auto b = biases[i];
      auto db = bias_derivatives[i];

      auto mb = first_bias_moments_[i];
      auto vb = second_bias_moments_[i];

#if 1
      *mb = beta1_ * (*mb) + (1.0 - beta1_) * (*db);
      *vb = beta2_ * vb->array() + (1.0 - beta2_) * (db->array() * db->array());
      auto mb_corrected = (*mb) / (1.0 - pow(beta1_, iteration_number_ + 1));
      auto vb_corrected = (*vb) / (1.0 - pow(beta2_, iteration_number_ + 1));

      //*b = b->array() - learning_rate * mb_corrected.array() / (epsilon_ + vb_corrected.array().sqrt());
      double stepsize = learning_rate; // / sqrt(static_cast<double>(iteration_number_ + 1));
      *b = b->array() - stepsize * mb_corrected.array() / (epsilon_ + vb_corrected.array().sqrt());
#else
      // Adagrad.
      *vb = vb->array() + (db->array() * db->array());
      //std::cout << epsilon_ + vb->array().sqrt() << std::endl;
      *b = b->array() - learning_rate * db->array() / (epsilon_ + vb->array().sqrt());
      //*b = b->array() - learning_rate * db->array();
#endif

    }

    for (int i = 0; i < weights.size(); ++i) {
      auto W = weights[i];
      auto dW = weight_derivatives[i];

      auto mW = first_weight_moments_[i];
      auto vW = second_weight_moments_[i];

#if 1
      *mW = beta1_ * (*mW) + (1.0 - beta1_) * (*dW);
      *vW = beta2_ * vW->array() + (1.0 - beta2_) * (dW->array() * dW->array());
      auto mW_corrected = (*mW) / (1.0 - pow(beta1_, iteration_number_ + 1));
      auto vW_corrected = (*vW) / (1.0 - pow(beta2_, iteration_number_ + 1));

      //      *W = W->array() - learning_rate * mW_corrected.array() / (epsilon_ + vW_corrected.array().sqrt());
      double stepsize = learning_rate; // / sqrt(static_cast<double>(iteration_number_ + 1));
      *W = W->array() - stepsize * mW_corrected.array() / (epsilon_ + vW_corrected.array().sqrt());
#else
      // Adagrad.
      *vW = vW->array() + (dW->array() * dW->array());
      //std::cout << epsilon_ + vW->array().sqrt() << std::endl;
      *W = W->array() - learning_rate * dW->array() / (epsilon_ + vW->array().sqrt());
      //*W = W->array() - learning_rate * dW->array();
#endif

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
  void UpdateParameters(double learning_rate) {
    // Update word embeddings.
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

  void ResetParameters() {
    RNNLayer<Real>::ResetParameters();

    Wxz_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whz_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    Wxr_ = Matrix<Real>::Zero(this->hidden_size_, this->input_size_);
    Whr_ = Matrix<Real>::Zero(this->hidden_size_, this->hidden_size_);
    bz_ = Vector<Real>::Zero(this->hidden_size_);
    br_ = Vector<Real>::Zero(this->hidden_size_);
  }

  void CollectAllParameters(std::vector<Matrix<Real>*> *weights,
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

  void CollectAllParameterDerivatives(
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

  double GetUniformInitializationLimit(Matrix<Real> *W) {
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

  void ResetGradients() {
    RNNLayer<Real>::ResetGradients();
    dWxz_.setZero(this->hidden_size_, this->input_size_);
    dWhz_.setZero(this->hidden_size_, this->hidden_size_);
    dWxr_.setZero(this->hidden_size_, this->input_size_);
    dWhr_.setZero(this->hidden_size_, this->hidden_size_);
    dbz_.setZero(this->hidden_size_);
    dbr_.setZero(this->hidden_size_);
  }

  void RunForward() {
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

  void RunBackward() {
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
