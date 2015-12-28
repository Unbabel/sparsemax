#ifndef NN_UTILS_H_
#define NN_UTILS_H_

#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"

typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::RowVectorXf RowVector;

struct ActivationFunctions {
  enum types {
    TANH = 0,
    LOGISTIC,
    RELU,
    NUM_ACTIVATIONS
  };
};

void LoadMatrixParameter(const std::string& name, Matrix* W) {
  //std::string param_file = "model.1/" + name + ".txt";
  std::string param_file = "../theano/model/" + name + ".txt";
  std::ifstream is;
  is.open(param_file.c_str(), std::ifstream::in);
  assert(is.good());
  std::string line;
  int i = 0;
  if (is.is_open()) {
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, " ", &fields);
      assert(fields.size() == W->cols());
      for (int j = 0; j < fields.size(); ++j) {
        (*W)(i, j) = atof(fields[j].c_str());
      }
      ++i;
    }
  }
  assert(i == W->rows());
  is.close();
}

void LoadVectorParameter(const std::string& name, Vector* b) {
  //std::string param_file = "model.1/" + name + ".txt";
  std::string param_file = "../theano/model/" + name + ".txt";
  std::ifstream is;
  is.open(param_file.c_str(), std::ifstream::in);
  assert(is.good());
  std::string line;
  int i = 0;
  if (is.is_open()) {
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, " ", &fields);
      assert(fields.size() == b->size());
      for (int j = 0; j < fields.size(); ++j) {
        (*b)(j) = atof(fields[j].c_str());
      }
      ++i;
    }
  }
  assert(i == 1);
  is.close();
}

void EvaluateActivation(int activation_function, const Matrix &hu,
                        Matrix *h) {
  if (activation_function == ActivationFunctions::TANH) {
#if 0
    *h = Matrix::Zero(hu.rows(), hu.cols());
    float const *begin = hu.data();
    float const *end = begin + hu.rows() * hu.cols();
    float *it_out = h->data();
    for(const float *it = begin;
        it != end;
        ++it, ++it_out) {
      *it_out = static_cast<float>(tanh(*it));
    }
#else
    *h = hu.unaryExpr(std::ptr_fun<float, float>(std::tanh));
#endif
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *h = hu.unaryExpr([](float t) -> float { return 1.0 / (1.0 + exp(-t)); });
  } else {
    assert(false);
  }
}

void DerivateActivation(int activation_function, const Matrix &dh,
                        Matrix *dhu) {
  if (activation_function == ActivationFunctions::TANH) {
    *dhu = (1.0 - dh.array() * dh.array());
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *dhu = (1.0 - dh.array()) * dh.array();
  } else {
    assert(false);
  }
}

double LogSumExp(const Vector &x) {
#if 0
  //return log(x.unaryExpr(std::ptr_fun(exp)).sum());
  return log(x.array().exp().sum());
#else
  double xmax = x.maxCoeff();
  return xmax + log((x.array() - xmax).exp().sum());
#endif
}

#endif /* NN_UTILS_H_ */
