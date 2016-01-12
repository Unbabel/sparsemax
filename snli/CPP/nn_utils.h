#ifndef NN_UTILS_H_
#define NN_UTILS_H_

#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include "utils.h"

//typedef Eigen::MatrixXf Matrix;
//typedef Eigen::VectorXf Vector;
//typedef Eigen::RowVectorXf RowVector;

template<typename Real>
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template<typename Real>
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

typedef Matrix<float> FloatMatrix;
typedef Vector<float> FloatVector;
typedef Matrix<double> DoubleMatrix;
typedef Vector<double> DoubleVector;

struct ActivationFunctions {
  enum types {
    TANH = 0,
    LOGISTIC,
    RELU,
    NUM_ACTIVATIONS
  };
};

template<typename Real>
void LoadMatrixParameter(const std::string& name, Matrix<Real>* W) {
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

template<typename Real>
void LoadVectorParameter(const std::string& name, Vector<Real>* b) {
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

template<typename Real>
void EvaluateActivation(int activation_function, const Matrix<Real> &hu,
                        Matrix<Real> *h) {
  if (activation_function == ActivationFunctions::TANH) {
#if 0
    *h = Matrix<Real>::Zero(hu.rows(), hu.cols());
    Real const *begin = hu.data();
    Real const *end = begin + hu.rows() * hu.cols();
    Real *it_out = h->data();
    for(const Real *it = begin;
        it != end;
        ++it, ++it_out) {
      *it_out = static_cast<Real>(tanh(*it));
    }
#else
    *h = hu.unaryExpr(std::ptr_fun<Real, Real>(std::tanh));
#endif
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *h = hu.unaryExpr([](Real t) -> Real { return 1.0 / (1.0 + exp(-t)); });
  } else {
    assert(false);
  }
}

template<typename Real>
void DerivateActivation(int activation_function, const Matrix<Real> &dh,
                        Matrix<Real> *dhu) {
  if (activation_function == ActivationFunctions::TANH) {
    *dhu = (1.0 - dh.array() * dh.array());
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *dhu = (1.0 - dh.array()) * dh.array();
  } else {
    assert(false);
  }
}

template<typename Real>
void EvaluateActivation(int activation_function, const Vector<Real> &hu,
                        Vector<Real> *h) {
  if (activation_function == ActivationFunctions::TANH) {
#if 0
    *h = Matrix::Zero(hu.rows(), hu.cols());
    Real const *begin = hu.data();
    Real const *end = begin + hu.rows() * hu.cols();
    Real *it_out = h->data();
    for(const Real *it = begin;
        it != end;
        ++it, ++it_out) {
      *it_out = static_cast<Real>(tanh(*it));
    }
#else
    *h = hu.unaryExpr(std::ptr_fun<Real, Real>(std::tanh));
#endif
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *h = hu.unaryExpr([](Real t) -> Real { return 1.0 / (1.0 + exp(-t)); });
  } else {
    assert(false);
  }
}

template<typename Real>
void DerivateActivation(int activation_function, const Vector<Real> &dh,
                        Vector<Real> *dhu) {
  if (activation_function == ActivationFunctions::TANH) {
    *dhu = (1.0 - dh.array() * dh.array());
  } else if (activation_function == ActivationFunctions::LOGISTIC) {
    *dhu = (1.0 - dh.array()) * dh.array();
  } else {
    assert(false);
  }
}

template<typename Real>
Real LogSumExp(const Vector<Real> &x) {
#if 0
  //return log(x.unaryExpr(std::ptr_fun(exp)).sum());
  return log(x.array().exp().sum());
#else
  Real xmax = x.maxCoeff();
  return xmax + log((x.array() - xmax).exp().sum());
#endif
}

template<typename Real>
void ProjectOntoSimplex(const Vector<Real> &x, Real r, Vector<Real> *p, Real *tau) {
  int j;
  int d = x.size();
  Real s = x.sum();
  *p = x;
  std::sort(p->data(), p->data() + d);
  for (j = 0; j < d; j++) {
    *tau = (s - r) / ((Real) (d - j));
    if ((*p)[j] > *tau) break;
    s -= (*p)[j];
  }
  for (j = 0; j < d; j++) {
    if (x[j] < *tau) {
      (*p)[j] = 0.0;
    } else {
      (*p)[j] = x[j] - (*tau);
    }
  }
}

#endif /* NN_UTILS_H_ */
