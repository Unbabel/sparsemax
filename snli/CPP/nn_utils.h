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

#if 0
typedef Matrix<float> FloatMatrix;
typedef Vector<float> FloatVector;
#else
typedef Matrix<double> FloatMatrix;
typedef Vector<double> FloatVector;
#endif
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
void ReadMatrixParameter(const std::string &prefix, const std::string &name,
                         Matrix<Real> *W) {
  //std::string param_file = "model.1/" + name + ".txt";
  std::string param_file = prefix + name + ".txt";
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
void ReadVectorParameter(const std::string &prefix, const std::string &name,
                         Vector<Real> *b) {
  //std::string param_file = "model.1/" + name + ".txt";
  std::string param_file = prefix + name + ".txt";
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
void LoadMatrixParameterFromFile(const std::string &prefix,
                                 const std::string &name,
                                 Matrix<Real> *W) {
  std::string param_file = prefix + name + ".bin";
  FILE *fs = fopen(param_file.c_str(), "rb");
  LoadMatrixParameter(fs, W);
  fclose(fs);
}

template<typename Real>
void LoadVectorParameterFromFile(const std::string &prefix,
                                 const std::string &name,
                                 Vector<Real> *b) {
  std::string param_file = prefix + name + ".bin";
  FILE *fs = fopen(param_file.c_str(), "rb");
  LoadVectorParameter(fs, b);
  fclose(fs);
}

template<typename Real>
void SaveMatrixParameterToFile(const std::string &prefix,
                               const std::string &name,
                               Matrix<Real> *W) {
  std::string param_file = prefix + name + ".bin";
  FILE *fs = fopen(param_file.c_str(), "wb");
  SaveMatrixParameter(fs, W);
  fclose(fs);
}

template<typename Real>
void SaveVectorParameterToFile(const std::string &prefix,
                               const std::string &name,
                               Vector<Real> *b) {
  std::string param_file = prefix + name + ".bin";
  FILE *fs = fopen(param_file.c_str(), "wb");
  SaveVectorParameter(fs, b);
  fclose(fs);
}

template<typename Real>
void LoadMatrixParameter(FILE *fs, Matrix<Real> *W) {
  int num_rows, num_columns;
  if (1 != fread(&num_rows, sizeof(int), 1, fs)) assert(false);
  if (1 != fread(&num_columns, sizeof(int), 1, fs)) assert(false);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_columns; ++j) {
      double value;
      if (1 != fread(&value, sizeof(double), 1, fs)) assert(false);
      (*W)(i, j) = value;
    }
  }
}

template<typename Real>
void LoadVectorParameter(FILE *fs, Vector<Real> *b) {
  int length;
  if (1 != fread(&length, sizeof(int), 1, fs)) assert(false);
  for (int j = 0; j < length; ++j) {
    double value;
    if (1 != fread(&value, sizeof(double), 1, fs)) assert(false);
    (*b)(j) = value;
  }
}

template<typename Real>
void SaveMatrixParameter(FILE *fs, Matrix<Real> *W) {
  int num_rows = W->rows();
  int num_columns = W->cols();
  if (1 != fwrite(&num_rows, sizeof(int), 1, fs)) assert(false);
  if (1 != fwrite(&num_columns, sizeof(int), 1, fs)) assert(false);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_columns; ++j) {
      double value = (*W)(i, j);
      if (1 != fwrite(&value, sizeof(double), 1, fs)) assert(false);
    }
  }
}

template<typename Real>
void SaveVectorParameter(FILE *fs, Vector<Real> *b) {
  int length = b->size();
  if (1 != fwrite(&length, sizeof(int), 1, fs)) assert(false);
  for (int j = 0; j < length; ++j) {
    double value = (*b)(j);
    if (1 != fwrite(&value, sizeof(double), 1, fs)) assert(false);
  }
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
