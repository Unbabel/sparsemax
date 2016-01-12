//#include <cstdlib>
//#include <Eigen/Dense>
//#include <cmath>
#include "layer.h"

int main(int argc, char** argv) {

  LinearLayer<double> linear_layer(5, 5);
  GRULayer<double> rnn_layer(5, 5);
  AttentionLayer<double> attention_layer(5, 5, 5, true);
  FeedforwardLayer<double> feedforward_layer(5, 5);
  SoftmaxOutputLayer<double> output_layer(5, 3);

  double delta = 1e-7;
  int num_checks = 20;
  int num_tokens = 4;

  DoubleMatrix x, dx;
  DoubleMatrix *dy;

  linear_layer.InitializeParameters();
  linear_layer.ResetGradients();
  x = DoubleMatrix::Random(linear_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(linear_layer.input_size(), num_tokens);
  linear_layer.SetNumInputs(1);
  linear_layer.SetInput(0, x);
  linear_layer.SetInputDerivative(0, &dx);
  dy = linear_layer.GetOutputDerivative();
  dy->setRandom(linear_layer.output_size(), num_tokens);
  linear_layer.CheckGradient(num_checks, delta);

  feedforward_layer.InitializeParameters();
  feedforward_layer.ResetGradients();
  x = DoubleMatrix::Random(feedforward_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(feedforward_layer.input_size(), num_tokens);
  feedforward_layer.SetNumInputs(1);
  feedforward_layer.SetInput(0, x);
  feedforward_layer.SetInputDerivative(0, &dx);
  dy = feedforward_layer.GetOutputDerivative();
  dy->setRandom(feedforward_layer.output_size(), num_tokens);
  feedforward_layer.CheckGradient(num_checks, delta);

  rnn_layer.InitializeParameters();
  rnn_layer.ResetGradients();
  x = DoubleMatrix::Random(rnn_layer.input_size(), num_tokens);
  dx = DoubleMatrix::Zero(rnn_layer.input_size(), num_tokens);
  rnn_layer.SetNumInputs(1);
  rnn_layer.SetInput(0, x);
  rnn_layer.SetInputDerivative(0, &dx);
  dy = rnn_layer.GetOutputDerivative();
  dy->setRandom(rnn_layer.hidden_size(), num_tokens);
  rnn_layer.CheckGradient(num_checks, delta);

  attention_layer.InitializeParameters();
  attention_layer.ResetGradients();
  DoubleMatrix x1 = DoubleMatrix::Random(attention_layer.input_size(),
                                         num_tokens);
  DoubleMatrix x2 = DoubleMatrix::Random(attention_layer.control_size(), 1);
  DoubleMatrix dx1 = DoubleMatrix::Zero(attention_layer.input_size(),
                                        num_tokens);
  DoubleMatrix dx2 = DoubleMatrix::Zero(attention_layer.control_size(), 1);
  attention_layer.SetNumInputs(2);
  attention_layer.SetInput(0, x1);
  attention_layer.SetInput(1, x2);
  attention_layer.SetInputDerivative(0, &dx1);
  attention_layer.SetInputDerivative(1, &dx2);
  dy = attention_layer.GetOutputDerivative();
  dy->setRandom(attention_layer.hidden_size(), 1);
  attention_layer.CheckGradient(num_checks, delta);

  output_layer.InitializeParameters();
  output_layer.ResetGradients();
  x = DoubleMatrix::Random(output_layer.input_size(), 1);
  dx = DoubleMatrix::Zero(output_layer.input_size(), 1);
  output_layer.SetNumInputs(1);
  output_layer.SetInput(0, x);
  output_layer.SetInputDerivative(0, &dx);
  dy = output_layer.GetOutputDerivative();
  //dy->setRandom(output_layer.output_size(), 1);
  output_layer.RunForward();
  int l = 0;
  output_layer.set_output_label(l);
  dy->setZero(output_layer.output_size(), 1);
  const DoubleMatrix &output = output_layer.GetOutput();
  (*dy)(l) = -1.0 / output(l);
  output_layer.CheckGradient(num_checks, delta);

#if 0
  Matrix *output_derivative = output_layer.GetOutputDerivative();
  const Matrix &output = output_layer.GetOutput();
  output_derivative->setZero(output_layer.output_size(), 1);
  int l = output_layer.output_label();
  (*output_derivative)(l) = -1.0 / output(l);
  output_layer.CheckGradient(num_checks);

  attention_layer.CheckGradient(num_checks);
  feedforward_layer.CheckGradient(num_checks);
  rnn_layer.CheckGradient(num_checks);
#endif

  return 0;
}
