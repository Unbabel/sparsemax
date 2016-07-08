#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include "layer.h"
#include <map>
#include <set>
#include <queue>

template<typename Real> class NeuralNetwork {
 public:
  NeuralNetwork() {}
  virtual ~NeuralNetwork() {}

  const std::vector<Layer<Real>*> &GetLayers() { return layers_; }

  void AddLayer(Layer<Real> *layer) {
    layers_.push_back(layer);
    children_layers_[layer] = std::set<Layer<Real>*>();
    parent_layers_[layer] = std::set<Layer<Real>*>();
  }

  void ConnectLayers(Layer<Real> *origin, Layer<Real> *target,
                     int origin_output, int target_input) {
    target->SetInput(target_input, origin->GetOutput(origin_output));
    target->SetInputDerivative(target_input, origin->
                               GetMutableOutputDerivative(origin_output));
    const std::set<Layer<Real> *> &children = children_layers_[target];
    for (auto it = children.begin(); it != children.end(); ++it) {
      if (*it == origin) return;
    }
    children_layers_[target].insert(origin);
    parent_layers_[origin].insert(target);
  }

  void SortLayersByTopologicalOrder() {
    // Use Kahn's algorithm.
    layers_.clear();
    std::queue<Layer<Real>*> roots;
    auto active_children = children_layers_;
    for (auto it = children_layers_.begin();
         it != children_layers_.end();
         ++it) {
      if (it->second.empty()) {
        roots.push(it->first);
        active_children.erase(it->first);
      }
    }
    while (!roots.empty()) {
      Layer<Real> *layer = roots.front();
      roots.pop();
      layers_.push_back(layer);
      const std::set<Layer<Real> *> &parents = parent_layers_[layer];
      for (auto it = parents.begin(); it != parents.end(); ++it) {
        active_children[*it].erase(layer);
        if (active_children[*it].empty()) {
          roots.push(*it);
          active_children.erase(*it);
        }
      }
    }
    // If the graph is non-empty, then there is a cycle.
    assert(active_children.empty());

    for (int k = 0; k < layers_.size(); ++k) {
      std::cout << layers_[k]->name() << std::endl;
    }
  }

 protected:
  std::vector<Layer<Real>*> layers_;
  std::map<Layer<Real>*, std::set<Layer<Real>*> > children_layers_;
  std::map<Layer<Real>*, std::set<Layer<Real>*> > parent_layers_;
};

#endif /* NEURAL_NETWORK_H_ */
