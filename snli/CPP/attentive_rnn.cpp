#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <assert.h>
#include "attentive_rnn.h"

const int kMaxAffixSize = 4;

void LoadWordVectors(const std::string &word_vector_file,
		     std::unordered_map<std::string, std::vector<double> > *word_vectors) {
  // Read word vectors.
  std::cout << "Loading word vectors..." << std::endl;
  std::ifstream is;
  is.open(word_vector_file.c_str(), std::ifstream::in);
  assert(is.good());
  int num_dimensions = -1;
  std::string line;
  if (is.is_open()) {
    getline(is, line); // Skip first line.
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, " ", &fields);
      if (num_dimensions < 0) {
	num_dimensions = fields.size()-1;
	std::cout << "Number of dimensions: " << num_dimensions << std::endl;
      } else {
	assert(num_dimensions == fields.size()-1);
      }
      std::string word = fields[0];
      std::vector<double> word_vector(num_dimensions, 0.0);
      for (int i = 0; i < num_dimensions; ++i) {
	word_vector[i] = atof(fields[1+i].c_str());
      }
      assert(word_vectors->find(word) == word_vectors->end());
      (*word_vectors)[word] = word_vector;
    }
  }
  is.close();
  std::cout << "Loaded " << word_vectors->size() << " word vectors." << std::endl;
}

void ReadDataset(const std::string &dataset_file,
                 bool locked_alphabets,
                 int cutoff,
                 Dictionary *dictionary,
                 std::vector<std::vector<Input> > *input_sequences,
                 std::vector<int> *output_labels) {
  // Read data.
  std::ifstream is;
  is.open(dataset_file.c_str(), std::ifstream::in);
  assert(is.good());
  std::vector<std::vector<std::string> > sentence_fields;
  std::string line;
  std::vector<Sentence*> dataset;
  if (is.is_open()) {
    while (!is.eof()) {
      getline(is, line);
      if (line == "") break;
      std::vector<std::string> fields;
      StringSplit(line, "\t", &fields);
      if (fields.size() != 3) std::cout << line << std::endl;
      assert(fields.size() == 3);
      std::string label = fields[0];
      std::string premise = fields[1];
      std::string hypothesis = fields[2];
      std::vector<std::string> premise_words;
      std::vector<std::string> hypothesis_words;
      StringSplit(premise, " ", &premise_words);
      StringSplit(hypothesis, " ", &hypothesis_words);
      std::vector<std::string> words;
      words.assign(premise_words.begin(), premise_words.end());
      words.push_back("__START__");
      words.insert(words.end(), hypothesis_words.begin(),
                   hypothesis_words.end());
      Sentence *sentence = new Sentence;
      sentence->Initialize(words, label);
      dataset.push_back(sentence);
    }
  }
  is.close();

  // Load dictionary if necessary.
  if (!locked_alphabets) {
    dictionary->AddWordsFromDataset(dataset, cutoff, 0);
  }

  // Create numeric dataset.
  input_sequences->clear();
  output_labels->clear();
  for (int i = 0; i < dataset.size(); ++i) {
    Sentence *sentence = dataset[i];
    std::vector<Input> word_sequence(sentence->Size());
    const std::string &label = sentence->GetLabel();
    int lid = dictionary->GetLabelId(label);
    for (int j = 0; j < sentence->Size(); ++j) {
      const std::string &word = sentence->GetWord(j);
      word_sequence[j].Initialize(word, kMaxAffixSize, *dictionary);
    }
    input_sequences->push_back(word_sequence);
    output_labels->push_back(lid);
  }
}

int main(int argc, char** argv) {
  std::string train_file = argv[1];
  std::string dev_file = argv[2];
  std::string test_file = argv[3];
  std::string word_vector_file = argv[4];
  bool use_attention = static_cast<bool>(atoi(argv[5]));
  bool sparse_attention = static_cast<bool>(atoi(argv[6]));
  int num_hidden_units = atoi(argv[7]);
  int num_epochs = atoi(argv[8]);
  double learning_rate = atof(argv[9]);

  int embedding_dimension = 300; //64;
  int word_cutoff = 1;

  if (use_attention) {
    if (sparse_attention) {
      std::cout << "Using sparse-max attention." << std::endl;
    } else {
      std::cout << "Using soft-max attention." << std::endl;
    }
  } else {
    std::cout << "Not using attention." << std::endl;
  }

  Dictionary dictionary;
  dictionary.Clear();
  dictionary.set_max_affix_size(kMaxAffixSize);

  std::unordered_map<std::string, std::vector<double> > word_vectors;
  LoadWordVectors(word_vector_file, &word_vectors);
  int num_fixed_embeddings = word_vectors.size();
  Matrix fixed_embeddings = Matrix::Zero(embedding_dimension,
					 num_fixed_embeddings);
  std::vector<int> word_ids; 
  for (auto it = word_vectors.begin(); it != word_vectors.end(); ++it) {
    int wid = dictionary.AddWord(it->first);
    const std::vector<double> &word_vector = it->second;
    word_ids.push_back(wid);
    for (int k = 0; k < word_vector.size(); ++k) {
      fixed_embeddings(k, wid) = static_cast<float>(word_vector[k]);
    }
  }

  std::vector<std::vector<Input> > input_sequences;
  std::vector<int> output_labels;
  ReadDataset(train_file, false, word_cutoff, &dictionary,
              &input_sequences, &output_labels);

  std::vector<std::vector<Input> > input_sequences_dev;
  std::vector<int> output_labels_dev;
  ReadDataset(dev_file, true, -1, &dictionary,
              &input_sequences_dev, &output_labels_dev);

  std::vector<std::vector<Input> > input_sequences_test;
  std::vector<int> output_labels_test;
  ReadDataset(test_file, true, -1, &dictionary,
              &input_sequences_test, &output_labels_test);

  std::cout << "Number of sentences: " << input_sequences.size() << std::endl;
  std::cout << "Number of words: " << dictionary.GetNumWords() << std::endl;
  std::cout << "Number of labels: " <<  dictionary.GetNumLabels() << std::endl;

  //BiRNN_GRU
  RNN rnn(&dictionary, embedding_dimension,
          num_hidden_units, dictionary.GetNumLabels(),
	  use_attention, sparse_attention);
  rnn.InitializeParameters();
  rnn.SetFixedEmbeddings(fixed_embeddings, word_ids);
  rnn.Train(input_sequences, output_labels,
            input_sequences_dev, output_labels_dev,
            input_sequences_test, output_labels_test,
            num_epochs, learning_rate);
}
