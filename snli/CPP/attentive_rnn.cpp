#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <assert.h>
#include "attentive_rnn.h"

const int kMaxAffixSize = 4;

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
    dictionary->CreateFromDataset(dataset, kMaxAffixSize, cutoff, 0);
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
  int num_hidden_units = atoi(argv[4]);
  int num_epochs = atoi(argv[5]);
  double learning_rate = atof(argv[6]);

  int embedding_dimension = 64;
  int word_cutoff = 1;

  Dictionary dictionary;
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
          num_hidden_units, dictionary.GetNumLabels());
  rnn.InitializeParameters();
  rnn.Train(input_sequences, output_labels,
            input_sequences_dev, output_labels_dev,
            input_sequences_test, output_labels_test,
            num_epochs, learning_rate);
}
