#ifndef SNLI_DATA_H_
#define SNLI_DATA_H_

#include <vector>
#include "utils.h"
#include "nn_utils.h"

class Sentence {
 public:
  void Initialize(const std::vector<std::string> &words,
                  const std::string label) {
    words_ = words;
    label_ = label;
  }
  int Size() { return words_.size(); }
  const std::string &GetWord(int i) { return words_[i]; }
  const std::string &GetLabel() { return label_; }

 protected:
  std::vector<std::string> words_;
  std::string label_;
};

class Dictionary {
 public:
  void Clear() {
    word_alphabet_.clear();
    prefix_alphabet_.clear();
    suffix_alphabet_.clear();
    label_alphabet_.clear();
  }

  int AddWord(const std::string &word) {
    assert(word_alphabet_.find(word) == word_alphabet_.end());
    int wid = word_alphabet_.size();
    word_alphabet_[word] = wid;
    return wid;
  }

  int max_affix_size() const { return max_affix_size_; }
  void set_max_affix_size(int max_affix_size) {
    max_affix_size_ = max_affix_size;
  }

  int GetNumWords() const { return word_alphabet_.size(); }
  int GetNumPrefixes() const { return prefix_alphabet_.size(); }
  int GetNumSuffixes() const { return suffix_alphabet_.size(); }
  int GetNumLabels() const { return label_alphabet_.size(); }

  int GetWordId(const std::string &word) const {
    std::unordered_map<std::string, int>::const_iterator it =
      word_alphabet_.find(word);
    if (it != word_alphabet_.end()) return it->second;
    return word_alphabet_.find("__UNK__")->second; // Unknown symbol.
  }

  int GetPrefixId(const std::string &prefix) const {
    std::unordered_map<std::string, int>::const_iterator it =
      prefix_alphabet_.find(prefix);
    if (it != prefix_alphabet_.end()) return it->second;
    return prefix_alphabet_.find("__UNK__")->second; // Unknown symbol.
  }

  int GetSuffixId(const std::string &suffix) const {
    std::unordered_map<std::string, int>::const_iterator it =
      suffix_alphabet_.find(suffix);
    if (it != suffix_alphabet_.end()) return it->second;
    return suffix_alphabet_.find("__UNK__")->second; // Unknown symbol.
  }

  int GetLabelId(const std::string &label) const {
    std::unordered_map<std::string, int>::const_iterator it =
      label_alphabet_.find(label);
    if (it != label_alphabet_.end()) return it->second;
    assert(false); // Cannot be unknown symbol.
  }

  void AddWordsFromDataset(const std::vector<Sentence*> &dataset,
			   int word_cutoff,
			   int affix_cutoff) {
    // Temporary dictionaries.
    std::unordered_map<std::string, int> word_alphabet = word_alphabet_;
    std::unordered_map<std::string, int> prefix_alphabet = prefix_alphabet_;
    std::unordered_map<std::string, int> suffix_alphabet = suffix_alphabet_;

    Clear();

    // Zero ID for the unknown word.
    int wid_unknown = -1;
    int pid_unknown = -1;
    int sid_unknown = -1;
    std::string unknown_symbol = "__UNK__";
    if (word_alphabet.find(unknown_symbol) != word_alphabet.end()) {
      wid_unknown = word_alphabet[unknown_symbol];
    } else {
      wid_unknown = word_alphabet.size();
      word_alphabet[unknown_symbol] = wid_unknown;
    }
    if (prefix_alphabet.find(unknown_symbol) != prefix_alphabet.end()) {
      pid_unknown = prefix_alphabet[unknown_symbol];
    } else {
      pid_unknown = prefix_alphabet.size();
      prefix_alphabet[unknown_symbol] = pid_unknown;
    }
    if (suffix_alphabet.find(unknown_symbol) != suffix_alphabet.end()) {
      sid_unknown = suffix_alphabet[unknown_symbol];
    } else {
      sid_unknown = suffix_alphabet.size();
      suffix_alphabet[unknown_symbol] = sid_unknown;
    }

    int num_initial_words = word_alphabet.size();
    int num_initial_prefixes = prefix_alphabet.size();
    int num_initial_suffixes = suffix_alphabet.size();

    std::vector<int> word_frequencies(word_alphabet.size(), 0);
    std::vector<int> prefix_frequencies(prefix_alphabet.size(), 0);
    std::vector<int> suffix_frequencies(suffix_alphabet.size(), 0);

    for (int i = 0; i < dataset.size(); ++i) {
      Sentence *sentence = dataset[i];
      const std::string &label = sentence->GetLabel();
      for (int j = 0; j < sentence->Size(); ++j) {
        const std::string &word = sentence->GetWord(j);

        int wid, pid, sid;
        std::unordered_map<std::string, int>::iterator it =
          word_alphabet.find(word);
        if (it != word_alphabet.end()) {
          wid = it->second;
          ++word_frequencies[wid];
        } else {
          wid = word_alphabet.size();
          word_alphabet[word] = wid;
          word_frequencies.push_back(1);
        }

        for (int k = 1; k <= max_affix_size(); ++k) {
          std::string prefix = word.substr(0, k);
          it = prefix_alphabet.find(prefix);
          if (it != prefix_alphabet.end()) {
            pid = it->second;
            ++prefix_frequencies[pid];
          } else {
            pid = prefix_alphabet.size();
            prefix_alphabet[prefix] = pid;
            prefix_frequencies.push_back(1);
          }
        }

        for (int k = 1; k <= max_affix_size(); ++k) {
          int start = word.length() - k;
          if (start < 0) start = 0;
          std::string suffix = word.substr(start, k);
          it = suffix_alphabet.find(suffix);
          if (it != suffix_alphabet.end()) {
            sid = it->second;
            ++suffix_frequencies[sid];
          } else {
            sid = suffix_alphabet.size();
            suffix_alphabet[suffix] = sid;
            suffix_frequencies.push_back(1);
          }
        }
      }
      int lid;
      std::unordered_map<std::string, int>::iterator it =
        label_alphabet_.find(label);
      if (it != label_alphabet_.end()) {
        lid = it->second;
      } else {
        lid = label_alphabet_.size();
        label_alphabet_[label] = lid;
      }
    }

    std::cout << "Number of words before cutoff: " << word_alphabet.size()
              << std::endl;
    std::cout << "Number of prefixes before cutoff: " << prefix_alphabet.size()
              << std::endl;
    std::cout << "Number of suffixes before cutoff: " << suffix_alphabet.size()
              << std::endl;
    std::vector<std::string> words(word_frequencies.size());
    for (auto it = word_alphabet.begin(); it != word_alphabet.end(); ++it) {
      words[it->second] = it->first;
    }
    for (int wid = 0; wid < word_frequencies.size(); ++wid) {
      if (wid < num_initial_words || word_frequencies[wid] > word_cutoff) {
        int new_wid = word_alphabet_.size();
        word_alphabet_[words[wid]] = new_wid;
      }
    }
    std::vector<std::string> prefixes(prefix_frequencies.size());
    for (auto it = prefix_alphabet.begin(); it != prefix_alphabet.end(); ++it) {
      prefixes[it->second] = it->first;
    }
    for (int pid = 0; pid < prefix_frequencies.size(); ++pid) {
      if (pid < num_initial_prefixes || prefix_frequencies[pid] > affix_cutoff) {
        int new_pid = prefix_alphabet_.size();
        prefix_alphabet_[prefixes[pid]] = new_pid;
      }
    }
    std::vector<std::string> suffixes(suffix_frequencies.size());
    for (auto it = suffix_alphabet.begin(); it != suffix_alphabet.end(); ++it) {
      suffixes[it->second] = it->first;
    }
    for (int sid = 0; sid < suffix_frequencies.size(); ++sid) {
      if (sid < num_initial_suffixes || suffix_frequencies[sid] > affix_cutoff) {
        int new_sid = suffix_alphabet_.size();
        suffix_alphabet_[suffixes[sid]] = new_sid;
      }
    }
    std::cout << "Number of words after cutoff: " << word_alphabet_.size()
              << std::endl;
    std::cout << "Number of prefixes after cutoff: " << prefix_alphabet_.size()
              << std::endl;
    std::cout << "Number of suffixes after cutoff: " << suffix_alphabet_.size()
              << std::endl;
  }

 protected:
  int max_affix_size_;
  std::unordered_map<std::string, int> word_alphabet_;
  std::unordered_map<std::string, int> prefix_alphabet_;
  std::unordered_map<std::string, int> suffix_alphabet_;
  std::unordered_map<std::string, int> label_alphabet_;
};

class Input {
 public:
  int wid() const { return wid_; }
  const std::vector<int> &pids() const { return pids_; }
  const std::vector<int> &sids() const { return sids_; }
  void Initialize(const std::string &word,
                  int max_affix_size,
                  const Dictionary &dictionary) {
    wid_ = dictionary.GetWordId(word);
    pids_.clear();
    sids_.clear();
    for (int k = 1; k <= max_affix_size; ++k) {
      std::string prefix = word.substr(0, k);
      pids_.push_back(dictionary.GetPrefixId(prefix));
    }
    for (int k = 1; k <= max_affix_size; ++k) {
      int start = word.length() - k;
      if (start < 0) start = 0;
      std::string suffix = word.substr(start, k);
      sids_.push_back(dictionary.GetSuffixId(suffix));
    }
  }

 protected:
  int wid_;
  std::vector<int> pids_;
  std::vector<int> sids_;
};

#endif // SNLI_DATA_H_
