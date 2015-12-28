#ifndef UTILS_H_
#define UTILS_H_

#include <sys/time.h>
#include <string>

// Time difference in milliseconds.
int diff_ms(timeval t1, timeval t2) {
  return (((t1.tv_sec - t2.tv_sec) * 1000000) +
          (t1.tv_usec - t2.tv_usec))/1000;
}

// Split string str on any delimiting character in delim, and write the result
// as a vector of strings.
void StringSplit(const std::string &str,
                 const std::string &delim,
                 std::vector<std::string> *results) {
  size_t cutAt;
  std::string tmp = str;
  while ((cutAt = tmp.find_first_of(delim)) != tmp.npos) {
    if(cutAt > 0) {
      // Note: this "if" guarantees that every field is not empty.
      // This complies with multiple consecutive occurrences of the
      // delimiter (e.g. several consecutive occurrences of a whitespace
      // will count as a single delimiter).
      // To allow empty fields, this if-condition should be removed.
      results->push_back(tmp.substr(0,cutAt));
    }
    tmp = tmp.substr(cutAt+1);
  }
  if(tmp.length() > 0) results->push_back(tmp);
}

#endif // UTILS_H_

