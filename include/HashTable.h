#ifndef HASHTABLE_HEADERDEF
#define HASHTABLE_HEADERDEF

#include <functional>
class HashTable
{
  std::hash<int> h0;
  std::hash<int> h1;
  std::hash<int> h2;
  std::ArrayXd table;

  int operator()(Eigen::Vector3i row)
  {
    return h0(row(0)) + h1(row(1)) + h2(row(2));
  }
}
#endif
