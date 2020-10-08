# C++ Knowledge

## map vs. unordered map
[link](https://www.geeksforgeeks.org/map-vs-unordered_map-c/)

`map` is implemented with `Red Black Tree`, always `log(n)`. `unordered_map` is implemented with hash table, with a worst case of `O(n)`.

## Integer math
```cpp
(-24) % 50 == -24
```

## Sorting
reverse sort
```cpp
std::sort(numbers.begin(), numbers.end(), std::greater<int>());

// custom sort

struct greater
{
    template<class T>
    bool operator()(T const &a, T const &b) const { return a > b; }
};

std::sort(numbers.begin(), numbers.end(), greater());
```