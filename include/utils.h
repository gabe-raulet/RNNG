#ifndef UTILS_H_
#define UTILS_H_

#undef NDEBUG
#include <assert.h>

#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <random>
#include <deque>
#include <tuple>
#include <type_traits>
#include <mpi.h>

using Index = int64_t;
using Real = float;

#ifdef MPI_INDEX
#undef MPI_INDEX
#endif

#define MPI_INDEX MPI_INT64_T

#ifdef MPI_REAL
#undef MPI_REAL
#endif

#define MPI_REAL MPI_FLOAT

using Tuple = std::tuple<Index, Real>;
using Triple = std::tuple<Index, Index, Real>;
using IndexPair = std::tuple<Index, Index>;

using IndexVector = std::vector<Index>;
using RealVector = std::vector<Real>;
using MemVector = std::vector<void*>;
using BoolVector = std::vector<bool>;
using IndexQueue = std::deque<Index>;
using IndexMap = std::unordered_map<Index, Index>;
using IndexSet = std::unordered_set<Index>;
using IndexSetVector = std::vector<IndexSet>;
using TupleVector = std::vector<Tuple>;
using TripleVector = std::vector<Triple>;
using IndexPairVector = std::vector<IndexPair>;

using IndexIter = typename IndexVector::const_iterator;

template <class Iter>
std::string container_repr(Iter first, Iter last)
{
    std::stringstream ss;
    ss << "[";

    while (first != last)
    {
        ss << *first;
        first++;

        if (first != last) ss << ", ";
    }

    ss << "]";
    return ss.str();
}

#define CONTAINER_REPR(container) (container_repr((container).begin(), (container).end()).c_str())

void selection_sample(Index range, Index size, IndexVector& sample, int seed);
std::string format_large_number(Index number, int prec);
std::string format_large_number_in_bytes(Index number);

#define LARGE(number) format_large_number((number), (1)).c_str()
#define LARGE_PREC(number, prec) format_large_number((number), (prec)).c_str()
#define LARGE_BYTES(number) format_large_number_in_bytes((number)).c_str()

int get_comm_rank(MPI_Comm comm);
int get_comm_size(MPI_Comm comm);

std::string get_dataset_from_path(const char *path);

template <class T> MPI_Datatype mpi_type();

class Binom
{
    public:

        using Pair = std::pair<Index, Index>;

        struct HashPair
        {
            size_t operator()(const Pair& p) const
            {
                size_t hash1 = std::hash<Index>{}(p.first);
                size_t hash2 = std::hash<Index>{}(p.second);
                return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
            }
        };

        using Memo = std::unordered_map<Pair, Index, HashPair>;

        Binom() {}

        Index operator()(Index n, Index k);

    private:

        Memo memo;

        Index factorial(Index n) const;
};

#include "utils.hpp"

#endif
