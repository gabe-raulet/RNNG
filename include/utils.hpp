void selection_sample(Index range, Index size, IndexVector& sample, int seed)
{
    /*
     * Reference for random sampling: https://bastian.rieck.me/blog/2017/selection_sampling/
     */

    if (seed < 0)
    {
        static std::random_device rd;
        seed = rd();
    }

    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> U(0, std::nextafter(1.0, std::numeric_limits<double>::max()));

    sample.resize(size);
    Index pos = 0;

    for (Index i = 0; i < range; ++i)
    {
        if ((range - i) * U(gen) < size - pos)
        {
            sample[pos++] = i;
        }

        if (pos == size)
            break;
    }
}

std::string format_large_number(Index number, int prec)
{
    const char *units[] = {"", "K", "M", "B", "T", "P"};
    Real n = number;
    std::ostringstream ss;

    for (int i = 0; i < 6; ++i)
    {
        if (std::abs(n) < 1000)
        {
            ss << std::showpoint << std::fixed << std::setprecision(prec) << n << units[i];
            break;
        }

        n /= 1000;
    }

    return ss.str();
}

std::string format_large_number_in_bytes(Index number)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    Real n = number;
    std::ostringstream ss;

    for (int i = 0; i < 6; ++i)
    {
        if (std::abs(n) < 1024)
        {
            ss << std::showpoint << std::fixed << std::setprecision(1) << n << units[i];
            break;
        }

        n /= 1024;
    }

    return ss.str();
}

int get_comm_rank(MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int get_comm_size(MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

std::string get_dataset_from_path(const char *path)
{
    std::string p(path);
    const auto pos = p.find_last_of('/')+1;
    return p.substr(pos, p.find_last_of('.')-pos);
}

template <class T> MPI_Datatype mpi_type()
{
    if      constexpr (std::is_same_v<T, char>)               return MPI_CHAR;
    else if constexpr (std::is_same_v<T, signed char>)        return MPI_SIGNED_CHAR;
    else if constexpr (std::is_same_v<T, short>)              return MPI_SHORT;
    else if constexpr (std::is_same_v<T, int>)                return MPI_INT;
    else if constexpr (std::is_same_v<T, long>)               return MPI_LONG;
    else if constexpr (std::is_same_v<T, long long>)          return MPI_LONG_LONG;
    else if constexpr (std::is_same_v<T, unsigned char>)      return MPI_UNSIGNED_CHAR;
    else if constexpr (std::is_same_v<T, unsigned short>)     return MPI_UNSIGNED_SHORT;
    else if constexpr (std::is_same_v<T, unsigned int>)       return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<T, unsigned long>)      return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<T, unsigned long long>) return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<T, float>)              return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>)             return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, long double>)        return MPI_LONG_DOUBLE;
    else if constexpr (std::is_same_v<T, bool>)               return MPI_CXX_BOOL;
    else
    {
        throw std::runtime_error("error: mpi_type()");
        return MPI_BYTE;
    }
}

Index Binom::operator()(Index n, Index k)
{
    if (k > n) return 0;

    Pair p = {n, k};

    auto it = memo.find(p);

    if (it == memo.end())
    {
        Index a = 1, b = 1;

        for (Index i = n; i >= n-k+1; --i)
            a *= i;

        for (Index i = k; i >= 1; --i)
            b *= i;

        a /= b;

        memo.insert({p, a});

        return a;
    }
    else
    {
        return it->second;
    }
}

Index Binom::factorial(Index n) const
{
    if (n < 0) return 0;
    else if (n == 0 || n == 1) return 1;
    else
    {
        Index f = n;

        for (Index i = 2; i < n; ++i)
            f *= i;

        return f;
    }
}
