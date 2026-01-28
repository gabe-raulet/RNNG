
template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, const IndexVector& sizes)
{
    IndexVector indices(sizes.size());
    std::iota(indices.begin(), indices.end(), (Index)0);
    init(atoms, sizes, indices);
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices)
{
    init(atoms, sizes, indices);
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, Index size, Index dim)
{
    IndexVector indices(size);
    std::iota(indices.begin(), indices.end(), (Index)0);
    init(atoms, size, dim, indices);
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices)
{
    init(atoms, size, dim, indices);
}

template <class Atom_>
void PointContainer<Atom_>::init(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices)
{
    Index point_count = sizes.size();
    assert((point_count == indices.size()));

    auto first = atoms.cbegin();

    points.clear();
    points.reserve(point_count);

    for (Index i = 0; i < point_count; ++i)
    {
        auto last = first + sizes[i];
        points.emplace_back(first, last, indices[i]);
        first = last;
    }
}

template <class Atom_>
void PointContainer<Atom_>::init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices)
{
    assert((atoms.size() == size*dim));
    assert((indices.size() == size));

    auto first = atoms.cbegin();

    points.clear();
    points.reserve(size);

    for (Index i = 0; i < size; ++i)
    {
        auto last = first + dim;
        points.emplace_back(first, last, indices[i]);
        first = last;
    }
}

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname)
{
    assert((std::same_as<Atom, float>));

    std::ifstream is;
    size_t filesize, total;
    std::vector<char> p;
    int d;

    is.open(fname, std::ios::binary | std::ios::in);

    is.seekg(0, is.end);
    filesize = is.tellg();
    is.seekg(0, is.beg);

    is.read((char*)&d, sizeof(int));
    is.seekg(0, is.beg);

    size_t point_size = sizeof(Atom)*d;
    size_t record_size = sizeof(int) + point_size;

    assert((filesize % record_size == 0));
    total = filesize / record_size;

    p.resize(record_size);
    points.clear();
    points.reserve(total);
    AtomVector atoms(d);

    for (size_t i = 0; i < total; ++i)
    {
        is.read(p.data(), record_size);

        const char *ds = p.data();
        const char *ps = p.data() + sizeof(int);

        int dt;
        std::memcpy(&dt, ds, sizeof(int)); assert((dt == d));

        char *dest = (char*)(&atoms[0]);
        std::memcpy(dest, ps, point_size);

        points.emplace_back(atoms.begin(), atoms.end(), i);
    }

    is.close();

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname)
{
    assert((std::same_as<Atom, char>));

    std::ifstream is;
    std::string line;

    points.clear();
    Index id = 0;

    is.open(fname, std::ios::in);

    while (std::getline(is, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        points.emplace_back(line.begin(), line.end(), id++);
    }

    is.close();

    return points.size();
}
