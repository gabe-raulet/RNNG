
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

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, float>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_File fh;
    MPI_Aint extent;
    MPI_Offset filesize, filedisp;
    Index total, myleft, mysize;
    int dim;

    MPI_Datatype MPI_POINT;
    MPI_Datatype MPI_ATOM = MPI_FLOAT;

    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    if (!myrank)
    {
        MPI_File_get_size(fh, &filesize);
        MPI_File_read(fh, &dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
    MPI_Bcast(&filesize, 1, MPI_OFFSET, 0, comm);

    extent = 4 * (dim + 1);
    total = filesize / extent;

    assert((sizeof(Atom) == 4));
    assert((filesize % extent == 0));

    mysize = total / nprocs;
    myleft = total % nprocs;

    if (myrank < myleft)
        mysize++;

    IndexVector sizes(nprocs);
    sizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, sizes.data(), 1, MPI_INDEX, comm);

    IndexVector offsets(nprocs);
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), (Index)0);
    Index totsize = offsets.back() + sizes.back();
    Index myoffset = offsets[myrank];

    AtomVector myatoms(mysize*dim);

    assert((dim >= 1));
    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    MPI_Datatype filetype;
    MPI_Type_create_resized(MPI_POINT, 0, extent, &filetype);
    MPI_Type_commit(&filetype);

    filedisp = myoffset*extent + sizeof(int);
    MPI_File_set_view(fh, filedisp, MPI_POINT, filetype, "native", MPI_INFO_NULL);

    MPI_File_read(fh, myatoms.data(), (int)mysize, MPI_POINT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    MPI_Type_free(&filetype);
    MPI_Type_free(&MPI_POINT);

    IndexVector indices(mysize);
    std::iota(indices.begin(), indices.end(), myoffset);

    init(myatoms, mysize, dim, indices);

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, char>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = MPI_CHAR;

    PointContainer<Atom> allpoints;
    AtomVector alldata;
    IndexVector alldataoffsets;

    Index point_count;
    Index atom_count;

    if (!myrank)
    {
        allpoints.read_seqs(fname);

        point_count = allpoints.num_points();
        atom_count = allpoints.num_atoms();

        alldataoffsets.reserve(point_count+1);
        alldata.reserve(atom_count);

        for (Index i = 0; i < point_count; ++i)
        {
            const PointType& p = allpoints[i];

            alldataoffsets.push_back(alldata.size());
            alldata.insert(alldata.end(), p.begin(), p.end());
        }

        alldataoffsets.push_back(alldata.size());
    }

    MPI_Bcast(&point_count, 1, MPI_INDEX, 0, comm);
    MPI_Bcast(&atom_count, 1, MPI_INDEX, 0, comm);

    if (myrank != 0)
    {
        alldata.resize(atom_count);
        alldataoffsets.resize(point_count+1);
    }

    MPI_Bcast(alldata.data(), static_cast<int>(atom_count), MPI_ATOM, 0, comm);
    MPI_Bcast(alldataoffsets.data(), static_cast<int>(point_count+1), MPI_INDEX, 0, comm);

    Index mysize = point_count/nprocs;
    Index myleft = point_count%nprocs;

    if (myrank < myleft)
        mysize++;

    AtomVector mydata;
    IndexVector mydatasizes;

    Index myoffset;
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    for (Index i = myoffset; i < myoffset+mysize; ++i)
    {
        Index dataoffset = alldataoffsets[i];
        Index datasize = alldataoffsets[i+1] - dataoffset;

        auto first = alldata.begin() + dataoffset;
        auto last = first + datasize;

        std::copy(first, last, std::back_inserter(mydata));
        mydatasizes.push_back(datasize);
    }

    IndexVector myindices(mysize);
    std::iota(myindices.begin(), myindices.end(), myoffset);

    init(mydata, mydatasizes, myindices);

    return point_count;
}
