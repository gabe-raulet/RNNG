
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

    data = atoms;
    ids = indices;

    offsets.clear();
    offsets.reserve(point_count+1);
    Index disp = 0;

    for (Index i = 0; i < point_count; ++i)
    {
        offsets.push_back(disp);
        disp += sizes[i];
    }

    offsets.push_back(disp);
}

template <class Atom_>
void PointContainer<Atom_>::init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices)
{
    assert((atoms.size() == size*dim));
    assert((indices.size() == size));

    data = atoms;
    ids = indices;

    offsets.clear();
    offsets.resize(size+1);

    Index disp = 0;

    for (Index i = 0; i <= size; ++i)
    {
        offsets[i] = disp;
        disp += dim;
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

    data.resize(total*d);
    offsets.resize(total+1);
    ids.resize(total);

    Index disp = 0;

    for (size_t i = 0; i < total; ++i)
    {
        is.read(p.data(), record_size);

        const char *ds = p.data();
        const char *ps = p.data() + sizeof(int);

        int dt;
        std::memcpy(&dt, ds, sizeof(int)); assert((dt == d));

        char *dest = (char*)(&data[i*d]);
        std::memcpy(dest, ps, point_size);

        offsets[i] = disp;
        ids[i] = i;
        disp += d;
    }

    is.close();

    offsets[total] = disp;

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname)
{
    assert((std::same_as<Atom, char>));

    std::ifstream is;
    std::string line;

    Index id = 0;

    is.open(fname, std::ios::in);

    offsets.clear();
    ids.clear();
    data.clear();

    while (std::getline(is, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        offsets.push_back(data.size());
        ids.push_back(id++);

        std::copy(line.begin(), line.end(), std::back_inserter(data));
    }

    offsets.push_back(data.size());

    is.close();

    return id;
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
            const Point<Atom>& p = allpoints[i];

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

template <class Atom_>
void PointContainer<Atom_>::sendrecv(PointContainer& recvbuf, int recvrank, int sendrank, MPI_Comm comm, SendrecvRequest& req)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    int sendcount_buf[2], recvcount_buf[2];
    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;

    sendcount = num_points();
    sendcount_atoms = num_atoms();

    sendcount_buf[0] = sendcount;
    sendcount_buf[1] = sendcount_atoms;

    MPI_Request *reqs = req.reqs;

    MPI_Irecv(recvcount_buf, 2, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
    MPI_Isend(sendcount_buf, 2, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    recvcount = recvcount_buf[0];
    recvcount_atoms = recvcount_buf[1];

    AtomVector& senddata = data;
    IndexVector& sendids = ids;
    IndexVector& sendoffsets = offsets;

    AtomVector& recvdata = recvbuf.data;
    IndexVector& recvids = recvbuf.ids;
    IndexVector& recvoffsets = recvbuf.offsets;

    recvdata.resize(recvcount_atoms);
    recvids.resize(recvcount);
    recvoffsets.resize(recvcount+1);

    MPI_Irecv(recvdata.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
    MPI_Isend(senddata.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

    MPI_Irecv(recvids.data(), recvcount, MPI_INDEX, recvrank, myrank+2*nprocs, comm, &reqs[2]);
    MPI_Isend(sendids.data(), sendcount, MPI_INDEX, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

    MPI_Irecv(recvoffsets.data(), recvcount+1, MPI_INDEX, recvrank, myrank+3*nprocs, comm, &reqs[4]);
    MPI_Isend(sendoffsets.data(), sendcount+1, MPI_INDEX, sendrank, sendrank+3*nprocs, comm, &reqs[5]);
}

template <class Atom_>
void PointContainer<Atom_>::swap(PointContainer& other)
{
    std::swap(data, other.data);
    std::swap(offsets, other.offsets);
    std::swap(ids, other.ids);
}
