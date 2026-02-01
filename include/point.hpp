#include "search.h"
#include <string.h>

template <class Atom_>
PointContainer<Atom_>::PointContainer(const std::vector<Point<Atom>>& points)
{
    Index point_count = points.size();
    Index atom_count = 0;

    for (const auto& p : points)
        atom_count += p.size();

    data.reserve(atom_count);
    offsets.reserve(point_count+1);
    ids.reserve(point_count);

    for (Index i = 0; i < point_count; ++i)
    {
        Point<Atom> p = points[i];

        offsets.push_back(data.size());
        ids.push_back(p.id());
        data.insert(data.end(), p.begin(), p.end());
    }

    offsets.push_back(data.size());
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices)
{
    init(atoms, sizes, indices);
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices)
{
    init(atoms, size, dim, indices);
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const PointContainer& lhs, const PointContainer& rhs)
{
    data.reserve(lhs.num_atoms() + rhs.num_atoms());
    offsets.reserve(lhs.num_points() + rhs.num_points() + 1);
    ids.reserve(lhs.num_points() + rhs.num_points());

    offsets.push_back(0);

    for (Index i = 0; i < lhs.num_points(); ++i)
    {
        Point<Atom> p = lhs[i];

        data.insert(data.end(), p.begin(), p.end());
        offsets.push_back(data.size());
        ids.push_back(p.id());
    }

    for (Index i = 0; i < rhs.num_points(); ++i)
    {
        Point<Atom> p = rhs[i];

        data.insert(data.end(), p.begin(), p.end());
        offsets.push_back(data.size());
        ids.push_back(p.id());
    }
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
        memcpy(&dt, ds, sizeof(int)); assert((dt == d));

        char *dest = (char*)(&data[i*d]);
        memcpy(dest, ps, point_size);

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

template <class Atom_>
void PointContainer<Atom_>::allgather(const PointContainer& sendbuf, MPI_Comm comm)
{
    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    data.clear();
    offsets.clear();
    ids.clear();

    const AtomVector& sendbuf_atoms = sendbuf.data;
    const IndexVector& sendbuf_ids = sendbuf.ids;

    AtomVector& recvbuf_atoms = data;
    IndexVector& recvbuf_ids = ids;
    IndexVector& recvbuf_offsets = offsets;

    Index mysize = sendbuf.num_points();

    IndexVector sendbuf_sizes(mysize);
    IndexVector recvbuf_sizes;

    for (Index i = 0; i < mysize; ++i)
    {
        sendbuf_sizes[i] = sendbuf.size(i);
    }

    std::vector<int> recvcounts(nprocs), rdispls(nprocs);
    std::vector<int> recvcounts_atoms(nprocs), rdispls_atoms(nprocs);

    recvcounts[myrank] = mysize;
    recvcounts_atoms[myrank] = sendbuf_atoms.size();

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts_atoms.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
    std::exclusive_scan(recvcounts_atoms.begin(), recvcounts_atoms.end(), rdispls_atoms.begin(), 0);

    int totrecv = recvcounts.back() + rdispls.back();
    int totrecv_atoms = recvcounts_atoms.back() + rdispls_atoms.back();

    recvbuf_atoms.resize(totrecv_atoms);
    recvbuf_ids.resize(totrecv);
    recvbuf_sizes.resize(totrecv);

    MPI_Allgatherv(sendbuf_ids.data(), recvcounts[myrank], MPI_INDEX, recvbuf_ids.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);
    MPI_Allgatherv(sendbuf_sizes.data(), recvcounts[myrank], MPI_INDEX, recvbuf_sizes.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);
    MPI_Allgatherv(sendbuf_atoms.data(), recvcounts_atoms[myrank], MPI_ATOM, recvbuf_atoms.data(), recvcounts_atoms.data(), rdispls_atoms.data(), MPI_ATOM, comm);

    recvbuf_offsets.resize(totrecv);

    std::exclusive_scan(recvbuf_sizes.begin(), recvbuf_sizes.end(), recvbuf_offsets.begin(), (Index)0);
    recvbuf_offsets.push_back(recvbuf_offsets.back() + recvbuf_sizes.back());

    assert((recvbuf_offsets.back() == recvbuf_atoms.size()));
}

template <class Atom_>
void PointContainer<Atom_>::indexed_gather(const PointContainer& sendbuf, const IndexVector& index_offsets)
{
    Index newsize = index_offsets.size();

    offsets.resize(newsize+1);
    ids.resize(newsize);

    Index atom_count = 0;

    for (Index i = 0; i < newsize; ++i)
    {
        offsets[i] = atom_count;
        ids[i] = sendbuf.id(index_offsets[i]);
        atom_count += sendbuf.size(index_offsets[i]);
    }

    offsets[newsize] = atom_count;

    data.clear();
    data.reserve(atom_count);

    for (Index i = 0; i < newsize; ++i)
    {
        Point<Atom> p = sendbuf[index_offsets[i]];
        data.insert(data.end(), p.begin(), p.end());
    }
}

template <class Atom_>
void PointContainer<Atom_>::push_back(const Point<Atom>& p)
{
    std::copy(p.begin(), p.end(), std::back_inserter(data));
    offsets.push_back(data.size());
    ids.push_back(p.id());
}

template <class Atom_>
VoronoiCell<Atom_>::VoronoiCell(const std::vector<Point<Atom>>& points, const RealVector& dist_to_centers, Index cell_index)
    : PointContainer<Atom>(points), dist_to_centers(dist_to_centers), cell_index(cell_index), interior(points.size(), false) {}

template <class Atom_>
void VoronoiCell<Atom_>::add_ghost_point(const Point<Atom>& p)
{
    ghost_points.push_back(p);
}

template <class Atom_>
template <class Distance>
VoronoiDiagram<Atom_>::VoronoiDiagram(const PointContainer<Atom>& points, const PointContainer<Atom>& centers, Distance& distance)
    : PointContainer<Atom>(points),
      centers(centers),
      cell_indices(points.num_points(), 0),
      dist_to_centers(points.num_points(), std::numeric_limits<Real>::max())
{
    Index size = PointContainer<Atom>::num_points();
    Index num_centers = centers.num_points();

    for (Index i = 0; i < size; ++i)
    {
        for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
        {
            Real dist = distance(centers[cell_index], (*this)[i]);

            if (dist <= dist_to_centers[i])
            {
                dist_to_centers[i] = dist;
                cell_indices[i] = cell_index;
            }
        }
    }
}

template <class Atom_>
void VoronoiDiagram<Atom_>::coalesce_cells(std::vector<Cell>& mycells, MPI_Comm comm) const
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index myoffset;
    Index mysize = PointContainer<Atom>::num_points();
    Index num_centers = centers.num_points();

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    IndexVector cell_point_counts(num_centers), cell_atom_counts(num_centers);
    IndexVector my_cell_point_counts(num_centers, 0), my_cell_atom_counts(num_centers, 0);

    for (Index i = 0; i < mysize; ++i)
    {
        Index cell_index = cell_indices[i];
        my_cell_point_counts[cell_index]++;
        my_cell_atom_counts[cell_index] += (*this)[i].size();
    }

    MPI_Allreduce(my_cell_point_counts.data(), cell_point_counts.data(), (int)num_centers, MPI_INDEX, MPI_SUM, comm);
    MPI_Allreduce(my_cell_atom_counts.data(), cell_atom_counts.data(), (int)num_centers, MPI_INDEX, MPI_SUM, comm);

    std::vector<int> dests(num_centers);
    IndexPairVector pairs;

    for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
    {
        pairs.emplace_back(cell_atom_counts[cell_index], cell_index);
    }

    std::sort(pairs.rbegin(), pairs.rend());

    IndexVector bins(nprocs, 0);

    for (const auto& [size, cell_index] : pairs)
    {
        int dest = std::min_element(bins.begin(), bins.end()) - bins.begin();
        bins[dest] += size;
        dests[cell_index] = dest;
    }

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    Index my_assigned_cells = 0;

    IndexVector cellmap(num_centers);
    IndexVector rankcounts(nprocs, 0);

    for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
    {
        int dest = dests[cell_index];
        cellmap[cell_index] = rankcounts[dest]++;
        if (dest == myrank) my_assigned_cells++;
    }

    std::vector<int> sendcounts(nprocs,0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);
    std::vector<int> sendcounts_atoms(nprocs,0), recvcounts_atoms(nprocs), sdispls_atoms(nprocs), rdispls_atoms(nprocs);

    struct PointEnvelope
    {
        Index id;
        Index cell;
        Index size;
        Real dist;

        PointEnvelope() {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    Index totsend, totrecv, totsend_atoms, totrecv_atoms;
    AtomVector sendbuf_atoms, recvbuf_atoms;
    PointEnvelopeVector sendbuf_envs, recvbuf_envs;

    for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
    {
        int dest = dests[cell_index];
        sendcounts[dest] += my_cell_point_counts[cell_index];
        sendcounts_atoms[dest] += my_cell_atom_counts[cell_index];
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    MPI_Alltoall(sendcounts_atoms.data(), 1, MPI_INT, recvcounts_atoms.data(), 1, MPI_INT, comm);

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

    std::exclusive_scan(sendcounts_atoms.begin(), sendcounts_atoms.end(), sdispls_atoms.begin(), 0);
    std::exclusive_scan(recvcounts_atoms.begin(), recvcounts_atoms.end(), rdispls_atoms.begin(), 0);

    totsend = sendcounts.back() + sdispls.back();
    totrecv = recvcounts.back() + rdispls.back();

    totsend_atoms = sendcounts_atoms.back() + sdispls_atoms.back();
    totrecv_atoms = recvcounts_atoms.back() + rdispls_atoms.back();

    sendbuf_atoms.resize(totsend_atoms), recvbuf_atoms.resize(totrecv_atoms);
    sendbuf_envs.resize(totsend), recvbuf_envs.resize(totrecv);

    auto sptrs = sdispls;

    for (Index i = 0; i < totsend; ++i)
    {
        Index cell_index = cell_indices[i];
        int dest = dests[cell_index];
        Index loc = sptrs[dest]++;
        Point<Atom> pt = (*this)[i];

        sendbuf_envs[loc].id = i+myoffset;
        sendbuf_envs[loc].cell = cell_index;
        sendbuf_envs[loc].size = pt.size();
        sendbuf_envs[loc].dist = dist_to_centers[i];
    }

    auto it = sendbuf_atoms.begin();

    for (Index i = 0; i < totsend; ++i)
    {
        Index id = sendbuf_envs[i].id - myoffset;
        Point<Atom> pt = (*this)[id];

        it = std::copy(pt.begin(), pt.end(), it);
        assert((pt.size() == sendbuf_envs[i].size));
    }

    MPI_Request reqs[2];

    MPI_Ialltoallv(sendbuf_envs.data(), sendcounts.data(), sdispls.data(), MPI_POINT_ENVELOPE,
                   recvbuf_envs.data(), recvcounts.data(), rdispls.data(), MPI_POINT_ENVELOPE, comm, &reqs[0]);

    MPI_Ialltoallv(sendbuf_atoms.data(), sendcounts_atoms.data(), sdispls_atoms.data(), MPI_ATOM,
                   recvbuf_atoms.data(), recvcounts_atoms.data(), rdispls_atoms.data(), MPI_ATOM, comm, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    IndexVector cellcounts(my_assigned_cells, 0);
    IndexVector recv_offsets(totrecv);

    Index disp = 0;

    for (Index i = 0; i < totrecv; ++i)
    {
        Index cell = recvbuf_envs[i].cell;
        Index size = recvbuf_envs[i].size;

        cellcounts[cellmap[cell]]++;

        recv_offsets[i] = disp;
        disp += size;
    }

    std::vector<std::vector<Point<Atom>>> cell_points(my_assigned_cells);
    std::vector<RealVector> cell_dist_to_centers(my_assigned_cells);
    IndexVector cell_center_offsets(my_assigned_cells);
    IndexVector global_cell_indices(my_assigned_cells);

    for (Index i = 0; i < totrecv; ++i)
    {
        Index cell_index = cellmap[recvbuf_envs[i].cell];
        Index size = recvbuf_envs[i].size;
        global_cell_indices[cell_index] = recvbuf_envs[i].cell;

        if (centers[recvbuf_envs[i].cell].id() == recvbuf_envs[i].id)
            cell_center_offsets[cell_index] = cell_points[cell_index].size();

        const Atom *mem = &recvbuf_atoms[recv_offsets[i]];
        cell_points[cell_index].emplace_back(mem, size, recvbuf_envs[i].id);
        cell_dist_to_centers[cell_index].push_back(recvbuf_envs[i].dist);
    }

    mycells.clear();
    mycells.reserve(my_assigned_cells);

    for (Index cell = 0; cell < my_assigned_cells; ++cell)
    {
        auto& pts = cell_points[cell];
        auto& dists = cell_dist_to_centers[cell];

        if (!pts.empty())
        {
            std::swap(pts[0], pts[cell_center_offsets[cell]]);
            std::swap(dists[0], dists[cell_center_offsets[cell]]);
        }

        mycells.emplace_back(pts, dists, global_cell_indices[cell]);
    }

    MPI_Type_free(&MPI_POINT_ENVELOPE);
}

template <class Atom_>
template <class Distance>
void VoronoiCell<Atom_>::add_ghost_points(std::vector<VoronoiCell>& cells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    struct PointEnvelope
    {
        Index id;
        Index size;
        Real dist;

        PointEnvelope() {}
        PointEnvelope(Index id, Index size, Real dist) : id(id), size(size), dist(dist) {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    AtomVector sendbuf_atoms;
    AtomVector recvbuf_atoms;

    PointEnvelopeVector sendbuf_envs;
    PointEnvelopeVector recvbuf_envs;

    Index my_assigned_cells = cells.size();
    Index my_assigned_points = 0;
    Index my_assigned_atoms = 0;

    for (const VoronoiCell& cell : cells)
    {
        my_assigned_points += cell.num_points();
        my_assigned_atoms += cell.num_atoms();
    }

    sendbuf_atoms.reserve(my_assigned_atoms);
    sendbuf_envs.reserve(my_assigned_points);

    AtomVector mycenters_buf;
    IndexVector mycenters_sizes;
    IndexVector mycenters_ids;

    std::vector<CoverTree> trees;

    for (const VoronoiCell& cell : cells)
    {
        Index cell_point_count = cell.num_points();

        for (Index i = 0; i < cell_point_count; ++i)
        {
            Point<Atom> p = cell[i];
            sendbuf_envs.emplace_back(p.id(), p.size(), cell.dist_to_centers[i]);
            sendbuf_atoms.insert(sendbuf_atoms.end(), p.begin(), p.end());
        }

        if (cell_point_count >= 1)
        {
            Point<Atom> p = cell[0];

            mycenters_ids.push_back(mycenters_ids.size());
            mycenters_sizes.push_back(p.size());
            std::copy(p.begin(), p.end(), std::back_inserter(mycenters_buf));
        }

        trees.emplace_back(cover, leaf_size);
        trees.back().build(cell, distance);
    }

    PointContainer<Atom> mycenters(mycenters_buf, mycenters_sizes, mycenters_ids);

    std::vector<IndexSet> treeids_set(my_assigned_cells);

    for (Index cell_index = 0; cell_index < my_assigned_cells; ++cell_index)
    {
        treeids_set[cell_index].insert(cells[cell_index].ids.begin(), cells[cell_index].ids.end());
    }

    CoverTree mycentertree(cover, 1);
    mycentertree.build(mycenters, distance);

    MPI_Request reqs[8];

    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;

    int sendtarg = myrank;
    int recvtarg;

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    int sendcount_buf[2], recvcount_buf[2];

    IndexVector ghostcells;
    auto functor = [&](const Point<Atom>& p, const Point<Atom>& q, Real dist) { ghostcells.push_back(p.id()); };

    for (int step = 0; step <= nprocs/2; ++step)
    {
        recvtarg = (sendtarg+1)%nprocs;
        sendcount = sendbuf_envs.size();
        sendcount_atoms = sendbuf_atoms.size();

        sendcount_buf[0] = sendcount;
        sendcount_buf[1] = sendcount_atoms;

        double t = -MPI_Wtime();
        MPI_Irecv(recvcount_buf, 2, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
        MPI_Isend(sendcount_buf, 2, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        t += MPI_Wtime();

        recvcount = recvcount_buf[0];
        recvcount_atoms = recvcount_buf[1];

        recvbuf_atoms.resize(recvcount_atoms);
        recvbuf_envs.resize(recvcount);

        MPI_Irecv(recvbuf_atoms.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
        MPI_Isend(sendbuf_atoms.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

        MPI_Irecv(recvbuf_envs.data(), recvcount, MPI_POINT_ENVELOPE, recvrank, myrank+2*nprocs, comm, &reqs[2]);
        MPI_Isend(sendbuf_envs.data(), sendcount, MPI_POINT_ENVELOPE, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

        Index targsize = sendcount;

        const Atom* mem = sendbuf_atoms.data();

        for (Index i = 0; i < targsize; ++i)
        {
            Index dim = sendbuf_envs[i].size;
            Index index = sendbuf_envs[i].id;
            Real dist = sendbuf_envs[i].dist;

            Point<Atom> query(mem, dim, index);
            mem += dim;

            ghostcells.clear();
            mycentertree.radius_query(mycenters, distance, query, dist + 2*radius, functor);

            if (ghostcells.empty())
                continue;

            for (Index cell : ghostcells)
                if (cells[cell].num_points() != 0 && !treeids_set[cell].contains(index))
                {
                    if (trees[cell].has_radius_neighbor(cells[cell], distance, query, radius))
                    {
                        cells[cell].add_ghost_point(query);
                    }

                    treeids_set[cell].insert(index);
                }
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        sendtarg = recvtarg;
        sendbuf_atoms.swap(recvbuf_atoms);
        sendbuf_envs.swap(recvbuf_envs);
    }

    MPI_Type_free(&MPI_POINT_ENVELOPE);
}

template <class Atom_>
template <class Distance>
void VoronoiCell<Atom_>::add_ghost_points_rips(std::vector<VoronoiCell>& cells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    struct PointEnvelope
    {
        Index id;
        Index size;
        Real dist;

        PointEnvelope() {}
        PointEnvelope(Index id, Index size, Real dist) : id(id), size(size), dist(dist) {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT_ENVELOPE;
    MPI_Type_contiguous(sizeof(PointEnvelope), MPI_CHAR, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    AtomVector sendbuf_atoms;
    AtomVector recvbuf_atoms;

    PointEnvelopeVector sendbuf_envs;
    PointEnvelopeVector recvbuf_envs;

    IndexVector sendbuf_shared;
    IndexVector recvbuf_shared;

    Index my_assigned_cells = cells.size();
    Index my_assigned_points = 0;
    Index my_assigned_atoms = 0;

    for (const VoronoiCell& cell : cells)
    {
        my_assigned_points += cell.num_points();
        my_assigned_atoms += cell.num_atoms();
    }

    sendbuf_atoms.reserve(my_assigned_atoms);
    sendbuf_envs.reserve(my_assigned_points);
    sendbuf_shared.resize(my_assigned_points, 0);

    AtomVector mycenters_buf;
    IndexVector mycenters_sizes;
    IndexVector mycenters_ids;

    std::vector<CoverTree> trees;

    for (const VoronoiCell& cell : cells)
    {
        Index cell_point_count = cell.num_points();

        for (Index i = 0; i < cell_point_count; ++i)
        {
            Point<Atom> p = cell[i];
            sendbuf_envs.emplace_back(p.id(), p.size(), cell.dist_to_centers[i]);
            sendbuf_atoms.insert(sendbuf_atoms.end(), p.begin(), p.end());
        }

        if (cell_point_count >= 1)
        {
            Point<Atom> p = cell[0];

            mycenters_ids.push_back(mycenters_ids.size());
            mycenters_sizes.push_back(p.size());
            std::copy(p.begin(), p.end(), std::back_inserter(mycenters_buf));
        }

        trees.emplace_back(cover, leaf_size);
        trees.back().build(cell, distance);
    }

    PointContainer<Atom> mycenters(mycenters_buf, mycenters_sizes, mycenters_ids);

    std::vector<IndexSet> treeids_set(my_assigned_cells);

    for (Index cell_index = 0; cell_index < my_assigned_cells; ++cell_index)
    {
        treeids_set[cell_index].insert(cells[cell_index].ids.begin(), cells[cell_index].ids.end());
    }

    CoverTree mycentertree(cover, 1);
    mycentertree.build(mycenters, distance);

    MPI_Request reqs[8];

    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;

    int sendtarg = myrank;
    int recvtarg;

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    int sendcount_buf[2], recvcount_buf[2];

    IndexVector ghostcells;
    auto functor = [&](const Point<Atom>& p, const Point<Atom>& q, Real dist) { ghostcells.push_back(p.id()); };

    for (int step = 0; step <= nprocs; ++step)
    {
        recvtarg = (sendtarg+1)%nprocs;
        sendcount = sendbuf_envs.size();
        sendcount_atoms = sendbuf_atoms.size();

        sendcount_buf[0] = sendcount;
        sendcount_buf[1] = sendcount_atoms;

        double t = -MPI_Wtime();
        MPI_Irecv(recvcount_buf, 2, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
        MPI_Isend(sendcount_buf, 2, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        t += MPI_Wtime();

        recvcount = recvcount_buf[0];
        recvcount_atoms = recvcount_buf[1];

        recvbuf_atoms.resize(recvcount_atoms);
        recvbuf_envs.resize(recvcount);

        MPI_Irecv(recvbuf_atoms.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
        MPI_Isend(sendbuf_atoms.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

        MPI_Irecv(recvbuf_envs.data(), recvcount, MPI_POINT_ENVELOPE, recvrank, myrank+2*nprocs, comm, &reqs[2]);
        MPI_Isend(sendbuf_envs.data(), sendcount, MPI_POINT_ENVELOPE, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

        Index targsize = sendcount;

        const Atom* mem = sendbuf_atoms.data();

        for (Index i = 0; i < targsize; ++i)
        {
            Index dim = sendbuf_envs[i].size;
            Index index = sendbuf_envs[i].id;
            Real dist = sendbuf_envs[i].dist;

            Point<Atom> query(mem, dim, index);
            mem += dim;

            ghostcells.clear();
            mycentertree.radius_query(mycenters, distance, query, dist + 2*radius, functor);

            if (ghostcells.empty())
                continue;

            for (Index cell : ghostcells)
                if (cells[cell].num_points() != 0 && !treeids_set[cell].contains(index))
                {
                    if (trees[cell].has_radius_neighbor(cells[cell], distance, query, radius))
                    {
                        cells[cell].add_ghost_point(query);
                        sendbuf_shared[i]++;
                    }

                    treeids_set[cell].insert(index);
                }
        }

        recvbuf_shared.resize(recvcount);

        MPI_Irecv(recvbuf_shared.data(), recvcount, MPI_INDEX, recvrank, myrank+3*nprocs, comm, &reqs[4]);
        MPI_Isend(sendbuf_shared.data(), sendcount, MPI_INDEX, sendrank, sendrank+3*nprocs, comm, &reqs[5]);

        MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE);

        sendtarg = recvtarg;
        sendbuf_atoms.swap(recvbuf_atoms);
        sendbuf_envs.swap(recvbuf_envs);
        sendbuf_shared.swap(recvbuf_shared);
    }

    for (Index cell_index = 0, i = 0; cell_index < my_assigned_cells; ++cell_index)
    {
        Index n = cells[cell_index].num_points();

        for (Index j = 0; j < n; ++i, ++j)
        {
            if (recvbuf_shared[i] == 0)
            {
                cells[cell_index].set_interior(j);
            }
        }
    }

    MPI_Type_free(&MPI_POINT_ENVELOPE);
}

template <class Atom_>
VoronoiComplex<Atom_>::VoronoiComplex(const VoronoiCell<Atom>& cell, Real radius, Index maxdim)
    : PointContainer<Atom>(cell, cell.ghosts()),
      radius(radius),
      maxdim(maxdim),
      local(cell.num_points()),
      interior(cell.interiors()) {}

template <class Atom_>
template <class Distance>
void VoronoiComplex<Atom_>::build_filtration(Distance& distance, Real cover, Index leaf_size)
{
    using WeightMap = std::unordered_map<Index, Real>;
    using WeightMapVector = std::vector<WeightMap>;

    Index n = PointContainer<Atom>::num_points();

    IndexSetVector graph(n);
    WeightMapVector weights(maxdim+1);

    CoverTree tree(cover, leaf_size);
    tree.build(*this, distance);

    auto query_functor = [&](const Point<Atom>& p, const Point<Atom>& q, Real dist)
    {
        graph[q.offset()].insert(p.offset());
    };

    for (Index i = 0; i < n; ++i)
    {
        tree.radius_query(*this, distance, (*this)[i], radius, query_functor);
    }

    auto neighbor_functor = [&](Index u, Index v) { return graph[u].find(v) != graph[u].end(); };
    auto simplex_functor = [&](const IndexVector& s)
    {
        bool is_interior = false;

        for (Index v : s)
        {
            if (interior[v])
            {
                is_interior = true;
                break;
            }
        }

        Index p = s.size()-1;

        filtration.emplace_back(s, is_interior);
        const WeightedSimplex& sigma = filtration.back();

        if (p == 0) weights[0].insert({sigma.getid(), 0.});
        else if (p == 1) weights[1].insert({sigma.getid(), distance((*this)[s[0]], (*this)[s[1]])});
        else weights[p].insert({sigma.getid(), 0.});
    };

    IndexVector current;
    IndexVector candidates(n);

    std::iota(candidates.begin(), candidates.end(), (Index)0);

    bron_kerbosch(current, candidates, -1, neighbor_functor, simplex_functor);

    for (Index p = 2; p <= maxdim; ++p)
    {
        for (auto& [id, weight] : weights[p])
        {
            weight = 0;
            Simplex sigma(id);
            IndexVector facet_ids;
            sigma.get_facet_ids(facet_ids);

            for (Index fid : facet_ids)
            {
                weight = std::max(weight, weights[p-1][fid]);
            }
        }
    }

    for (auto& s : filtration)
    {
        Index id = s.getid();
        Index dim = s.getdim();

        s.value = weights[dim][id];
    }

    for (auto& s : filtration)
    {
        IndexVector verts = s.getverts();

        for (Index& v : verts)
        {
            v = PointContainer<Atom>::id(v);
        }

        Index newid = Simplex(verts).getid();
        s.id = newid;
    }

    std::sort(filtration.begin(), filtration.end());
}

template <class Atom_>
template <class Functor, class NeighborTest>
void VoronoiComplex<Atom_>::bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborTest& neighbor, const Functor& functor) const
{
    if (!current.empty())
        functor(current);

    if (current.size() == static_cast<size_t>(maxdim) + 1)
        return;

    Index m = cands.size();

    for (Index j = excluded+1; j < m; ++j)
    {
        current.push_back(cands[j]);

        IndexVector new_cands;

        for (Index i = 0; i < j; ++i)
            if (neighbor(cands[i], cands[j]))
                new_cands.push_back(cands[i]);

        Index ex = new_cands.size();

        for (Index i = j+1; i < m; ++i)
            if (neighbor(cands[i], cands[j]))
                new_cands.push_back(cands[i]);

        excluded = ex-1;

        bron_kerbosch(current, new_cands, excluded, neighbor, functor);
        current.pop_back();
    }
}

template <class Atom_>
void VoronoiComplex<Atom_>::write_filtration_file(const char *fname, bool use_ids) const
{
    FILE *f;

    f = fopen(fname, "w");

    for (const auto& s : filtration)
    {
        if (use_ids)
        {
            fprintf(f, "%f\t%lld\t%d\n", s.value, s.getid(), static_cast<int>(s.interior));
        }
        else
        {
            IndexVector verts = s.getverts();
            Index n = verts.size();

            std::stringstream ss;
            ss << "<";

            for (Index i = 0; i < n-1; ++i)
            {
                ss << verts[i] << ",";
            }

            ss << verts[n-1] << ">";

            std::string st = ss.str();
            fprintf(f, "%f\t%s\t%d\n", s.value, st.c_str(), static_cast<int>(s.interior));
        }
    }

    fclose(f);
}
