void Graph::write_file(const char *fname) const
{
    FILE *f;
    Index num_edges;

    f = fopen(fname, "w");

    num_edges = edges.size();

    fprintf(f, "%% %lld %lld %lld\n", num_vertices, num_vertices, num_edges);

    for (const auto& [i, j, w] : edges)
    {
        fprintf(f, "%lld %lld %.6f\n", i, j, w);
    }

    fclose(f);
}

void Graph::write_file(const char *fname, MPI_Comm comm) const
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index num_edges = edges.size();

    MPI_Allreduce(MPI_IN_PLACE, &num_edges, 1, MPI_INDEX, MPI_SUM, comm);

    std::ostringstream ss;
    if (!myrank) ss << "% " << num_vertices << " " << num_vertices << " " << num_edges << "\n";

    ss << std::setprecision(6);

    for (const auto& [u, v, w] : edges)
        ss << u << " " << v << " " << w << "\n";

    std::string s = ss.str();
    std::vector<char> buf(s.begin(), s.end());

    assert((buf.size() <= std::numeric_limits<int>::max()));

    MPI_Offset mycount = buf.size(), fileoffset, filesize;
    MPI_Exscan(&mycount, &fileoffset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (!myrank) fileoffset = 0;

    int truncate = 0;

    MPI_File fh;
    MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &filesize);
    truncate = (filesize > 0);
    MPI_Bcast(&truncate, 1, MPI_INT, 0, comm);
    if (truncate) MPI_File_set_size(fh, 0);
    MPI_File_write_at_all(fh, fileoffset, buf.data(), static_cast<int>(buf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void Graph::redistribute_edges(MPI_Comm comm)
{
    MPI_Datatype MPI_EDGE;
    MPI_Type_contiguous(sizeof(Edge), MPI_CHAR, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index chunk = num_vertices / nprocs;
    Index myoffset = myrank * chunk;
    Index mysize = (myrank == nprocs-1)? num_vertices - (nprocs-1)*chunk : chunk;

    std::vector<int> sendcounts(nprocs,0), sdispls(nprocs), recvcounts(nprocs), rdispls(nprocs);

    EdgeVector sendbuf, recvbuf;

    int dest;

    for (const auto& [u, v, w] : edges)
    {
        dest = std::min(static_cast<int>(u/chunk), nprocs-1);
        sendcounts[dest]++;

        dest = std::min(static_cast<int>(v/chunk), nprocs-1);
        sendcounts[dest]++;
    }

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
    sendbuf.resize(sendcounts.back()+sdispls.back());

    auto sptrs = sdispls;

    for (const auto& [u, v, w] : edges)
    {
        dest = std::min(static_cast<int>(u/chunk), nprocs-1);
        sendbuf[sptrs[dest]++] = std::make_tuple(u, v, w);

        dest = std::min(static_cast<int>(v/chunk), nprocs-1);
        sendbuf[sptrs[dest]++] = std::make_tuple(v, u, w);
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
    recvbuf.resize(recvcounts.back()+rdispls.back());

    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_EDGE,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_EDGE, comm);

    std::sort(recvbuf.begin(), recvbuf.end());
    recvbuf.erase(std::unique(recvbuf.begin(), recvbuf.end()), recvbuf.end());

    MPI_Type_free(&MPI_EDGE);

    std::swap(edges, recvbuf);
}
