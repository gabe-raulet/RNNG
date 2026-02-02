Simplex::Simplex() : id(0) {}

Simplex::Simplex(Index id) : id(id) {}

Simplex::Simplex(const IndexVector& verts)
{
    IndexVector vertices(verts);
    std::sort(vertices.begin(), vertices.end());

    uint64_t p = vertices.size()-1;
    uint64_t uid = 0;

    for (Index i = p; i >= 0; --i)
    {
        uid += binom(vertices[i], i+1);
    }

    id = static_cast<Index>(uid | (p << 60));
}

Index Simplex::getid() const
{
    return id;
}

Index Simplex::getuid() const
{
    uint64_t _id = id;
    uint64_t _uid = _id & 0xFFFFFFFFFFFFFFF;
    return _uid;
}

Index Simplex::getdim() const
{
    uint64_t _dim;
    uint64_t _id = id;

    _dim = (_id >> 60) & 0xF;

    return _dim;
}

IndexVector Simplex::getverts(Index n) const
{
    auto get_max_vertex = [&n](size_t uid, size_t d)
    {
        int64_t left = 0;
        int64_t right = n;
        int64_t i;

        while (left < right)
        {
            i = left + ((right-left)>>1);

            if (binom(i,d+1) > uid)
                right = i;
            else
                left = i+1;
        }

        return right-1;
    };

    size_t uid = getuid();
    size_t dim = getdim();
    IndexVector vertices;

    for (Index i = dim; i >= 0; --i)
    {
        int64_t l = get_max_vertex(uid, i);
        vertices.push_back(l);
        uid -= binom(l, i+1);
    }

    if (vertices.size() > 1)
        std::sort(vertices.begin(), vertices.end());

    return vertices;
}

void Simplex::get_facets(std::vector<Simplex>& facets, Index n) const
{
    facets.clear();

    IndexVector vertices = getverts(n);
    Index dim = vertices.size()-1;

    for (Index i = 0; i <= dim; ++i)
    {
        IndexVector facet(vertices);

        for (Index j = i; j < dim; ++j)
            facet[j] = facet[j+1];

        facet.pop_back();
        facets.emplace_back(facet);
    }
}

void Simplex::get_facet_ids(IndexVector& ids, Index n) const
{
    ids.clear();

    std::vector<Simplex> facets;
    get_facets(facets, n);

    for (const auto& s : facets)
    {
        ids.push_back(s.getid());
    }
}

std::string Simplex::get_simplex_repr(Index n) const
{
    IndexVector verts = getverts(n);
    Index size = verts.size();

    std::stringstream ss;
    ss << "<";

    for (Index i = 0; i < size-1; ++i)
    {
        ss << verts[i] << ",";
    }

    ss << verts[size-1] << ">";
    return ss.str();
}

void merge_and_write_filtration(const char *fname, const std::vector<WeightedSimplex>& mysimplices, Index num_vertices, bool use_ids, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_SIMPLEX_ENVELOPE;
    MPI_Type_contiguous(sizeof(SimplexEnvelope), MPI_CHAR, &MPI_SIMPLEX_ENVELOPE);
    MPI_Type_commit(&MPI_SIMPLEX_ENVELOPE);

    int sendcount;
    std::vector<int> recvcounts, rdispls;
    std::vector<SimplexEnvelope> sendbuf, recvbuf;

    sendbuf.reserve(mysimplices.size());

    for (const WeightedSimplex& simplex : mysimplices)
        sendbuf.emplace_back(simplex);

    std::sort(sendbuf.begin(), sendbuf.end());
    sendbuf.erase(std::unique(sendbuf.begin(), sendbuf.end()), sendbuf.end());

    sendcount = sendbuf.size();

    if (!myrank)
    {
        recvcounts.resize(nprocs);
        rdispls.resize(nprocs);
    }

    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);

    if (!myrank)
    {
        std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
        recvbuf.resize(recvcounts.back() + rdispls.back());
    }

    MPI_Gatherv(sendbuf.data(), sendcount, MPI_SIMPLEX_ENVELOPE, recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_SIMPLEX_ENVELOPE, 0, comm);

    if (!myrank)
    {
        std::sort(recvbuf.begin(), recvbuf.end());
        recvbuf.erase(std::unique(recvbuf.begin(), recvbuf.end()), recvbuf.end());

        FILE *f;

        f = fopen(fname, "w");

        for (const auto& s : recvbuf)
        {
            if (use_ids)
            {
                fprintf(f, "%f\t%lld\n", s.value, s.id);
            }
            else
            {
                std::string st = Simplex(s.id).get_simplex_repr(num_vertices);
                fprintf(f, "%f\t%s\n", s.value, st.c_str());
            }
        }

        fclose(f);
    }

    MPI_Type_free(&MPI_SIMPLEX_ENVELOPE);
}
