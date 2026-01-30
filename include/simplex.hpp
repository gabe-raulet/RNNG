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

void Simplex::get_facets(std::vector<Simplex>& facets) const
{
    facets.clear();

    IndexVector vertices = getverts();
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

void Simplex::get_facet_ids(IndexVector& ids) const
{
    ids.clear();

    std::vector<Simplex> facets;
    get_facets(facets);

    for (const auto& s : facets)
    {
        ids.push_back(s.getid());
    }
}
