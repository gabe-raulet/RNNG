#ifndef SIMPLEX_H_
#define SIMPLEX_H_

#include "utils.h"
#include <limits>

static Binom binom;

struct Simplex
{
    Simplex();
    Simplex(Index id);
    Simplex(const IndexVector& verts);

    Index getid() const;
    Index getuid() const;
    Index getdim() const;
    IndexVector getverts(Index n) const;

    void get_facets(std::vector<Simplex>& facets, Index n) const;
    void get_facet_ids(IndexVector& ids, Index n) const;

    Index id;

    std::string get_simplex_repr(Index n) const;
};

struct WeightedSimplex : public Simplex
{
    WeightedSimplex(const IndexVector& verts, bool interior) : Simplex(verts), value(0), interior(interior) {}

    friend bool operator<(const WeightedSimplex& lhs, const WeightedSimplex& rhs) { return std::tie(lhs.value, lhs.id) < std::tie(rhs.value, rhs.id); }
    friend bool operator==(const WeightedSimplex& lhs, const WeightedSimplex& rhs) { return (lhs.id == rhs.id); }
    friend bool operator!=(const WeightedSimplex& lhs, const WeightedSimplex& rhs) { return (lhs.id != rhs.id); }

    Real getvalue() const { return value; }

    Real value;
    bool interior;
};

struct SimplexEnvelope
{
    Index id;
    Real value;

    SimplexEnvelope() {}
    SimplexEnvelope(Index id, Real value) : id(id), value(value) {}
    SimplexEnvelope(const WeightedSimplex& simplex) : id(simplex.getid()), value(simplex.getvalue()) {}

    friend bool operator<(const SimplexEnvelope& lhs, const SimplexEnvelope& rhs) { return std::tie(lhs.value, lhs.id) < std::tie(rhs.value, rhs.id); }
    friend bool operator==(const SimplexEnvelope& lhs, const SimplexEnvelope& rhs) { return (lhs.id == rhs.id); }
    friend bool operator!=(const SimplexEnvelope& lhs, const SimplexEnvelope& rhs) { return (lhs.id != rhs.id); }
};

void merge_and_write_filtration(const char *fname, const std::vector<WeightedSimplex>& mysimplices, Index num_vertices, bool use_ids, MPI_Comm comm);

#include "simplex.hpp"

#endif
