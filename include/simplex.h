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
    IndexVector getverts(Index n = std::numeric_limits<Index>::max()) const;

    void get_facets(std::vector<Simplex>& facets) const;
    void get_facet_ids(IndexVector& ids) const;

    Index id;
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

#include "simplex.hpp"

#endif
