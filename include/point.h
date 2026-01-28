#ifndef POINT_H_
#define POINT_H_

#include "utils.h"

template <class Atom_>
class Point
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using AtomIter = typename AtomVector::const_iterator;

        Point(const Atom *mem, Index dim) : mem(mem), dim(dim), index(-1) {}
        Point(const Atom *mem, Index dim, Index index) : mem(mem), dim(dim), index(index) {}

        Index id() const { return index; }
        Index size() const { return dim; }
        Atom operator[](Index i) const { return mem[i]; }
        explicit operator const Atom *() const { return mem; }

        const Atom* begin() const { return mem; }
        const Atom* end() const { return mem+dim; }

    private:

        const Atom *mem;
        Index dim, index;
};

template <class Atom_>
class PointContainer
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;

        PointContainer() {}
        PointContainer(const AtomVector& atoms, const IndexVector& sizes);
        PointContainer(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);

        PointContainer(const AtomVector& atoms, Index size, Index dim);
        PointContainer(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);

        Index num_points() const { return ids.size(); }
        Index num_atoms() const { return data.size(); }

        Point<Atom> operator[](Index i) const { return Point<Atom>(mem(i), size(i), id(i)); }

        Index read_fvecs(const char *fname);
        Index read_seqs(const char *fname);

        Index read_fvecs(const char *fname, MPI_Comm comm);
        Index read_seqs(const char *fname, MPI_Comm comm);

    private:

        AtomVector data;
        IndexVector offsets, ids;

        void init(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);
        void init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);

        inline const Atom* mem(Index i) const { return &data[offsets[i]]; }
        inline Index size(Index i) const { return offsets[i+1]-offsets[i]; }
        inline Index id(Index i) const { return ids[i]; }
};

#include "point.hpp"

#endif
