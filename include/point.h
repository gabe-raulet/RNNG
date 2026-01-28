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

        Point(const AtomVector& data) : data(data), index(-1) {}
        Point(const AtomVector& data, Index index) : data(data), index(index) {}

        Point(const Atom *mem, Index dim) : data(mem, mem+dim), index(-1) {}
        Point(const Atom *mem, Index dim, Index index) : data(mem, mem+dim), index(index) {}

        template <class Iter> Point(Iter first, Iter last) : data(first, last), index(-1) {}
        template <class Iter> Point(Iter first, Iter last, Index index) : data(first, last), index(index) {}

        Index id() const { return index; }
        Index size() const { return data.size(); }
        Atom operator[](Index i) const { return data[i]; }
        explicit operator const Atom *() const { return data.data(); }

        AtomIter begin() const { return data.cbegin(); }
        AtomIter end() const { return data.cend(); }

    private:

        AtomVector data;
        Index index;
};

template <class Atom_>
class PointContainer
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using PointType = Point<Atom>;

        PointContainer() {}
        PointContainer(const AtomVector& atoms, const IndexVector& sizes);
        PointContainer(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);

        PointContainer(const AtomVector& atoms, Index size, Index dim);
        PointContainer(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);

        Index num_points() const { return points.size(); }
        Index num_atoms() const { return std::accumulate(points.begin(), points.end(), (Index)0, [](const Index& lhs, const PointType& rhs) { return lhs + rhs.size(); });  }

        PointType operator[](Index i) const { return points[i]; }

        Index read_fvecs(const char *fname);
        Index read_seqs(const char *fname);

    private:

        std::vector<PointType> points;

        void init(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);
        void init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);
};

#include "point.hpp"

#endif
