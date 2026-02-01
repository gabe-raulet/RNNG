#ifndef SEARCH_H_
#define SEARCH_H_

#include "utils.h"
#include "point.h"

template <class T>
class Search
{
    public:

        template <class Atom, class Distance>
        void build(const PointContainer<Atom>& points, Distance& distance)
        {
            static_cast<T*>(this)->build(points, distance);
        }

        template <class Atom, class Distance, class Functor>
        Index radius_query(const PointContainer<Atom>& points, Distance& distance, const Point<Atom>& query, const Functor& functor) const
        {
            return static_cast<T*>(this)->radius_query(points, distance, query, functor);
        }
};

class BruteForce : public Search<BruteForce>
{
    public:

        BruteForce() {}

        template <class Atom, class Distance>
        void build(const PointContainer<Atom>& points, Distance& distance);

        template <class Atom, class Distance, class Functor>
        Index radius_query(const PointContainer<Atom>& points, Distance& distance, const Point<Atom>& query, Real radius, const Functor& functor) const;
};

class CoverTree : public Search<CoverTree>
{
    public:

        CoverTree(Real cover=1.5, Index leaf_size=1) : cover(cover), leaf_size(leaf_size) {}

        template <class Atom, class Distance>
        void build(const PointContainer<Atom>& points, Distance& distance);

        template <class Atom, class Distance, class Functor>
        Index radius_query(const PointContainer<Atom>& points, Distance& distance, const Point<Atom>& query, Real radius, const Functor& functor) const;

        template <class Atom, class Distance, class Functor>
        Index radius_query_batched(const PointContainer<Atom>& points, Distance& distance, const PointContainer<Atom>& batch, Real radius, const Functor& functor) const;

        template <class Atom, class Distance>
        bool has_radius_neighbor(const PointContainer<Atom>& points, Distance& distance, const Point<Atom>& query, Real radius) const;

    private:

        Real cover;
        Index leaf_size;

        IndexVector childarr; /* size m-1; children array */
        IndexVector childptrs; /* size m+1; children pointers */
        IndexVector centers; /* size m; vertex centers */
        RealVector radii; /* size m; vertex radii */

        IndexIter child_begin(Index vertex) const { return childarr.begin() + childptrs[vertex]; }
        IndexIter child_end(Index vertex) const { return childarr.begin() + childptrs[vertex+1]; }

        void clear_tree() { childarr.clear(); childptrs.clear(); centers.clear(); radii.clear(); }
        void allocate(Index num_verts);
};

#include "search.hpp"

#endif
