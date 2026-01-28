#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "point.h"

class CoverTree
{
    public:

        CoverTree(Real cover=1.5, Index leaf_size=1) : cover(cover), leaf_size(leaf_size) {}

        template <class Atom, class Distance>
        void build(const PointContainer<Atom>& points, const Distance& distance);

        template <class Atom, class Distance, class Functor>
        Index radius_query(const PointContainer<Atom>& points, const Distance& distance, const Point<Atom>& query, Real radius, const Functor& functor) const;

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

#include "cover_tree.hpp"

#endif
