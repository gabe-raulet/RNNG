#ifndef POINT_H_
#define POINT_H_

#include "utils.h"
#include "simplex.h"

template <class Atom_>
class Point
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using AtomIter = typename AtomVector::const_iterator;

        Point(const Atom *mem, Index dim, Index index) : mem(mem), dim(dim), index(index), container_offset(-1) {}
        Point(const Atom *mem, Index dim, Index index, Index container_offset) : mem(mem), dim(dim), index(index), container_offset(container_offset) {}

        Index id() const { return index; }
        Index size() const { return dim; }
        Index offset() const { return container_offset; }
        Atom operator[](Index i) const { return mem[i]; }
        explicit operator const Atom *() const { return mem; }

        const Atom* begin() const { return mem; }
        const Atom* end() const { return mem+dim; }

    private:

        const Atom *mem;
        Index dim, index, container_offset;
};

template <class Atom_>
class PointContainer
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;

        PointContainer() : offsets({0}) {}
        PointContainer(const std::vector<Point<Atom>>& points);
        PointContainer(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);
        PointContainer(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);
        PointContainer(const PointContainer& lhs, const PointContainer& rhs);

        Index num_points() const { return ids.size(); }
        Index num_atoms() const { return data.size(); }

        Point<Atom> operator[](Index i) const { return Point<Atom>(mem(i), size(i), id(i), i); }

        Index read_fvecs(const char *fname);
        Index read_seqs(const char *fname);

        Index read_fvecs(const char *fname, MPI_Comm comm);
        Index read_seqs(const char *fname, MPI_Comm comm);

        struct SendrecvRequest
        {
            MPI_Request reqs[6];
            void wait() { MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE); }
        };

        void sendrecv(PointContainer& recvbuf, int recvrank, int sendrank, MPI_Comm comm, SendrecvRequest& req);
        void swap(PointContainer& other);

        void allgather(const PointContainer& sendbuf, MPI_Comm comm);
        void indexed_gather(const PointContainer& sendbuf, const IndexVector& index_offsets);

        void push_back(const Point<Atom>& p);

    protected:

        AtomVector data;
        IndexVector offsets, ids;

        void init(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);
        void init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);

        inline const Atom* mem(Index i) const { return &data[offsets[i]]; }
        inline Index size(Index i) const { return offsets[i+1]-offsets[i]; }
        inline Index id(Index i) const { return ids[i]; }
};

template <class Atom_>
class VoronoiCell : public PointContainer<Atom_>
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;

        VoronoiCell(const std::vector<Point<Atom>>& points, const RealVector& dist_to_centers, Index cell_index);

        void add_ghost_point(const Point<Atom>& p);

        Index id() const { return cell_index; }
        Index num_ghosts() const { return ghost_points.num_points(); }

        template <class Distance>
        static void add_ghost_points(std::vector<VoronoiCell>& cells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm);

        Point<Atom> ghost(Index i) const { return ghost_points[i]; }
        const PointContainer<Atom>& ghosts() const { return ghost_points; }
        const BoolVector& interiors() const { return interior; }

        void set_interior(Index offset) { interior[offset] = true; }

    protected:

        Index cell_index;
        RealVector dist_to_centers;
        BoolVector interior;
        PointContainer<Atom> ghost_points;

    private:

        using PointContainer<Atom>::data;
        using PointContainer<Atom>::offsets;
        using PointContainer<Atom>::ids;
};

template <class Atom_>
class VoronoiComplex : public PointContainer<Atom_>
{
    public:

        using Atom = Atom_;

        VoronoiComplex(const VoronoiCell<Atom>& cell, Real radius, Index maxdim);

        template <class Distance>
        void build_filtration(Distance& distance, Real cover, Index leaf_size);

        void write_filtration_file(const char *fname, bool use_ids=true) const;

    private:

        Real radius;
        Index maxdim;
        Index local;
        BoolVector interior;

        std::vector<WeightedSimplex> filtration;

        template <class Functor, class NeighborTest>
        void bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborTest& neighbor, const Functor& functor) const;
};

template <class Atom_>
class VoronoiDiagram : public PointContainer<Atom_>
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using Cell = VoronoiCell<Atom>;

        template <class Distance>
        VoronoiDiagram(const PointContainer<Atom>& points, const PointContainer<Atom>& centers, Distance& distance);

        void coalesce_cells(std::vector<Cell>& mycells, MPI_Comm comm) const;

    private:

        PointContainer<Atom> centers;
        IndexVector cell_indices;
        RealVector dist_to_centers;
};

#include "point.hpp"

#endif
