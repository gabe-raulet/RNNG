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

        PointContainer() : offsets({0}) {}
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

        struct SendrecvRequest
        {
            MPI_Request reqs[6];
            void wait() { MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE); }
        };

        void sendrecv(PointContainer& recvbuf, int recvrank, int sendrank, MPI_Comm comm, SendrecvRequest& req);
        void swap(PointContainer& other);

        void allgather(const PointContainer& sendbuf, MPI_Comm comm);
        void indexed_gather(const PointContainer& sendbuf, const IndexVector& index_offsets);

        void clear() { data.clear(); offsets.assign({0}); ids.clear(); }
        void reserve_atoms(Index atom_count) { data.reserve(atom_count); }

        void push_back(const Point<Atom>& p);

    private:

        AtomVector data;
        IndexVector offsets, ids;

        void init(const AtomVector& atoms, const IndexVector& sizes, const IndexVector& indices);
        void init(const AtomVector& atoms, Index size, Index dim, const IndexVector& indices);

        inline const Atom* mem(Index i) const { return &data[offsets[i]]; }
        inline Index size(Index i) const { return offsets[i+1]-offsets[i]; }
        inline Index id(Index i) const { return ids[i]; }
};

template <class Atom_>
class LocalCell
{
    public:

        using Atom = Atom_;

        LocalCell(const PointContainer<Atom>& points, const RealVector& dists, Index cell_index) : points(points), dists(dists), cell_index(cell_index) {}

        friend std::ostream& operator<<(std::ostream& os, const LocalCell& cell)
        {
            char buf[1024];
            snprintf(buf, 1024, "LocalCell(cell_index=%lld, points=%lld, atoms=%lld)", cell.cell_index, cell.points.num_points(), cell.points.num_atoms());
            os << buf;
            return os;
        }

    private:

        PointContainer<Atom> points;
        RealVector dists;
        Index cell_index;
};

template <class Atom_>
class Diagram
{
    public:

        using Atom = Atom_;

        Diagram(const PointContainer<Atom>& points);

        template <class Distance>
        void random_partition(Index num_centers, const Distance& distance, int rng_seed, MPI_Comm comm);

        Index num_landmarks() const { return landmarks.size(); }

        void coalesce_local_cells(std::vector<LocalCell<Atom>>& cells) const;

    private:

        PointContainer<Atom> points;
        PointContainer<Atom> centers;

        IndexVector mycells;
        RealVector mydists;
        IndexVector landmarks;
};

#include "point.hpp"

#endif
