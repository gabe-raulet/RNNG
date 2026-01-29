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
class VoronoiDiagram : public PointContainer<Atom_>
{
    public:

        using Atom = Atom_;

        VoronoiDiagram(const PointContainer<Atom>& points, const PointContainer<Atom>& centers)
            : PointContainer<Atom>(points),
              centers(centers),
              cell_indices(points.num_points(), 0),
              dist_to_centers(points.num_points(), std::numeric_limits<Real>::max()) {}

        template <class Distance>
        void compute_point_partitioning(const Distance& distance)
        {
            Index size = PointContainer<Atom>::num_points();
            Index num_centers = centers.num_points();

            for (Index i = 0; i < size; ++i)
            {
                for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
                {
                    Real dist = distance(centers[cell_index], (*this)[i]);

                    if (dist <= dist_to_centers[i])
                    {
                        dist_to_centers[i] = dist;
                        cell_indices[i] = cell_index;
                    }
                }
            }
        }

    private:

        PointContainer<Atom> centers;
        IndexVector cell_indices;
        RealVector dist_to_centers;
};

#include "point.hpp"

#endif
