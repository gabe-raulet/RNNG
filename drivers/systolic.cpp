#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <algorithm>

#include "utils.h"
#include "point.h"
#include "cover_tree.h"

MPI_Comm comm;
int myrank, nprocs;

Real radius = -1;
const char *infile = NULL;
const char *outfile = NULL;

Real cover = 1.5;
Index leaf_size = 10;
int verbosity = 1;

template <class Atom>
struct L2Distance
{
    Real operator()(const Point<Atom>& p, const Point<Atom>& q) const;
};

void parse_cmdline(int argc, char *argv[]);
int main_mpi(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    parse_cmdline(argc, argv);
    int err = main_mpi(argc, argv);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return err;
}

int main_mpi(int argc, char *argv[])
{
    using Atom = float;

    double mytime, time;
    double mytottime, tottime;

    Index num_points;
    PointContainer<Atom> points;
    L2Distance<Atom> distance;

    MPI_Barrier(comm);
    mytottime = -MPI_Wtime();
    mytime = -MPI_Wtime();

    num_points = points.read_fvecs(infile);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] read input file '%s' [size=%lld]\n", time, infile, num_points);
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    CoverTree tree(cover, leaf_size);
    tree.build(points, distance);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] built cover tree\n", time);
        fflush(stderr);
    }

    using Edge = std::tuple<Index, Index, Real>;
    using EdgeVector = std::vector<Edge>;

    EdgeVector graph;

    auto functor = [&](const Point<Atom>& p, const Point<Atom>& q, Real dist)
    {
        graph.emplace_back(q.id(), p.id(), dist);
    };

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    for (Index i = 0; i < num_points; ++i)
    {
        tree.radius_query(points, distance, points[i], radius, functor);
    }

    mytime += MPI_Wtime();
    mytottime += MPI_Wtime();

    Index num_edges = graph.size();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] found neighbors [points=%lld,edges=%lld,density=%.3f]\n", time, num_points, num_edges, (num_edges+0.0)/num_points);
        fflush(stderr);
    }

    return 0;
}

void parse_cmdline(int argc, char *argv[])
{
    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <points> -r <radius>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT cover tree base [%.2f]\n", cover);
            fprintf(stderr, "         -l INT   leaf size [%lld]\n", leaf_size);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -o FILE  output edge file\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "i:r:c:l:v:o:h")) >= 0)
    {

        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!infile)
    {
        if (!myrank) fprintf(stderr, "error: missing input file argument! (-i)\n");
        usage(1, myrank == 0);
    }

    if (radius < 0)
    {
        if (!myrank) fprintf(stderr, "error: missing radius argument! (-r)\n");
        usage(1, myrank == 0);
    }
}

template <class Atom>
Real L2Distance<Atom>::operator()(const Point<Atom>& p, const Point<Atom>& q) const
{
    Index dim = p.size();
    assert((dim == q.size()));

    Real val = 0;
    Real delta;

    for (Index i = 0; i < dim; ++i)
    {
        delta = static_cast<Real>(p[i] - q[i]);
        val += delta*delta;
    }

    return std::sqrt(val);
}
