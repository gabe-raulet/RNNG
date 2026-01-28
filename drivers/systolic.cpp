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
#include "search.h"
#include "graph.h"

MPI_Comm comm;
int myrank, nprocs;

Real radius = -1;
const char *infile = NULL;
const char *outfile = NULL;
const char *metric = "l2";

Real cover = 1.5;
Index leaf_size = 10;
int verbosity = 1;

template <class Atom>
struct L2Distance
{
    Real operator()(const Point<Atom>& p, const Point<Atom>& q) const;
};

template <class Atom>
struct EditDistance
{
    Real operator()(const Point<Atom>& p, const Point<Atom>& q) const;
};

template <class Atom, class Distance>
int main_mpi(int argc, char *argv[]);

void parse_cmdline(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    int err;
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    parse_cmdline(argc, argv);

    if (!strcmp(metric, "edit"))
        err = main_mpi<char, EditDistance<char>>(argc, argv);
    else if (!strcmp(metric, "l2"))
        err = main_mpi<float, L2Distance<float>>(argc, argv);

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return err;
}

template <class Atom, class Distance>
int main_mpi(int argc, char *argv[])
{
    double mytime, time;
    double mytottime, tottime;

    Index num_points, mysize, myoffset;
    PointContainer<Atom> mypoints;
    Distance distance;

    MPI_Barrier(comm);
    mytottime = -MPI_Wtime();
    mytime = -MPI_Wtime();

    if (!strcmp(metric, "edit"))
        num_points = mypoints.read_seqs(infile, comm);
    else if (!strcmp(metric, "l2"))
        num_points = mypoints.read_fvecs(infile, comm);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] read input file '%s' [size=%lld]\n", time, infile, num_points);
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    CoverTree search(cover, leaf_size);
    search.build(mypoints, distance);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] built cover tree\n", time);
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    using Edge = std::tuple<Index, Index, Real>;
    using EdgeVector = std::vector<Edge>;

    EdgeVector myedges;

    auto functor = [&](const Point<Atom>& p, const Point<Atom>& q, Real dist)
    {
        myedges.emplace_back(q.id(), p.id(), dist);
    };

    mysize = mypoints.num_points();

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    PointContainer<Atom> sendbuf = mypoints;
    PointContainer<Atom> recvbuf;

    using SendrecvRequest = typename PointContainer<Atom>::SendrecvRequest;

    SendrecvRequest request;

    for (int step = 0; step < nprocs; ++step)
    {
        sendbuf.sendrecv(recvbuf, recvrank, sendrank, comm, request);

        Index targsize = sendbuf.num_points();

        for (Index i = 0; i < targsize; ++i)
        {
            search.radius_query(mypoints, distance, sendbuf[i], radius, functor);
        }

        request.wait();

        sendbuf.swap(recvbuf);
    }

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] found neighbors\n", time);
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    Graph graph(myedges, num_points);
    graph.redistribute_edges(comm);

    mytime += MPI_Wtime();
    mytottime += MPI_Wtime();

    if (verbosity >= 1)
    {
        Index num_edges;
        Index my_num_edges = graph.num_edges();

        MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!myrank) fprintf(stderr, "[time=%.3f] redistributed edges [points=%lld,edges=%lld,density=%.3f]\n", time, num_points, num_edges, (num_edges+0.0)/num_points);
        fflush(stderr);
    }

    if (outfile)
    {
        MPI_Barrier(comm);
        mytime = -MPI_Wtime();
        graph.write_file(outfile, comm);
        mytime += MPI_Wtime();

        if (verbosity >= 1)
        {
            MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!myrank) fprintf(stderr, "[time=%.3f] wrote edges to file '%s'\n", time, outfile);
            fflush(stderr);
        }
    }

    MPI_Reduce(&mytottime, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!myrank) fprintf(stderr, "[time=%.3f] complete\n", tottime);
    fflush(stderr);

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
            fprintf(stderr, "         -D STR   metric [%s]\n", metric);
            fprintf(stderr, "         -o FILE  output edge file\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "i:r:c:l:v:o:D:h")) >= 0)
    {

        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'D') metric = optarg;
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

    if (strcmp(metric, "edit") && strcmp(metric, "l2"))
    {
        if (!myrank) fprintf(stderr, "error: invalid metric argument! (-D)\n");
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

template <class Atom>
Real EditDistance<Atom>::operator()(const Point<Atom>& s, const Point<Atom>& t) const
{
    Index m = s.size();
    Index n = t.size();

    IndexVector v0(n+1), v1(n+1);

    for (Index i = 0; i <= n; ++i)
        v0[i] = i;

    for (Index i = 0; i < m; ++i)
    {
        v1[0] = i+1;

        for (Index j = 0; j < n; ++j)
        {
            Index del = v0[j+1]+1;
            Index ins = v1[j+0]+1;
            Index sub = (s[i] == t[j])? v0[j] : v0[j]+1;

            v1[j+1] = std::min(del, std::min(ins, sub));
        }

        std::swap(v0, v1);
    }

    return static_cast<Real>(v0[n]);
}
