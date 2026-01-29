#include <mpi.h>
#include <stdio.h>
#include <iostream>
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
Index num_centers = 1;
int rng_seed = -1;
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

void parse_cmdline(int argc, char *argv[]);

template <class Atom, class Distance>
int main_mpi(int argc, char *argv[]);

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

    mysize = mypoints.num_points();
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    mytime += MPI_Wtime();

    assert((num_centers <= num_points));

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] read input file '%s' [size=%lld]\n", time, infile, num_points);
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    IndexVector landmarks, mylandmarks;
    if (!myrank) selection_sample(num_points, num_centers, landmarks, rng_seed);
    else landmarks.resize(num_points);

    MPI_Bcast(landmarks.data(), (int)num_centers, MPI_INDEX, 0, comm);

    for (Index id : landmarks)
        if (myoffset <= id && id < myoffset+mysize)
            mylandmarks.push_back(id-myoffset);

    PointContainer<Atom> mycenters, centers;

    mycenters.indexed_gather(mypoints, mylandmarks);
    centers.allgather(mycenters, comm);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] picked centers [centers=%lld]\n", time, num_centers);
        fflush(stdout);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    VoronoiDiagram<Atom> diagram(mypoints, centers);
    diagram.compute_point_partitioning(distance);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!myrank) fprintf(stderr, "[time=%.3f] computed point partitioning\n", time);
        fflush(stdout);
    }

    //std::vector<Cell> cells;

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //diagram.coalesce_cells(cells);
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[time=%.3f] coalesced voronoi cells\n", time);
    //    fflush(stdout);
    //}

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //diagram.add_ghost_points(cells, radius, cover, leaf_size);
    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[time=%.3f] added ghost points\n", time);
    //    fflush(stdout);
    //}

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();

    //Index cellid;
    //Index numcells = cells.size();
    //MPI_Exscan(&numcells, &cellid, 1, MPI_INDEX, MPI_SUM, comm);
    //if (!myrank) cellid = 0;

    //double loctime;
    //Index inside_neighbors, ghost_neighbors;
    //std::stringstream locss;
    //char locbuf[4096+1];

    //for (Cell& cell : cells)
    //{
    //    loctime = -MPI_Wtime();
    //    cell.add_neighbors(graph, radius, inside_neighbors, ghost_neighbors);
    //    loctime += MPI_Wtime();

    //    if (verbosity >= 2)
    //    {
    //        snprintf(locbuf, 4096, "[rank=%d,time=%.3f] queried voronoi cell [cell=%lld,points=%s,ghosts=%s,inside_neighbors=%s,ghost_neighbors=%s]", myrank, loctime, cellid++, LARGE(cell.num_points()), LARGE(cell.num_ghosts()), LARGE(inside_neighbors), LARGE(ghost_neighbors));
    //        locss << locbuf << "\n";
    //    }
    //}

    //if (verbosity >= 2) std::cout << locss.str() << std::endl;

    //mytime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[time=%.3f] queried neighbors\n", time);
    //    fflush(stdout);
    //}

    //MPI_Barrier(comm);
    //mytime = -MPI_Wtime();
    //graph.redistribute();
    //mytime += MPI_Wtime();
    //mytottime += MPI_Wtime();

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[time=%.3f] redistribted edges\n", time);
    //    fflush(stdout);
    //}

    //Index edges = graph.total_edges(0);
    //Real density = (edges+0.0)/num_points;

    //if (verbosity >= 1)
    //{
    //    MPI_Reduce(&mytottime, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //    if (!myrank) printf("[time=%.3f] built near neighbor graph [vertices=%lld,edges=%lld,density=%.3f]\n", tottime, num_points, edges, density);
    //    fflush(stdout);
    //}

    //if (outfile)
    //{
    //    double t;

    //    MPI_Barrier(comm);
    //    t = -MPI_Wtime();
    //    graph.write_edge_file(outfile);
    //    t += MPI_Wtime();

    //    if (verbosity >= 1)
    //    {
    //        double tot;
    //        MPI_Reduce(&t, &tot, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    //        if (!myrank) printf("[time=%.3f] wrote edges to file '%s'\n", tot, outfile);
    //        fflush(stdout);
    //    }
    //}

    return 0;
}

void parse_cmdline(int argc, char *argv[])
{
    bool fix_num_centers = false;

    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <points> -r <radius>\n", argv[0]);
            fprintf(stderr, "Options: -m INT   number of centers [%lld]\n", num_centers);
            fprintf(stderr, "         -c FLOAT cover tree base [%.2f]\n", cover);
            fprintf(stderr, "         -l INT   leaf size [%lld]\n", leaf_size);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -D STR   distance metric [%s]\n", metric);
            fprintf(stderr, "         -o FILE  output edge file\n");
            fprintf(stderr, "         -s INT   random number seed\n");
            fprintf(stderr, "         -F       fix number of centers\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "i:r:m:c:l:v:o:s:FD:h")) >= 0)
    {
        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'm') num_centers = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'F') fix_num_centers = true;
        else if (c == 's') rng_seed = atoi(optarg);
        else if (c == 'D') metric = optarg;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!fix_num_centers) num_centers *= nprocs;

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

    if (strcmp(metric, "edit") && strcmp(metric, "l1") && strcmp(metric, "l2"))
    {
        if (!myrank) fprintf(stderr, "error: invalid metric parameter! (-D) [must be one of: edit, l1, l2]\n");
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
