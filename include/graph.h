#ifndef GRAPH_H_
#define GRAPH_H_

#include "utils.h"
#include <mpi.h>

class Graph
{
    public:

        using Edge = std::tuple<Index, Index, Real>;
        using EdgeVector = std::vector<Edge>;

        Graph(const EdgeVector& edges, Index num_vertices) : edges(edges), num_vertices(num_vertices) { std::sort(this->edges.begin(), this->edges.end()); }

        void write_file(const char *fname) const;
        void write_file(const char *fname, MPI_Comm comm) const;

        void redistribute_edges(MPI_Comm comm);

        Index num_edges() const { return edges.size(); }

    private:

        EdgeVector edges;
        Index num_vertices;
};

#include "graph.hpp"

#endif
