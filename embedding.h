#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__
#include <vector>
#include "comm.h"
namespace als{
struct rated{
    rated(int i, FLOAT r):id(i), rating(r){}
    int id;
    FLOAT rating;
};
class embedding {
  public:
    embedding(int num, int dim);
    int get_dim();
    int get_rank();
    void init();
    void sync();
    
    ~embedding();
    FLOAT *get_embedding(int id);
    void add_rated(int a, int b, FLOAT r);
    int get_begin();
    int get_end();
    const std::vector<rated> &get_rated(int id);
    void debug();
  private:
    int __num;
    int __dim;
    FLOAT *__embeddings;
    int __world_size;
    int __world_rank;
    std::vector<std::vector<rated> > __rated_data;
    int __data_begin;
    int __data_end;
    
  private:
    // MPI data
    int *__mpi_recv_num;
    int *__mpi_disp;
    
};
}
#endif
