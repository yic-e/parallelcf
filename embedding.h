#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__
#include <vector>
#include "comm.h"
class embedding {
  public:
    embedding(int num, int dim);
    void init();
    void sync();
    ~embedding();
  private:
    int __num;
    int __dim;
    FLOAT *__embeddings;
    int __world_size;
    int __world_rank;

  private:
    // MPI data
    int *__mpi_recv_num;
    int *__mpi_disp;
    int __send_start;
    int __send_end;
    
};

#endif
