#include <cmath>
#include <cstdlib>
#include "mpi.h"
#include "embedding.h"
#include "util.h"

embedding::embedding(int num, int dim):__num(num), __dim(dim){
    __embeddings = new FLOAT(num * dim);
    MPI_Comm_rank(MPI_COMM_WORLD,&__world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &__world_size);
    __mpi_recv_num = new int[__num_processes];
    __mpi_disp = new int[__num_processes];

    int send_num = (int)ceil((double)__num / __num_processes);
    int __send_start = __rank * send_num;
    int __send_end = (__rank + 1) * send_num;
    if(__send_end >= __num){
        __send_end = __num;
    }

    __mpi_recv_num = new int[__world_size];
    __mpi_mpi_disp = new int[__world_size];
    for(int i = 0; i != __world_size; ++i){
        if(i != __world_size - 1)
            __mpi_recv_num[i] = send_num * dim;
        else
            __mpi_recv_num[i] = num * dim - i * send_num * dim;
    }

    for(int i = 0; i != __world_size; ++i){
        __mpi_disp[i] = i * send_num * dim;
    }
    
}
embedding::~embedding(){
    if(NULL != __mpi_recv_num){
        delete []__mpi_recv_num;
    }
    if(NULL != __mpi_disp){
        delete []__mpi_disp;
    }
    if(NULL != __embeddings){
        delete []__embeddigs;
    }
}
void embedding::init(){
    for(int i = 0; i != __num * __dim; ++i){
        __embeddings[i] = rand();
    }
}

void embedding::sync(){
    int count = (end - start) * __dim;
    MPI_Allgatherv(MPI_IN_PLACE,
                   (__send_end - __send_start) * __dim,
                   mpi_type<FLOAT>()::type(),
                   __embeddings,
                   __mpi_recv_num,
                   __mpi_disp,
                   mpi_type<FLOAT>()::type(),
                   MPI_COMM_WORLD);
}
