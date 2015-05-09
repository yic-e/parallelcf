#include <cmath>
#include <cstdlib>
#include <tuple>
#include "mpi.h"
#include "embedding.h"
#include "util.h"
namespace als{
void embedding::debug(){
    printf("data-beg=%d data-end=%d world-rank=%d/%d\n", __data_begin, __data_end, __world_rank, __world_size);
}
embedding::embedding(int num, int dim):__num(num), __dim(dim){
    __embeddings = align_malloc<float>(align_32(dim) * num);
    MPI_Comm_rank(MPI_COMM_WORLD,&__world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &__world_size);

    // The number of nodes to be sent
    int send_num = (int)ceil((double)__num / __world_size);

    __data_begin = __world_rank * send_num;
    __data_end = (__world_rank + 1) * send_num;
    if(__data_end >= num){
        __data_end = num;
    }

    __mpi_recv_num = new int[__world_size];
    for(int i = 0; i != __world_size; ++i){
        if(i != __world_size - 1)
            __mpi_recv_num[i] = send_num * align_32(dim);
        else
            __mpi_recv_num[i] = (num - i * send_num) * align_32(dim);
    }

    __mpi_disp = new int[__world_size];
    for(int i = 0; i != __world_size; ++i){
        __mpi_disp[i] = i * send_num * align_32(dim);
    }
    __rated_data.resize(__data_end - __data_begin);
}

embedding::~embedding(){
    if(NULL != __mpi_recv_num){
        delete []__mpi_recv_num;
    }
    if(NULL != __mpi_disp){
        delete []__mpi_disp;
    }
    if(NULL != __embeddings){
        //delete []__embeddings;
        free(__embeddings);
    }
}
int embedding::get_rank(){
    return __world_rank;
}
int embedding::get_dim(){
    return __dim;
}


int embedding::get_begin(){
    return __data_begin;
}

int embedding::get_end(){
    return __data_end;
}

FLOAT *embedding::get_embedding(int id){
    return __embeddings + (id * align_32(__dim));
}

FLOAT *embedding::get_embedding(){
    return __embeddings;
}

void embedding::add_rated(int a, int b, FLOAT r){
    if(a < __data_begin || a >= __data_end)
        return;
    
    std::get<0>(__rated_data[a - __data_begin]).push_back(b);
    std::get<1>(__rated_data[a - __data_begin]).push_back(r);
}

const std::tuple<std::vector<int>, std::vector<float, ::boost::alignment::aligned_allocator<float, 32> > > &embedding::get_rated(int id){
    return __rated_data[id - __data_begin];
}

void embedding::init(){
    for(int i = 0; i != __num; ++i){
        for(int j = 0; j != __dim; ++j)
            __embeddings[i * align_32(__dim) + j] = fGetRand();
    }
}

void embedding::sync(){
    MPI_Allgatherv(MPI_IN_PLACE,
                   (__data_end - __data_begin) * align_32(__dim),
                   MPI_FLOAT_TYPE,
                   __embeddings,
                   __mpi_recv_num,
                   __mpi_disp,
                   MPI_FLOAT_TYPE,
                   MPI_COMM_WORLD);
}
}
