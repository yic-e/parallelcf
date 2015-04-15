#ifndef __TRAITS_H__
#define __TRAITS_H__
#include "mpi.h"
template<typename T>
struct mpi_type{
    MPI_Datatype type();
};

template<>
MPI_Datatype mpi_type<double>::type(){
    return MPI_DOUBLE;
}

templat<>
MPI_Datatype mpi_type<float>::type(){
    return MPI_FLOAT;
}

#endif
