#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include "cycle_timer.h"
#include "algebra.h"
#include "embedding.h"
#include "als.h"
#include "omp.h"
#include "mpi.h"
using namespace std;
using namespace als;


void test(embedding &emb1, embedding &emb2, int orig){
    int beg = emb1.get_begin();
    int end = emb1.get_end();
    int dim = emb1.get_dim();
    double res[2];
    res[0] = res[1] = 0.0;
    for(int id = beg; id != end; ++id){
        auto &rated = emb1.get_rated(id);
        int rated_num = std::get<0>(rated).size();
        res[0] += rated_num;
        for(int i = 0; i != rated_num; ++i){
            FLOAT *emb2_i = emb2.get_embedding(std::get<0>(rated)[i]);
            FLOAT *emb1_i = emb1.get_embedding(id);
            double single_rate = array_dot(emb1_i, emb2_i, dim);
            double diff = single_rate - (float)(std::get<1>(rated)[i]);
            res[1] += diff * diff;
        }
    }
    double final[2];
    MPI_Reduce(res, final, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double r = final[1] / final[0];
    r = sqrt(r);
    if(emb1.get_rank() == 0){
        printf("TestError=%f\n", r);
    }
}

int main(int argc, char *argv[]){
    if(argc != 8){
        cerr<<"usage: "<< argv[0] << "optimized K data user_num movie_num iter_num" << endl;
        return 1;
    }
    MPI_Init(&argc, &argv);
    char processor_name[128];

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int omp_t = atoi(argv[7]);
    int world_rank;
    int namelen;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Get_processor_name(processor_name,&namelen);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0){
            if(omp_t == -1){
                omp_t = omp_get_num_procs();
                printf("OMPT=%d\n", omp_t);
            }
        }
    }
    omp_set_num_threads(omp_t);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0){
            int nthreads = omp_get_num_threads();
            fprintf(stderr,"Process=%d\tProcessorName=%s\tProcsNum=%d\tThreadNum=%d\n", world_rank, processor_name, omp_get_num_procs(), nthreads);
        }
    }
    set_algebra_ompthread(8 > omp_t ? omp_t : 8);
    if(world_rank == 0){
        printf("WorldSize=%d\n", world_size);
    }
    int opti = atoi(argv[1]);
    int K = atoi(argv[2]);
    const char *data = argv[3];
    int user_num = atoi(argv[4]);
    int movie_num = atoi(argv[5]);
    int iter_num = atoi(argv[6]);
    embedding user_embedding(user_num, K);
    embedding movie_embedding(movie_num, K);
    ifstream ifs(data);

    while(ifs){
        int u, m;
        FLOAT r;
        ifs>>u>>m>>r;
        user_embedding.add_rated(u, m, r);
        movie_embedding.add_rated(m, u, r);
    }
    user_embedding.init();
    movie_embedding.init();
    user_embedding.sync();
    movie_embedding.sync();
    double total = 0.0;
    double start;
    for(int i = 0; i != iter_num; ++i){
        start = CycleTimer::currentSeconds();
        if(user_embedding.get_rank() == 0){
            printf("iter=%d\n", i);
        }
        if(opti){
            update_optimized(user_embedding, movie_embedding, 0.01);
        }
        else
            update(user_embedding, movie_embedding, 0.01);
        user_embedding.sync();
        
        if(opti)
            update_optimized(movie_embedding, user_embedding, 0.01);           
        else
            update(movie_embedding, user_embedding, 0.01);
        
        movie_embedding.sync();
        if(user_embedding.get_rank() == 0){
            double end = CycleTimer::currentSeconds();
            total += end - start;
            cout<<"TimeCost="<<(end - start)<<endl;
        }
        
        test(user_embedding, movie_embedding, opti);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    if(user_embedding.get_rank() == 0){
        cout<<"TrainingTime="<<total<<endl;
    }
    return 0;
}
