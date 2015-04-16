#include <iostream>
#include <fstream>
#include "embedding.h"
#include "als.h"
#include "mpi.h"
using namespace std;
using namespace als;
int main(int argc, char *argv[]){
    if(argc != 5){
        cerr<<"usage: "<< argv[0] << " K data user_num movie_num" << endl;
        return 1;
    }
    MPI_Init(&argc, &argv);
    int K = atoi(argv[1]);
    const char *data = argv[2];
    int user_num = atoi(argv[3]);
    int movie_num = atoi(argv[4]);
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
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    for(int i = 0; i != 20; ++i){
        if(user_embedding.get_rank() == 0)
            printf("iter=%d\n", i);
        update(user_embedding, movie_embedding, 0.01);
        user_embedding.sync();
        update(movie_embedding, user_embedding, 0.01);
        movie_embedding.sync();
        test(user_embedding, movie_embedding);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
