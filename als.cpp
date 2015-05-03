#include "als.h"
#include <iostream>
#include "mpi.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include "inverse.h"
using namespace std;
namespace als{
void update(embedding &emb1, embedding &emb2, FLOAT lambda){
    int beg = emb1.get_begin();
    int end = emb1.get_end();
    boost::numeric::ublas::identity_matrix<FLOAT> iden(emb2.get_dim());
    for(int id = beg; id != end; ++id){
        // Get Ai and R
        auto &rated = emb1.get_rated(id);
        int rated_num = rated.size();
        int dim = emb2.get_dim();
        boost::numeric::ublas::matrix<FLOAT> Mi(rated_num, emb2.get_dim());
        boost::numeric::ublas::vector<FLOAT> r_vec(rated_num);
        
        for(int i = 0; i != rated_num; ++i){
            FLOAT *emb = emb2.get_embedding(rated[i].id);
            r_vec(i) = rated[i].rating;
            for(int j = 0; j != dim; ++j){
                Mi(i,j) = emb[j];
            }
        }

        auto Mit = boost::numeric::ublas::trans(Mi);
        boost::numeric::ublas::matrix<FLOAT> Ai = boost::numeric::ublas::prod(Mit, Mi)
                + lambda * rated_num * iden;
        boost::numeric::ublas::matrix<FLOAT> invAi(dim, dim);
        bool invOk = invertMatrix(Ai, invAi);
        if(!invOk){
            cout<<Ai<<endl;
            cout<<Mi<<endl;
        }
        assert(invOk);
        // Get Vi
        auto Vi = boost::numeric::ublas::prod(Mit, r_vec);
        boost::numeric::ublas::vector<FLOAT> new_ui = boost::numeric::ublas::prod(invAi, boost::numeric::ublas::trans(Vi));
        // Update the value

        FLOAT *emb = emb1.get_embedding(id);
        for(int i = 0; i != dim; ++i){
            emb[i] = new_ui(i);
        }
    }
}

void test(embedding &emb1, embedding &emb2){
    int beg = emb1.get_begin();
    int end = emb1.get_end();
    int dim = emb1.get_dim();
    FLOAT res[2];
    res[0] = res[1] = 0.0;
    for(int id = beg; id != end; ++id){
        auto &rated = emb1.get_rated(id);
        int rated_num = rated.size();
        res[0] += rated_num;
        for(int i = 0; i != rated_num; ++i){
            FLOAT single_rate = 0.0;
            FLOAT *emb2_i = emb2.get_embedding(rated[i].id);
            FLOAT *emb1_i = emb1.get_embedding(id);
            for(int i = 0; i != dim; ++i){
                single_rate += emb1_i[i] * emb2_i[i];
            }
            FLOAT diff = single_rate - rated[i].rating;
            res[1] += diff * diff;
        }
    }
    FLOAT final[2];
    MPI_Reduce(res, final, 2, MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
    FLOAT r = final[1] / final[0];
    r = sqrt(r);
    if(emb1.get_rank() == 0)
        printf("TestError=%f\n", r);
}
}
