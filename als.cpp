#include "als.h"
#include "algebra.h"
#include "util.h"
#include <iostream>
#include "omp.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include "inverse.h"
using namespace std;
namespace als{


void update_optimized(embedding &emb1, embedding &emb2, FLOAT lambda){
    int beg = emb1.get_begin();
    int end = emb1.get_end();
    int tn = 0;
#pragma omp parallel for schedule(dynamic)
    for(int id = beg; id < end; ++id){

        auto &rated = emb1.get_rated(id);
        int rated_num = std::get<0>(rated).size();
        int dim = emb2.get_dim();

        int Mi_row = rated_num;
        int Mi_col = emb2.get_dim();
        int r_col = rated_num;
        
        float *emb = emb2.get_embedding();
        float *Mi = align_malloc<float>(align_32(Mi_row) * align_32(Mi_col));
        float *MiT = align_malloc<float>(align_32(Mi_col) * align_32(Mi_row));
        float *Ai = align_malloc<float>(align_32(Mi_col) * align_32(Mi_col));

        int Ai_size = Mi_col;
        int MiT_row = Mi_col;
        int MiT_col = Mi_row;
        // Get Mi
        matrix_slice(emb, Mi_col, std::get<0>(rated).data(), rated_num, Mi);
        // Get MiT
        matrix_transpose(Mi, Mi_row, Mi_col, MiT);
        // Ai = Mi * MiT
        matrix_prod_transpose(MiT, MiT_row, MiT_col, Ai);
        // Ai += I * lambda * rated_num
        matrix_add_eye(Ai, Ai_size, lambda * rated_num);

        float *Vi = align_malloc<float>(align_32(dim));

        // Vi = MiT * r_vec
        matrix_prod_vector(MiT, MiT_row, MiT_col, std::get<1>(rated).data(), Vi);

        solve_equation(Ai, emb1.get_embedding(id), Vi, Ai_size, 1e-5);
        free(Mi);
        free(MiT);
        free(Ai);
        free(Vi);
    }
}
    
void update(embedding &emb1, embedding &emb2, FLOAT lambda){
    int beg = emb1.get_begin();
    int end = emb1.get_end();
    boost::numeric::ublas::identity_matrix<FLOAT> iden(emb2.get_dim());

    for(int id = beg; id < end; ++id){
        // Get Ai and R
        auto &rated = emb1.get_rated(id);
        int rated_num = std::get<0>(rated).size();
        int dim = emb2.get_dim();
        boost::numeric::ublas::matrix<FLOAT> Mi(rated_num, emb2.get_dim());
        boost::numeric::ublas::vector<FLOAT> r_vec(rated_num);
        
        for(int i = 0; i != rated_num; ++i){
            FLOAT *emb = emb2.get_embedding(std::get<0>(rated)[i]);
            r_vec(i) = std::get<1>(rated)[i];
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
            cout<<"lambda="<<lambda<<endl;
            cout<<"rated_num="<<rated_num<<endl;
            cout<<"Mi="<<Mi<<endl;
            cout<<"Ai="<<Ai<<endl;
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

}
