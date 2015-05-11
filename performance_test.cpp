#include <iostream>
#include <cblas.h>
#include <boost/numeric/ublas/matrix.hpp>
#include "algebra.h"
#include "util.h"
#include "cycle_timer.h"
using namespace std;
void print_matrix(const float *mat, int row, int col){
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            float v = mat[i * align_32(col) + j];
            cout<<v<<" ";
        }
        cout<<endl;
    }
}

inline void do_naive_transpose(const float *input, float *output, int row, int col){
    const int WIDTH = 8;
    for(int i = 0; i < row; i += WIDTH){
        for(int j = 0; j < col; j += WIDTH){
            for(int k = 0; k != WIDTH; ++k){
                if(i + k > row)
                    break;
                for(int l = 0; l != WIDTH; ++l){
                    if(j+l > col)
                        break;
                    output[(j+l) * align_32(row) + (i + k)] = input[(i + k) * align_32(col) + (j + l)];
                }
            }
        }
    }
}

inline float do_blas_dot(const float *lsh, const float *rsh, int col){
    return cblas_sdot(col, lsh, 1, rsh, 1);
}

inline float naive_matrix_prod_trans(const float *mat, int row, int col, float *out){
    for(int i = 0; i < row; ++i){
        for(int j = 0; j <= i; ++j){
            float r = 0.0;
            for(int k = 0; k != col; ++k){
                r += mat[i * align_32(col) + k] * mat[j * align_32(col) + k];
            }
            out[i * align_32(row) + j] = out[j * align_32(row) + i] = r;
        }
    }
}

inline void do_blas_matrix_matrix(const float *mat, int row, int col, float *out){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, row, row, col, 1.0,
                mat, align_32(col), mat, align_32(col), 0, out, align_32(row));
}

void test_matrix_prod_trans_1024_1024(){
    //102,400x200
    // const int row = 1024;
    // const int col = 1024;

    //const int row = 128;
    //const int col = 102400;
    const int row = 1024;
    const int col = 1024;
    cout<<"row="<<row<<" col"<<col<<endl;

    //const int row = 128;
    //const int col = 102400;

    //const int row = 10240;
    //const int col = 128;

    float mat[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float blas_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float simd_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    boost::numeric::ublas::matrix<float> ubM(row, col);
    boost::numeric::ublas::matrix<float> ubRes(row, col);
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != col; ++j){
            ubM(i,j) = mat[i * align_32(col) + j] = 1;
        }
    }
    double start, end;
    double ub_cost = 1e10;

    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        ubRes = boost::numeric::ublas::prod(ubM, boost::numeric::ublas::trans(ubM));
        end = CycleTimer::currentSeconds();
        if(ub_cost > end - start){
            ub_cost = end - start;
        }
    }

    cout<<"uBlas prod_trans="<<ub_cost<<endl;

    double s_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        matrix_prod_transpose(mat, row, col, simd_res);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start)
            s_cost = end - start;
    }
    cout<<"SIMD prod_trans="<<s_cost<<endl;

    double b_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        do_blas_matrix_matrix(mat, row, col, blas_res);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start)
            b_cost = end - start;
    }
    cout<<"BLAS prod_trans="<<b_cost<<endl;
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != row; ++j){
            if(blas_res[i * align_32(row) + j] != simd_res[i * align_32(row) + j]){
                cout<<"Error mat="<<blas_res[i * align_32(row) + j]
                    <<" simd_mat="<< simd_res[i * align_32(row) + j]<< endl;
                cout<<"i="<<i<<" j="<<j<<endl;
                return;
            }
        }
    }
    cout<<"SIMD SPEEDUP="<<ub_cost / s_cost<<"x"<<endl;
    cout<<"BLAS SPEEDUP="<<ub_cost / b_cost<<"x"<<endl;
    
    
}

void test_matrix_prod_trans_128_102400(){
    //102,400x200
    const int row = 128;
    const int col = 102400;
    cout<<"row="<<row<<" col"<<col<<endl;

        //const int row = 128;
        //const int col = 102400;

        //const int row = 10240;
        //const int col = 128;

        float mat[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float blas_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float simd_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    boost::numeric::ublas::matrix<float> ubM(row, col);
    boost::numeric::ublas::matrix<float> ubRes(row, col);
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != col; ++j){
            ubM(i,j) = mat[i * align_32(col) + j] = 1;
        }
    }
    double start, end;
    double ub_cost = 1e10;

    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        ubRes = boost::numeric::ublas::prod(ubM, boost::numeric::ublas::trans(ubM));
        end = CycleTimer::currentSeconds();
        if(ub_cost > end - start){
            ub_cost = end - start;
        }
    }

    cout<<"uBlas prod_trans="<<ub_cost<<endl;

    double s_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        matrix_prod_transpose(mat, row, col, simd_res);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start)
            s_cost = end - start;
    }
    cout<<"SIMD prod_trans="<<s_cost<<endl;

    double b_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        do_blas_matrix_matrix(mat, row, col, blas_res);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start)
            b_cost = end - start;
    }
    cout<<"BLAS prod_trans="<<b_cost<<endl;
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != row; ++j){
            if(blas_res[i * align_32(row) + j] != simd_res[i * align_32(row) + j]){
                cout<<"Error mat="<<blas_res[i * align_32(row) + j]
                    <<" simd_mat="<< simd_res[i * align_32(row) + j]<< endl;
                cout<<"i="<<i<<" j="<<j<<endl;
                return;
            }
        }
    }
    cout<<"SIMD SPEEDUP="<<ub_cost / s_cost<<"x"<<endl;
    cout<<"BLAS SPEEDUP="<<ub_cost / b_cost<<"x"<<endl;
    
    
}

void test_matrix_prod_trans_10240_128(){
    //102,400x200
    const int row = 10240;
    const int col = 128;
    cout<<"row="<<row<<" col="<<col<<endl;

    float mat[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float blas_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    float simd_res[align_32(row) * align_32(row)] __attribute__ ((aligned (32)));
    boost::numeric::ublas::matrix<float> ubM(row, col);
    boost::numeric::ublas::matrix<float> ubRes(row, col);
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != col; ++j){
            ubM(i,j) = mat[i * align_32(col) + j] = 1;
        }
    }
    double start, end;
    double ub_cost = 1e10;

    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        ubRes = boost::numeric::ublas::prod(ubM, boost::numeric::ublas::trans(ubM));
        end = CycleTimer::currentSeconds();
        if(ub_cost > end - start){
            ub_cost = end - start;
        }
    }

    cout<<"uBlas prod_trans="<<ub_cost<<endl;

    double s_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        matrix_prod_transpose(mat, row, col, simd_res);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start)
            s_cost = end - start;
    }
    cout<<"SIMD prod_trans="<<s_cost<<endl;

    double b_cost = 1e10;
    for(int i = 0; i < 5; ++i){
        start = CycleTimer::currentSeconds();
        do_blas_matrix_matrix(mat, row, col, blas_res);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start)
            b_cost = end - start;
    }
    cout<<"BLAS prod_trans="<<b_cost<<endl;
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != row; ++j){
            if(blas_res[i * align_32(row) + j] != simd_res[i * align_32(row) + j]){
                cout<<"Error mat="<<blas_res[i * align_32(row) + j]
                    <<" simd_mat="<< simd_res[i * align_32(row) + j]<< endl;
                cout<<"i="<<i<<" j="<<j<<endl;
                return;
            }
        }
    }
    cout<<"SIMD SPEEDUP="<<ub_cost / s_cost<<"x"<<endl;
    cout<<"BLAS SPEEDUP="<<ub_cost / b_cost<<"x"<<endl;
    
    
}

void test_array_dot(){
    const int col = 10000;
    float m1[col] __attribute__ ((aligned (64)));
    float m2[col] __attribute__ ((aligned (64)));
    float res1, res2, res3;
    for(int i = 0; i < col; ++i){
        m1[i] = (float)rand() / RAND_MAX - 0.5;
        m2[i] = (float)rand() / RAND_MAX - 0.5;
    }
    double start, end;
    double s_cost = 1e10;
    for(int i = 0; i < 10; ++i){
        start = CycleTimer::currentSeconds();
        res2 = array_dot(m1, m2, col);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start)
            s_cost = end - start;
    }
    cout<<"SIMD dot cost= "<< s_cost<<endl;

    double n_cost = 1e10;
    for(int i = 0; i < 10; ++i){
        start = CycleTimer::currentSeconds();
        res1 = 0;
        for(int j = 0; j != col; ++j){
            res1 += m1[j] * m2[j];
        }
        end = CycleTimer::currentSeconds();
        if(n_cost > end - start)
            n_cost = end - start;
    }

    cout<<"Naive dot cost="<<n_cost<<endl;
    
    double b_cost = 1e10;
    for(int i = 0; i < 10; ++i){
        start = CycleTimer::currentSeconds();
        res3 = do_blas_dot(m1, m2, col);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start)
            b_cost = end - start;
    }
    cout<<"Blas dot cost="<<b_cost<<endl;
   
    cout<<"RES="<<res1<<" "<<res2<<" "<<res3<<endl;
    cout<<"SPEEDUP="<<n_cost / s_cost<<"x"<<endl;
    cout<<"SPEEDUP_BLAS="<<n_cost / b_cost<<"x"<<endl;
}

void test_simd_transpose(){
    const int row = 1000 * 100;
    const int col = 100;
    using namespace boost::numeric::ublas;
    matrix<float> bmat(row, col);
    
    float orig[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float res[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float simd_res[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != col; ++j){
            orig[align_32(col) * i + j] = i * j;
            bmat(i, j) = i;
        }
    }
    
    double n_cost = 1e10;
    double start, end;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        do_naive_transpose(orig, res, row, col);
        end = CycleTimer::currentSeconds();
        if(n_cost > end - start)
            n_cost = end - start;
    }
    cout<<"Naive tranpose cost="<<n_cost<<endl;

    double s_cost = 1e10;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        matrix_transpose(orig, row, col, simd_res);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start)
            s_cost = end - start;
    }
    cout<<"SIMD transpose cost= "<< s_cost<<endl;
    matrix<float> tmat(col, row);
    double b_cost = 1e10;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        tmat = trans(bmat);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start)
            b_cost = end - start;
        
    }
    cout<<"Boost transpose cost="<<b_cost<<endl;
    for(int i = 0; i != col; ++i){
        for(int j = 0; j != row; ++j){
            if(res[i * align_32(col) + j] != simd_res[i * align_32(col) + j]){
                printf("Error in simd_transpose test\n");
                break;
            }
        }
    }
    cout<<"SPEEDUP="<<b_cost / s_cost<<"x"<<endl;
}

inline void do_blas_matrix_prod(const float *mat, int row, int col, const float *vec, float *out){
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                row, col, 1.0,
                mat, align_32(col),
                vec, 1,
                0.0, out, 1);
}

void test_matrix_prod_vector(){
    const int row = 100000;
    const int col = 1000;
    float A[align_32(row) * align_32(col)] __attribute__ ((aligned (32)));
    float vec[align_32(col)] __attribute__ ((aligned (32)));
    float simd_res[align_32(row)] __attribute__ ((aligned (32)));
    float blas_res[align_32(row)] __attribute__ ((aligned (32)));
    boost::numeric::ublas::vector<float> ubvec(row);
    boost::numeric::ublas::vector<float> ubres(row);
    boost::numeric::ublas::matrix<float> ubA(row, col);
    for(int i = 0; i != row; ++i){
        for(int j = 0; j != col; ++j){
            ubA(i, j) = A[align_32(col) * i + j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    for(int i = 0; i != col; ++i){
        ubres(i) = vec[i] = (float)rand() / RAND_MAX - 0.5;
    }
    double start, end;

    double ub_cost = 1e10;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        ubres = boost::numeric::ublas::prod(ubA, ubvec);
        end = CycleTimer::currentSeconds();
        if(ub_cost > end - start){
            ub_cost = end - start;
        }
    }
    cout<<"uBlas matrix_prod_vector="<<ub_cost<<endl;
    
    double b_cost = 1e10;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        do_blas_matrix_prod(A, row, col, vec, blas_res);
        end = CycleTimer::currentSeconds();
        if(b_cost > end - start){
            b_cost = end - start;
        }
    }
    cout<<"Blas matrix_prod_vector="<<b_cost<<endl;

    double s_cost = 1e10;
    for(int i = 0; i < 3; ++i){
        start = CycleTimer::currentSeconds();
        matrix_prod_vector(A, row, col, vec, simd_res);
        end = CycleTimer::currentSeconds();
        if(s_cost > end - start){
            s_cost = end - start;
        }
    }
    cout<<"SIMD matrix_prod_vector="<<s_cost<<endl;
    cout<<"SIMD_SPEEDUP="<<ub_cost / s_cost<<endl;
    cout<<"BLAS_SPEEDUP="<<ub_cost / b_cost<<endl;
    for(int i = 0; i != row; ++i){
        if(abs(simd_res[i] - blas_res[i]) > 1e-5){
            cout<<"matrix_prod_vector failed "<<i<< " SIMD="<<simd_res[i]<<" BLAS="<<blas_res[i]<<endl;
            return;
        }
    }
}

int main(){
    test_simd_transpose();
    cout<<"----------------"<<endl;
    test_array_dot();
    cout<<"----------------"<<endl;
    test_matrix_prod_vector();
    cout<<"----------------"<<endl;
    //test_matrix_prod_trans();
    //test_matrix_prod_trans_128_102400();
    //test_matrix_prod_trans_1024_1024();
    //test_matrix_prod_trans_10240_128();
    return 0;
}
