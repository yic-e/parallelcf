#include "matrix.h"
#include <cstdlib>
#include <cstring>
namespace als{
array::array(int c):__col_num(c), __internal_data(NULL){}

array::array(int c, FLOAT *data):__col_num(c),
                                 __internal_data(data){}

void array::create(){
    __internal_data = new FLOAT[__col_num];
}

void array::destroy(){
    if(NULL != __internal_data){
        delete []__internal_data;
    }
    __internal_data = NULL;
}

array &array::operator=(array &row){
    __col_num = row.__col_num;
    memcpy(__internal_data, row.__internal_data, sizeof(FLOAT) * __col_num);
    return *this;
}


FLOAT &array::operator[](int i){
    return __internal_data[i];
}

matrix::matrix(int r, int c):__row_num(r),
                             __col_num(c),
                             __internal_data(NULL){}

matrix::matrix(int r, int c, FLOAT *data):__row_num(r),
                                          __col_num(c), 
                                          __internal_data(data) {}


array matrix::operator[](int r){
    array arr(__col_num, __internal_data + r * __col_num);
    return arr;
}

void matrix::create(){
    __internal_data = new FLOAT[__row_num * __col_num];
}

void matrix::destroy(){
    if(NULL != __internal_data){
        delete []__internal_data;
        __internal_data = NULL;
    }
}


void matrix::product(const matrix &rsh){
    
}
}
