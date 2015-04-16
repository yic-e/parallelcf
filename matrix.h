#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "comm.h"

namespace als{

class array{
  public:
    array(int c);
    array(int c, FLOAT *data);
    void create();
    void destroy();
    array& operator=(array &row);
    FLOAT& operator[](int r);
  private:
    FLOAT *__internal_data;
    int __col_num;
};

class matrix{
  public:
    matrix(int r, int c);
    matrix(int r, int c, FLOAT *data);
    array operator[](int r);
    void product(const matrix &rsh);
    void create();
    void destroy();
  private:
    FLOAT *__internal_data;
    int __row_num;
    int __col_num;
};

}
#endif
