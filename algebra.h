#ifndef __ALGEBRA_H__
#define __ALGEBRA_H__
void set_algebra_ompthread(int t);
float array_diff_norm(const float *arr1, const float *arr2, int size);
float array_dot(const float *arr1, const float *arr2, int size);
void array_weighted_add(const float *arr1, float w1, const float *arr2, float w2, float *out, int size);
void matrix_transpose(float *mat1, int row, int col, float *out);
void matrix_prod_transpose(float *mat, int row, int col, float *out);
void matrix_sub_prod_vector(const float *sub, const float *mat, int row, int col, const float *vec, float *out);
void matrix_prod_vector(const float *mat, int row, int col, const float *vec, float *out);
void matrix_inverse(float *mat, int size, float *out);
void matrix_slice(float *mat, int col, const int *offset, int num, float *out);
void matrix_add_eye(float *mat, int size, float v);
void solve_equation(const float *A, float *x, const float *b, int size, float eps);

#endif
