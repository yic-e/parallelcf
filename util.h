#ifndef __UTIL_H__
#define __UTIL_H__
#include <cstdlib>
#include <malloc.h>
#include <stdlib.h>
const int ALIGNMENT = 32;
inline int align_32(int v){
    return (v % 32 == 0) ? v : (v + 32) & ~(0x1f);
}
inline float fGetRand(){
    return (float)rand() / RAND_MAX;
}
template<typename T>
T *align_malloc(int size){
    return (T*)memalign(ALIGNMENT, sizeof(T) * size);
}

#endif
