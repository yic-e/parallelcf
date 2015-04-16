#ifndef __ALS_H__
#define __ALS_H__
#include "embedding.h"

namespace als{
void update(embedding &emb1, embedding &emb2, FLOAT lambda);
void test(embedding &emb1, embedding &emb2);
}
#endif
