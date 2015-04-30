CXX = /opt/mpich2/gnu/bin/mpic++
CXX_FLAGS = -std=c++11 -g -O0 -DBOOST_UBLAS_NDEBUG=1 

all: pals
pals: pals.cpp embedding.o matrix.o als.o
	$(CXX) $(CXX_FLAGS) embedding.o  als.o matrix.o pals.cpp -o pals
matrix.o: matrix.cpp matrix.h
	$(CXX) $(CXX_FLAGS) -c matrix.cpp
embedding.o: embedding.cpp embedding.h comm.h
	$(CXX) $(CXX_FLAGS) -c embedding.cpp 

als.o: als.cpp als.h
	$(CXX) $(CXX_FLAGS) -c als.cpp
.PHONY: clean
clean:
	-rm *.o *~ pals
