MXX = mpic++
CXX = g++
CXX_FLAGS = -std=c++11 -O3  -DBOOST_UBLAS_NDEBUG=1 -march=core-avx2
ICC = /opt/intel/bin/icc
all: pals performance_test
performance_test: performance_test.cpp algebra.o
	$(CXX) $(CXX_FLAGS) performance_test.cpp algebra.o -o performance_test
pals: pals.cpp embedding.o matrix.o als.o
	$(MXX) $(CXX_FLAGS) embedding.o  als.o matrix.o pals.cpp -o pals
algebra.o: algebra.cpp algebra.h
	$(CXX) $(CXX_FLAGS) -c algebra.cpp
matrix.o: matrix.cpp matrix.h
	$(MXX) $(CXX_FLAGS) -c matrix.cpp
embedding.o: embedding.cpp embedding.h comm.h
	$(MXX) $(CXX_FLAGS) -c embedding.cpp 
als.o: als.cpp als.h
	$(MXX) $(CXX_FLAGS) -c als.cpp
.PHONY: clean
clean:
	-rm *.o *~ pals performance_test
