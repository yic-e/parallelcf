MXX = /opt/intel/impi/5.0.3.048/intel64/bin/mpicxx
# MXX = mpicxx
CXX = g++
CXX_FLAGS = -std=c++11 -O3 -fprefetch-loop-arrays -g -fopenmp -DBOOST_UBLAS_NDEBUG=1 -march=core-avx2 -I/home/yicheng1/.local/include 
LD_FLAGS = -L/home/yicheng1/.local/lib -lopenblas
ICC = /opt/intel/bin/icc
# ICC = mpicc
all: pals performance_test 
performance_test: performance_test.cpp algebra.o
	$(CXX) $(CXX_FLAGS) performance_test.cpp $(LD_FLAGS) algebra.o -o performance_test
pals: pals.cpp embedding.o matrix.o als.o algebra.o
	$(MXX) $(CXX_FLAGS) embedding.o algebra.o als.o  pals.cpp -o pals
algebra.o: algebra.cpp algebra.h
	$(CXX) $(CXX_FLAGS) -c algebra.cpp
embedding.o: embedding.cpp embedding.h comm.h
	$(MXX) $(CXX_FLAGS) -c embedding.cpp 
als.o: als.cpp als.h
	$(CXX) $(CXX_FLAGS) -c als.cpp
.PHONY: clean
clean:
	-rm *.o *~ pals performance_test
