MXX = /opt/intel/impi/5.0.3.048/intel64/bin/mpicxx
# MXX = mpicxx
CXX = g++
CXX_FLAGS = -std=c++11 -O3 -g  -fopenmp -DBOOST_UBLAS_NDEBUG=1 -march=core-avx2 -I/home/yicheng1/.local/include 
ICC_FLAGS = -std=c++11 -O3 -g -fopenmp -DBOOST_UBLAS_NDEBUG=1  -I/home/yicheng1/.local/include
LD_FLAGS = -L/home/yicheng1/.local/lib -lopenblas
ICC = /opt/intel/bin/icpc
# ICC = mpicc
ALGEBRA = algebra
all: performance_test pals
performance_test: performance_test.cpp $(ALGEBRA).o
	$(CXX) $(CXX_FLAGS) performance_test.cpp $(LD_FLAGS) $(ALGEBRA).o -o performance_test
pals: pals.cpp embedding.o matrix.o als.o $(ALGEBRA).o
	$(MXX) $(CXX_FLAGS) embedding.o $(ALGEBRA).o als.o  pals.cpp -o pals
$(ALGEBRA).o: $(ALGEBRA).cpp algebra.h
	$(CXX) $(CXX_FLAGS) -c $(ALGEBRA).cpp
embedding.o: embedding.cpp embedding.h comm.h
	$(MXX) $(CXX_FLAGS) -c embedding.cpp 
als.o: als.cpp als.h
	$(MXX) $(CXX_FLAGS) -c als.cpp
.PHONY: clean
clean:
	-rm *.o *~ pals performance_test
