CC=gcc
CXX=g++

svm:
	$(CC) -c ./liblinear-1.96/blas/ddot.c $(CFLAGS)
	$(CC) -c ./liblinear-1.96/blas/dnrm2.c $(CFLAGS) $(INCS)
	$(CC) -c ./liblinear-1.96/blas/dscal.c $(CFLAGS)
	$(CC) -c ./liblinear-1.96/blas/daxpy.c
	$(CXX) -c ./liblinear-1.96/tron.cpp 
	$(CXX) -c ./liblinear-1.96/linear.cpp
	$(CXX) -o $@ *.o test.cpp
clean:
	rm -f svm
	rm -f *.o
