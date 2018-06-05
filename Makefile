
OPENCVINC=-I/home/ubuntu/armv8libs/opencv/include
OPENCVLIB=-L/home/ubuntu/armv8libs/opencv/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

lbp:
	g++ -std=c++11 -g -o lbp lbp.cpp $(OPENCVINC) $(OPENCVLIB)
clean:
	rm -f lbp
