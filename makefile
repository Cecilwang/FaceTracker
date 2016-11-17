CC ?= gcc
CXX ?= g++

INCLUDES += -I./include
LIBPATH += -L. -L./lib

LIBS = -lopencv_objdetect -lopencv_imgproc  \
       -lopencv_imgcodecs -lopencv_core  \
       -lopencv_tracking -lopencv_datasets  -lpthread \
	   	 -lopencv_highgui -lopencv_videoio

CFLAGS +=  -O3  -pthread
CXXFLAGS +=  -O3  -pthread
CXXFLAGS += -std=c++11
LDFLAGS += -fPIC -z defs

SRCS = face.cc kcf.cc haar.cc
OBJS = $(SRCS:.cc=.o)
MAIN = face

.PHONY: clean

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ $^  $(LIBPATH) $(LIBS) $(LDFLAGS)

.cc.o:
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *~ $(MAIN)
