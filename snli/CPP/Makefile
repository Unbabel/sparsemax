OBJS =
CC = g++
INCLUDES = -I. -I./Eigen
LIBS = -L/usr/local/lib
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 -fPIC $(INCLUDES)
LFLAGS = $(LIBS)

all: attentive_rnn attentive_rnn_test

attentive_rnn_test: $(OBJS) attentive_rnn_test.o
	$(CC) $(OBJS) attentive_rnn_test.o $(LFLAGS) -o attentive_rnn_test

attentive_rnn_test.o: attentive_rnn_test.cpp layer.h
	$(CC) $(CFLAGS) attentive_rnn_test.cpp

attentive_rnn: $(OBJS) attentive_rnn.o
	$(CC) $(OBJS) attentive_rnn.o $(LFLAGS) -o attentive_rnn

attentive_rnn.o: attentive_rnn.cpp attentive_rnn.h snli_data.h nn_utils.h utils.h layer.h
	$(CC) $(CFLAGS) attentive_rnn.cpp

clean:
	rm -f *.o *~ attentive_rnn

