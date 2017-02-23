#
# Makefile for annc (11-364)
#

CC=gcc
CFLAGS =  -Wall -std=c99  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBJS=  mnist/mnist.o network/network.o training/training.o main.o

all: annc

annc: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o annc $(LIBS)

mnist/mnist.o: mnist/mnist.c
	(cd mnist; make)

network/network.o: network/network.c
	(cd network; make)

training/training.o: training/training.c
	(cd training; make)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean: clean_network clean_training clean_mnist

clean_network:
	 (cd network; $(MAKE) clean)

clean_training:
	(cd training; $(MAKE) clean)

clean_mnist:
	(cd mnist; $(MAKE) clean)
