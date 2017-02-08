#
# Makefile for hw0 11-364
#

CFLAGS = -Wall -std=c99 -ggdb -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBS = mnist_network.o

all: mnist_network

mnist_network: $(OBS)
	$(CC) $(CFLAGS) -o mnist_network.o -c  mnist_network.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
