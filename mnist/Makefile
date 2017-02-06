#
# Makefile for hw0 11-364
#

CC=gcc
CFLAGS = -Wall -std=c99 -ggdb
OBS = mnist_loader.o

all: mnist_loader

mnist_loader: $(OBS)
	$(CC) $(CFLAGS) -o mnist_loader.o -c  mnist_loader.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
