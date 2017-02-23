#
# Makefile for hw0 11-364
#

CFLAGS =   -Wall -std=c99  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBS = mnist_loader.o

all: mnist_loader

mnist_loader: $(OBS)
	$(CC) $(CFLAGS) -o mnist_loader.o -c  mnist_loader.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
