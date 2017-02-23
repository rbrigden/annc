#
# Makefile for hw0 11-364
#

CFLAGS = -Wall -std=c99  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBS = mnist.o

all: mnist

mnist: $(OBS)
	$(CC) $(CFLAGS) -o mnist.o -c  mnist.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
