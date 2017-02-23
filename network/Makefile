#
# Makefile for network
#

CFLAGS =  -Wall -std=c99  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBS = network.o

all: network

network: $(OBS)
	$(CC) $(CFLAGS) -o network.o -c network.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
