#
# Makefile for hw0 11-364
#

CC=gcc
CFLAGS = -Wall -std=c99 -ggdb  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl
OBJS1 = network.o main.o
OBJS2=  network.o tests.o

all: dnn tests

dnn: $(OBJS1)
	$(CC) $(LDFLAGS) $(OBJS1) -o dnn $(LIBS)

tests: $(OBJS2)
	$(CC) $(LDFLAGS) $(OBJS2)  -o tests $(LIBS)


network.o: network.c
	$(CC) $(CFLAGS) -c network.c

tests.o: tests.c
	$(CC) $(CFLAGS) -c tests.c

main.o: main.c
	$(CC) $(CFLAGS) -c main.c





clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
