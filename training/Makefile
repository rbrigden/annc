#
# Makefile for training
#

CFLAGS =    -Wall -std=c99  -I/usr/local/include
LDFLAGS= -L/usr/local/lib
LIBS= -lgsl -lm -ldl
OBS = training.o

all: training 

training: $(OBS)
	$(CC) $(CFLAGS) -o training.o -c  training.c

clean:
	rm -f *~ *.o *.out  *.tar *.zip *.gzip *.bzip *.gz
