.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h

CFLAGS = -std=c99 -O3 -Wall -Wextra -Wno-unused-parameter -pedantic

all:  unit

HEADERS=./include/simdbitpacking.h ./include/simdcomputil.h ./include/simdintegratedbitpacking.h

OBJECTS= simdbitpacking.o simdintegratedbitpacking.o simdcomputil.o

simdcomputil.o: ./src/simdcomputil.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdcomputil.c -Iinclude  

simdbitpacking.o: ./src/simdbitpacking.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdbitpacking.c -Iinclude  

simdintegratedbitpacking.o: ./src/simdintegratedbitpacking.c  $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdintegratedbitpacking.c -Iinclude  

unit: ./src/unit.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o unit ./src/unit.c -Iinclude  $(OBJECTS)

clean: 
	rm -f unit *.o
