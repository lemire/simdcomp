.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h

CFLAGS = -std=c99 -O3 -Wall -Wextra -Wno-unused-parameter 

all:  unit simdbitpacking.o simdintegratedbitpacking.o

HEADERS=./include/simdbitpacking.h ./include/simdcomputil.h ./include/simdintegratedbitpacking.h

simdbitpacking.o: ./src/simdbitpacking.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdbitpacking.c -Iinclude  

simdintegratedbitpacking.o: ./src/simdintegratedbitpacking.c  $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdintegratedbitpacking.c -Iinclude  

unit: ./src/unit.c    $(HEADERS)
	$(CC) $(CFLAGS) -o unit ./src/unit.c -Iinclude  

clean: 
	rm -f unit *.o
