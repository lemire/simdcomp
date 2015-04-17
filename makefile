# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h

CFLAGS = -fPIC -std=c89 -O3 -march=native -Wall -Wextra -pedantic
CXXFLAGS = -fPIC -O3 -march=native -Wall -Wextra -pedantic
LDFLAGS = -shared
LIBNAME=libsimdcomp.so.0.0.3
all:  unit unit_chars $(LIBNAME)
test: 
	./unit
	./unit_chars
install: $(OBJECTS)
	cp $(LIBNAME) /usr/local/lib
	ln -s /usr/local/lib/$(LIBNAME) /usr/local/lib/libsimdcomp.so
	ldconfig
	cp $(HEADERS) /usr/local/include



HEADERS=./include/simdbitpacking.h ./include/simdcomputil.h ./include/simdintegratedbitpacking.h ./include/simdcomp.h 

uninstall:
	for h in $(HEADERS) ; do rm  /usr/local/$$h; done
	rm  /usr/local/lib/$(LIBNAME)
	rm /usr/local/lib/libsimdcomp.so
	ldconfig


OBJECTS= simdbitpacking.o simdintegratedbitpacking.o simdcomputil.o \
		 simdpackedsearch.o simdpackedselect.o

$(LIBNAME): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(LIBNAME) $(OBJECTS)  $(LDFLAGS) 



simdcomputil.o: ./src/simdcomputil.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdcomputil.c -Iinclude  

simdbitpacking.o: ./src/simdbitpacking.c $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdbitpacking.c -Iinclude  

simdintegratedbitpacking.o: ./src/simdintegratedbitpacking.c  $(HEADERS)
	$(CC) $(CFLAGS) -c ./src/simdintegratedbitpacking.c -Iinclude  

simdpackedsearch.o: ./src/simdpackedsearch.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c ./src/simdpackedsearch.cc -Iinclude  

simdpackedselect.o: ./src/simdpackedselect.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c ./src/simdpackedselect.cc -Iinclude -msse4

example: ./example.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o example ./example.c -Iinclude  $(OBJECTS)

unit: ./src/unit.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o unit ./src/unit.c -Iinclude  $(OBJECTS)
	
benchmark: ./src/benchmark.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o benchmark ./src/benchmark.c -Iinclude  $(OBJECTS)
dynunit: ./src/unit.c    $(HEADERS) $(LIBNAME)
	$(CC) $(CFLAGS) -o dynunit ./src/unit.c -Iinclude  -lsimdcomp 

unit_chars: ./src/unit_chars.c    $(HEADERS) $(OBJECTS)
	$(CC) $(CFLAGS) -o unit_chars ./src/unit_chars.c -Iinclude  $(OBJECTS)
clean: 
	rm -f unit *.o $(LIBNAME) example benchmark dynunit unit_chars
