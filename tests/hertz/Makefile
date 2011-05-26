include ../Makefile.common

OBJ = ${HERTZ_COMMONDIR}/libcommon.so

all: hertz

hertz: driver.cu $(OBJ)
	@if [ -z "${HERTZ_COMMONDIR}" ]; then echo "ERROR: set HERTZ_COMMONDIR env variable first."; exit 1; fi
	../../pairgen.py hertz.yml
	cp reference.hertz_pair_kernel.cu new.hertz_pair_kernel.cu
	mv -f new.hertz_pair_kernel.cu hertz_pair_kernel.cu
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) $^ -o $@

clean:
	rm -f hertz_* hertz
