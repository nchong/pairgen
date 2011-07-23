include ../Makefile.common

OBJ = ${HERTZ_COMMONDIR}/libcommon.so

all: hertz_tpa hertz_bpa libpairgen.a

$(OBJ):
	@if [ -z "${HERTZ_COMMONDIR}" ]; then echo "ERROR: set HERTZ_COMMONDIR env variable first."; exit 1; fi
	cd ${HERTZ_COMMONDIR}; make; cd -

hertz_pair_kernel.cu:
	../../pairgen.py hertz.yml
	cp reference.hertz_pair_kernel.cu new.hertz_pair_kernel.cu
	mv -f new.hertz_pair_kernel.cu hertz_pair_kernel.cu

hertz_tpa: driver.cu hertz_pair_kernel.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -I /usr/local/cuda/include/ $(filter-out hertz_pair_kernel.cu, $^) -D COMPUTE_TPA -o $@

hertz_bpa: driver.cu hertz_pair_kernel.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -I /usr/local/cuda/include/ $(filter-out hertz_pair_kernel.cu, $^)                -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDEDIR) -I /usr/local/cuda/include/ $^ -c -o $@

libpairgen.a: hertz_pair_kernel.o hertz_constants.o hertz_bpa_compute_kernel.o hertz_tpa_compute_kernel.o hertz_wrapper.o
	ar -crusv $@ $^

clean:
	rm -f hertz_* hertz pair_* libpairgen.a
