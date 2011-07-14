include ../Makefile.common

OBJ = ${HERTZ_COMMONDIR}/libcommon.so

all: libpairgen.a

lj_pair_kernel.cu:
	../../pairgen.py lj.yml
	cp reference.lj_pair_kernel.cu new.lj_pair_kernel.cu
	mv -f new.lj_pair_kernel.cu lj_pair_kernel.cu

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $^ -c -o $@

libpairgen.a: lj_pair_kernel.o lj_constants.o lj_bpa_compute_kernel.o lj_tpa_compute_kernel.o lj_wrapper.o
	ar -crusv $@ $^

clean:
	rm -f lj_* lj pair_* libpairgen.a
