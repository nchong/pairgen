#include "hertz_wrapper.cu"
#include "framework.h"
#include "check_result_vector.h"
#include "cuPrintf.cu"
#include <sstream>

#define NSLOT 32

using namespace std;

void run(struct params *input, int num_iter) {
  double dt = 0.00001;
  double nktv2p = 1;
  double yeff = 3134796.2382445144467056;
  double geff = 556173.5261401557363570;
  double betaeff = -0.3578571305033167;
  double coeffFrict = 0.5;
  one_time.push_back(SimpleTimer("hertz_constants"));
  one_time.back().start();
  hertz_setup_constants(dt, nktv2p, yeff, geff, betaeff, coeffFrict);
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("init_gpu_neighlist"));
  one_time.back().start();
  int *numneigh = new int[input->nnode];
  int *neighidx = new int[input->nnode*NSLOT];
  double *shear = new double[input->nnode*NSLOT*3];
  int *touch = new int[input->nnode*NSLOT];
  fill_n(neighidx, input->nnode*NSLOT, 0);
  fill_n(shear,    input->nnode*NSLOT*3, 0.0);
  fill_n(touch,    input->nnode*NSLOT, 0);
  one_time.back().stop_and_add_to_total();

  // NB: Not a one_time cost.
  // This cost is incurred everytime the neighbor list changes.
  NeighListLike *nl = new NeighListLike(input);
  one_time.push_back(SimpleTimer("rebuild_gpu_neighlist"));
  one_time.back().start();
  // reset numneigh count
  // could also reset neighidx (for debug reasons?) but not necessary
  fill_n(numneigh, input->nnode, 0);
  for (int ii=0; ii<nl->inum; ii++) {
    int i = nl->ilist[ii];
    for (int jj=0; jj<nl->numneigh[i]; jj++) {
      int j = nl->firstneigh[i][jj];
      assert(numneigh[i] < NSLOT);
      int idx = (i*NSLOT) + numneigh[i];
      neighidx[idx] = j;
      shear[(idx*3)+0] = nl->firstdouble[i][(jj*3)  ];
      shear[(idx*3)+1] = nl->firstdouble[i][(jj*3)+1];
      shear[(idx*3)+2] = nl->firstdouble[i][(jj*3)+2];
      touch[idx] = 1;
      numneigh[i]++;

      //insert the symmetric contact
      assert(numneigh[j] < NSLOT);
      idx = (j*NSLOT) + numneigh[j];
      neighidx[idx] = i;
      shear[(idx*3)+0] = nl->firstdouble[i][(jj*3)  ];
      shear[(idx*3)+1] = nl->firstdouble[i][(jj*3)+1];
      shear[(idx*3)+2] = nl->firstdouble[i][(jj*3)+2];
      touch[idx] = 1;
      numneigh[j]++;
    }
  }
  one_time.back().stop_and_add_to_total();

  one_time.push_back(SimpleTimer("hertz_init"));
  one_time.back().start();
  hertz_init(input->nnode, NSLOT, input->radius, input->mass, input->type);
  hertz_update_neigh(input->nnode, NSLOT, numneigh, neighidx);
  one_time.back().stop_and_add_to_total();

  per_iter.push_back(SimpleTimer("hertz_run"));
  for (int run=0; run<num_iter; run++) {

    per_iter[0].start();
    hertz_run(input->nnode, NSLOT, numneigh, neighidx, input->x, input->v,
        input->omega, input->force, input->torque, shear, touch);
    per_iter[0].stop_and_add_to_total();

#if 1
    if (run == 0) {
      for (int n=0; n<input->nnode; n++) {
        const double epsilon = 0.00001;
        bool verbose = false;
        bool die_on_flag = false;

        stringstream out;
        out << "force[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_force[(n*3)], &input->force[(n*3)], 
            epsilon, verbose, die_on_flag);
        out.str("");

        out << "torque[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_torque[(n*3)], &input->torque[(n*3)],
            epsilon, verbose, die_on_flag);
      }
    }
#endif
  }

  hertz_exit();
}
