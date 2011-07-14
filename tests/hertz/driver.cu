#include "hertz_constants.cu"
#include "hertz_wrapper.cu"
#include "framework.h"
#include "check_result_vector.h"
#include "cuPrintf.cu"
#include <sstream>

#define NSLOT 32

using namespace std;

void run(struct params *input, int num_iter) {
  // TODO: we only send scalar values as constants to the device.
  // We take advantage of the fact that all particles are the same type
  // So we only need one value from the yeff, geff, betaeff and coeffFrict arrays
  double dt = input->dt;
  double nktv2p = input->nktv2p;
  double yeff = input->yeff[3];
  double geff = input->geff[3];
  double betaeff = input->betaeff[3];
  double coeffFrict = input->coeffFrict[3];
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

  //internal copies of data mutated by kernel
  int *numneigh_copy = new int[input->nnode];
  int *neighidx_copy = new int[input->nnode*NSLOT];
  double *force_copy = new double[input->nnode*3];
  double *torque_copy = new double[input->nnode*3];
  double *shear_copy = new double[input->nnode*NSLOT*3];
  int *touch_copy = new int[input->nnode*NSLOT];

  per_iter.push_back(SimpleTimer("hertz_run"));
  per_iter.push_back(SimpleTimer("convert_neighbor"));
  for (int run=0; run<num_iter; run++) {

    //make copies
    copy(numneigh, numneigh + input->nnode, numneigh_copy);
    copy(neighidx, neighidx + input->nnode*3, neighidx_copy);
    copy(input->force, input->force + input->nnode*3, force_copy);
    copy(input->torque, input->torque + input->nnode*3, torque_copy);
    copy(shear, shear + input->nnode*NSLOT*3, shear_copy);
    copy(touch, touch + input->nnode*NSLOT, touch_copy);

    per_iter[0].start();
    hertz_run(input->nnode, NSLOT, numneigh_copy, neighidx_copy,
        input->x, input->v, input->omega,
        force_copy, torque_copy, shear_copy, touch_copy);
    per_iter[0].stop_and_add_to_total();

    per_iter[1].start();
    for (int ii=0; ii<nl->inum; ii++) {
      int i = nl->ilist[ii];
      for (int jj=0; jj<nl->numneigh[i]; jj++) {
        int j = nl->firstneigh[i][jj];
        if (i < j) {
          int idx = (i*NSLOT) + jj;
          nl->firstdouble[i][(jj*3)  ] = shear_copy[(idx*3)  ];
          nl->firstdouble[i][(jj*3)+1] = shear_copy[(idx*3)+1];
          nl->firstdouble[i][(jj*3)+2] = shear_copy[(idx*3)+2];
          nl->firsttouch[i][jj] = touch_copy[idx];
        }
      }
    }
    per_iter[1].stop_and_add_to_total();

#if 1
    if (run == 0) {
      const double epsilon = 0.00001;
      bool verbose = false;
      bool die_on_flag = false;

      for (int n=0; n<input->nnode; n++) {
        stringstream out;
        out << "force[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_force[(n*3)], &force_copy[(n*3)],
            epsilon, verbose, die_on_flag);
      }

      for (int n=0; n<input->nnode; n++) {
        stringstream out;
        out << "torque[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_torque[(n*3)], &torque_copy[(n*3)],
            epsilon, verbose, die_on_flag);
      }

      int ptr = 0;
      double *shear_check = new double[input->nedge*3];
      for (int ii=0; ii<nl->inum; ii++) {
        int i = nl->ilist[ii];
        for (int jj=0; jj<nl->numneigh[i]; jj++) {
          int idx = (i*NSLOT)+jj;
          shear_check[(ptr*3)  ] = shear_copy[(idx*3)  ];
          shear_check[(ptr*3)+1] = shear_copy[(idx*3)+1];
          shear_check[(ptr*3)+2] = shear_copy[(idx*3)+2];
          ptr++;
        }
      }
      verbose = true;
      for (int n=0; n<input->nedge; n++) {
        stringstream out;
        out << "shear[" << n << "]";
        check_result_vector(
            out.str().c_str(),
            &input->expected_shear[(n*3)], &shear_check[(n*3)],
            epsilon, verbose, die_on_flag);
      }
      delete[] shear_check;
    }
#endif
  }

  hertz_exit();
  delete[] numneigh;
  delete[] neighidx;
  delete[] shear;
  delete[] touch;

  delete[] numneigh_copy;
  delete[] neighidx_copy;
  delete[] force_copy;
  delete[] torque_copy;
  delete[] shear_copy;
  delete[] touch_copy;
}
