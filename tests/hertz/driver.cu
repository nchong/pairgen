#include "hertz_wrapper.h"
#include "hertz_constants.h"
#include "framework.h"
#include "check_result_vector.h"
#include <sstream>

#define NSLOT 32

using namespace std;

void run(struct params *input, int num_iter) {
  one_time.push_back(SimpleTimer("hertz_constants"));
  one_time.back().start();
  double dt = 0.00001;
  double nktv2p = 1;
  double yeff = 3134796.2382445144467056;
  double geff = 556173.5261401557363570;
  double betaeff = -0.3578571305033167;
  double coeffFrict = 0.5;
  hertz_setup_constants(dt, nktv2p, yeff, geff, betaeff, coeffFrict);
  one_time.back().stop_and_add_to_total();

  int *numneigh = new int[input->nnode*NSLOT];
  int *neighidx = new int[input->nnode*NSLOT];
  double *shear = new double[input->nnode*NSLOT*3];
  int *touch = new int[input->nnode*NSLOT];
  for (int i=0; i<input->nnode*NSLOT; i++) {
    numneigh[i] = 0;
    neighidx[i] = 0;
    shear[(i*3)+0] = 0.0f;
    shear[(i*3)+1] = 0.0f;
    shear[(i*3)+2] = 0.0f;
    touch[i] = 0;
  }
  for (int e=0; e<input->nedge; e++) {
    int n1 = input->edge[(e*2)  ];
    int n2 = input->edge[(e*2)+1];

    assert(numneigh[n1] < NSLOT);
    int idx = (n1*NSLOT) + numneigh[n1];
    neighidx[idx] = n2;
    shear[(idx*3)+0] = input->shear[(e*3)  ];
    shear[(idx*3)+1] = input->shear[(e*3)+1];
    shear[(idx*3)+2] = input->shear[(e*3)+2];
    touch[idx] = 1;
    numneigh[n1]++;

    //insert the symmetric contact
    assert(numneigh[n2] < NSLOT);
    idx = (n2*NSLOT) + numneigh[n2];
    neighidx[idx] = n1;
    shear[(idx*3)+0] = input->shear[(e*3)  ];
    shear[(idx*3)+1] = input->shear[(e*3)+1];
    shear[(idx*3)+2] = input->shear[(e*3)+2];
    touch[idx] = 1;
    numneigh[n2]++;
  }

  one_time.push_back(SimpleTimer("hertz_init"));
  one_time.back().start();
  hertz_init(input->nnode, NSLOT, input->radius, input->mass, input->type);
  one_time.back().stop_and_add_to_total();

  hertz_run(input->nnode, NSLOT, numneigh, neighidx, input->x, input->v,
      input->omega, input->force, input->torque, shear, touch);

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

  hertz_exit();
}
