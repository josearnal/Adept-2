/* mpi.h -- variables for MPI (Message Passing Interface) class

    Author: Jose Arnal <jose.arnal@mail.utoronto.ca>

    This file is part of the Adept library.

*/
#include <adept/AdeptMPI.h>
#include <adept/exception.h>

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#ifdef HAVE_MPI
#include "mpi.h"
#endif//HAVE_MPI

namespace adept {

  using namespace internal;

    int AdeptMPI::Number_of_Processors = 0;
    int AdeptMPI::This_Processor_Number = 0;
    int AdeptMPI::Main_Processor_Number = 0;

    // Initialize MPI.
    void AdeptMPI::Adept_Initialize_MPI(int Main_Processor_Number) {
    #ifdef HAVE_MPI
        int size, rank ;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        AdeptMPI::Number_of_Processors  = size;
        AdeptMPI::This_Processor_Number = rank;
        AdeptMPI::Main_Processor_Number = Main_Processor_Number;
    #else
        throw MPI_exception("Adept was compiled without MPI"
                                   ADEPT_EXCEPTION_LOCATION);
    #endif //HAVE_MPI
    }
}