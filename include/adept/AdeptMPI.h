/* mpi.h -- header file defining MPI (Message Passing Interface)
            subroutines.

    Author: Jose Arnal <jose.arnal@mail.utoronto.ca>

    This file is part of the Adept library.

*/
#ifndef AdeptMPI_H
#define AdeptMPI_H 1

#ifdef HAVE_MPI
#include "mpi.h"
#endif//HAVE_MPI

namespace adept {

    namespace internal {

        // -------------------------------------------------------------------
        // Definition of AdeptMPI class
        // -------------------------------------------------------------------
        class AdeptMPI{
        public:
            static int Number_of_Processors;   // Number of processors.
            static int This_Processor_Number;  // Processor number.
            static int Main_Processor_Number;  // Processor that will carry out calculation
            // use default constructor and destructor
       
            static void Adept_Initialize_MPI(int Main_Processor_Number);
            
        };
    };
};

#endif // AdeptMPI_H