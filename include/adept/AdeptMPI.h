/* mpi.h -- header file defining MPI (Message Passing Interface)
            subroutines.

    Author: Jose Arnal <jose.arnal@mail.utoronto.ca>

    This file is part of the Adept library.

*/
#ifndef AdeptMPI_H
#define AdeptMPI_H 1

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#ifdef HAVE_MPI
#include "mpi.h"
#endif//HAVE_MPI
#include <adept/Minimizer.h>


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
        
        // -------------------------------------------------------------------
        // MPI -- inline functions.
        // -------------------------------------------------------------------

        // Primary MPI Processor.
        inline bool Adept_Primary_MPI_Processor(void) {
           return(AdeptMPI::This_Processor_Number == AdeptMPI::Main_Processor_Number);
        }
        
        // Broadcast integers to all processors.
        inline void Adept_Broadcast_MPI(int *buffer, const int buffer_size) {
            #ifdef HAVE_MPI
            MPI_Bcast(buffer,
                      buffer_size,
                      MPI_INT,
                      AdeptMPI::Main_Processor_Number, 
                      MPI_COMM_WORLD );
            #endif
        }

        // Broadcast integers to all processors.
        inline void Adept_Broadcast_MPI(bool *buffer, const int buffer_size) {
            #ifdef HAVE_MPI
            MPI_Bcast(buffer,
                      buffer_size,
                      MPI_C_BOOL,
                      AdeptMPI::Main_Processor_Number, 
                      MPI_COMM_WORLD );
            #endif
        }

        // Broadcast Reals to all processors.
        inline void Adept_Broadcast_MPI(Real *buffer, const int buffer_size) {
            #ifdef HAVE_MPI
            #if ADEPT_REAL_TYPE_SIZE == 4
                MPI_Bcast(buffer,
                        buffer_size,
                        MPI_FLOAT,
                        AdeptMPI::Main_Processor_Number, 
                        MPI_COMM_WORLD );
            #elif ADEPT_REAL_TYPE_SIZE == 8
                MPI_Bcast(buffer,
                        buffer_size,
                        MPI_DOUBLE,
                        AdeptMPI::Main_Processor_Number, 
                        MPI_COMM_WORLD );
            #elif ADEPT_REAL_TYPE_SIZE == 16
                MPI_Bcast(buffer,
                        buffer_size,
                        MPI_LONG_DOUBLE ,
                        AdeptMPI::Main_Processor_Number, 
                        MPI_COMM_WORLD );
            #endif
            #endif
        }

        // Broadcast MinimizerStatus to all processors.
        inline void Adept_Broadcast_MPI(adept::MinimizerStatus* status) {
        #ifdef HAVE_MPI
        int buffer = static_cast<int>(*status);
        MPI_Bcast(&buffer, 
                  1, 
                  MPI_INT, 
                  AdeptMPI::Main_Processor_Number, 
                  MPI_COMM_WORLD );
        *status = static_cast<adept::MinimizerStatus>(buffer);
        #endif
        }

    };
};

#endif // AdeptMPI_H