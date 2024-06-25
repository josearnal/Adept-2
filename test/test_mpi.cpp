/* test_mpi.cpp - Test Adept minimizer parallel computation of cost, gradient, and/or Hessian  */

#include "../config.h"

#ifdef HAVE_MPI
#include "mpi.h"
#endif//HAVE_MPI

#include <adept/AdeptMPI.h>
#include <adept_optimize.h>

using namespace adept;
using namespace internal;


// Simple quadratic cost function. The main purpose of this class is to test parallel evaluation 
// of the cost and gradient.
class QuadraticN : public Optimizable {
public:

    int N; // Number of dimensions
    int size_local; // number of dimensions to process with the current processor

    QuadraticN(int N_){

        N = N_;

        // Divide N as evenly as possible. 
        // The remainder of the integer division is spread through the 
        // first N % Number_of_Processors processors.
        #ifdef HAVE_MPI
        size_local = N / AdeptMPI::Number_of_Processors;
        if((AdeptMPI::This_Processor_Number + 1) <= N % AdeptMPI::Number_of_Processors) size_local++;

        #else
        size_local = N;
        #endif
    }

    virtual Real calc_cost_function(const adept::Vector& x){

        // Send a part of x from the main processor to this processors
        adept::Vector x_local(size_local);
        Scatter_MPI(x.data(),x_local.data(),N);
        
        // Calculate cost function contribution from this processor
        double cost = 0.0;
        for (int i = 0; i<size_local; i++){
            cost += x_local[i]*x_local[i];
        }

        // Sum up all cost function contributions
        Sum_Reduce_MPI(&cost);

        return cost;
    }
    


    virtual Real calc_cost_function_gradient(const adept::Vector& x,
					     adept::Vector gradient) {
        
        // Send a part of x from the main processor to this processors
        adept::Vector x_local(size_local);
        adept::Vector gradient_local(size_local);
        Scatter_MPI(x.data(),x_local.data(),N);

        // Calculate cost function and gradient contributions from this processor
        double cost = 0.0;
        for (int i = 0; i<size_local; i++){
            cost += x_local[i]*x_local[i];
            gradient_local[i] = 2*x_local[i];
        }

        // Sum up all cost function contributions
        Sum_Reduce_MPI(&cost);

        // Send gradient from all processors to the main processor
        Gather_MPI(gradient_local.data(),gradient.data(),N);

        return cost;
    }


    virtual bool provides_derivative(int order) {
        if (order >= 0 && order <= 1) {
            return true;
        }
        else {
            return false;
        }
    }


    // Sum reduce double to main processor.
    void Sum_Reduce_MPI(double* buffer) {
        #ifdef HAVE_MPI
        if(Adept_Primary_MPI_Processor()){
            MPI_Reduce(MPI_IN_PLACE, 
                        buffer, 
                        1,
                        MPI_DOUBLE,
                        MPI_SUM,
                        AdeptMPI::Main_Processor_Number, 
                        MPI_COMM_WORLD );
        } else{
            MPI_Reduce(buffer, 
                       buffer, 
                        1,
                        MPI_DOUBLE,
                        MPI_SUM,
                        AdeptMPI::Main_Processor_Number, 
                        MPI_COMM_WORLD );
        }
        #endif
    }

    // Sends parts of a large array on the main processor 
    // to local arrays on every processor.
    void Scatter_MPI(const double* sendbuf,
                              double* recvbuf,
                              int total_num_elements){
        #ifdef HAVE_MPI

        // construct sendcounts and discplacement array
        // to allow for sendbuf of different size
        int num_procs = AdeptMPI::Number_of_Processors;
        std::vector<int> sendcounts(num_procs, total_num_elements / num_procs);
        std::vector<int> displs(num_procs, 0);
        
        int remainder = total_num_elements % num_procs;
        for (int i = 0; i < num_procs; ++i) {
            if (i < remainder) {
                sendcounts[i]++;
            }
            if (i > 0) {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
        }

        int count = sendcounts[AdeptMPI::This_Processor_Number];

        MPI_Scatterv(sendbuf, 
                    sendcounts.data(), 
                    displs.data(), 
                    MPI_DOUBLE, 
                    recvbuf, 
                    count, 
                    MPI_DOUBLE, 
                    AdeptMPI::Main_Processor_Number, 
                    MPI_COMM_WORLD );
        #else
        for (int i = 0; i < total_num_elements; ++i) {
            recvbuf[i] = sendbuf[i];
        }
        #endif

    }
    // Sends many small arrays from each processor to a 
    // large array on the main processor.
    void Gather_MPI(const double* sendbuf,
                              double* recvbuf,
                              int total_num_elements){
        #ifdef HAVE_MPI

        // construct recvcounts and discplacement array
        // to allow for recvbufs of different size
        int num_procs = AdeptMPI::Number_of_Processors;
        std::vector<int> recvcounts(num_procs, total_num_elements / num_procs);
        std::vector<int> displs(num_procs, 0);

        int remainder = total_num_elements % num_procs;
        for (int i = 0; i < num_procs; ++i) {
            if (i < remainder) {
                recvcounts[i]++;
            }
            if (i > 0) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }

        int count = recvcounts[AdeptMPI::This_Processor_Number];

        MPI_Gatherv(sendbuf, 
                    count, 
                    MPI_DOUBLE, 
                    recvbuf, 
                    recvcounts.data(), 
                    displs.data(), 
                    MPI_DOUBLE, 
                    AdeptMPI::Main_Processor_Number, 
                    MPI_COMM_WORLD);
        #else
        for (int i = 0; i < total_num_elements; ++i) {
            recvbuf[i] = sendbuf[i];
        }
        #endif

    }

};

int main(int argc, char** argv){

    if (!adept::have_linear_algebra()) {
        std::cout << "Adept compiled without linear-algebra support: minimizer not available\n";
        return 0;
    }

    // initialize MPI, set rank 0 to be the main processor
    int rank = 0;
    #ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    AdeptMPI::Adept_Initialize_MPI(0); // main processor is # 0
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    // Initial state vector
    int N = 100; // size of state vector
    int size;

    if (rank ==0){
        size = N;
    } else{
        size = 1;
    }

    Vector x(size);
    x = 10.0;
    
    QuadraticN quadratic(N);
    Minimizer minimizer(MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS);
    MinimizerStatus status = minimizer.minimize(quadratic, x);

    if (rank == 0){
        std::cout << "Status: " << minimizer_status_string(status) << "\n";
        std::cout << "Solution: x=" << x << "\n";
        std::cout << "Number of samples: " << minimizer.n_samples() << "\n";
    }

    #ifdef HAVE_MPI
    MPI_Finalize();
    #endif

    return static_cast<int>(status);
}
