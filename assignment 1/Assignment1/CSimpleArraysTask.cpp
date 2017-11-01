/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CSimpleArraysTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

//including signal for "debugging"(showing if error) from application
#include <csignal>
#include <signal.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CSimpleArraysTask

CSimpleArraysTask::CSimpleArraysTask(size_t ArraySize)
	: m_ArraySize(ArraySize)
{
}

CSimpleArraysTask::~CSimpleArraysTask()
{
	ReleaseResources();
}

bool CSimpleArraysTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources
	m_hA = new int[m_ArraySize];
	m_hB = new int[m_ArraySize];
	m_hC = new int[m_ArraySize];
	m_hGPUResult = new int[m_ArraySize];
	
	//fill A and B with random integers
	for(unsigned int i = 0; i < m_ArraySize; i++)
	{
		m_hA[i] = rand() % 1024;
		m_hB[i] = rand() % 1024;
	}

	//device resources

	/////////////////////////////////////////
	// Sect. 4.5

	//TO DO: allocate arrays!
	cl_int clError;
	m_dA = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create device-buffer for m_dA");
	m_dB = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create device-buffer for m_dB");
	m_dC = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create device-buffer for m_dC");


	/////////////////////////////////////////
	// Sect. 4.6.
	
	//TO DO: load and compile kernels
	size_t programSize = 0;
	string programCode;
	if (!CLUtil::LoadProgramSourceToMemory("../../Assignment1/VectorAdd.cl", programCode)) return false;
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
	if (m_Program == nullptr) return false;
	//std::raise(SIGINT);
	m_Kernel = clCreateKernel(m_Program, "VecAdd", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: VecAdd");
	
	//TO DO: bind kernel arguments
	clError = clSetKernelArg(m_Kernel, 0, sizeof(cl_mem), (void*)&m_dA);
	clError = clSetKernelArg(m_Kernel, 1, sizeof(cl_mem), (void*)&m_dB);
	clError = clSetKernelArg(m_Kernel, 2, sizeof(cl_mem), (void*)&m_dC);
	clError = clSetKernelArg(m_Kernel, 3, sizeof(cl_int), (void*)&m_ArraySize);
	V_RETURN_FALSE_CL(clError, "Failed to set Kernel args: VecAdd");


	return true;
}

void CSimpleArraysTask::ReleaseResources()
{
	//CPU resources
	SAFE_DELETE_ARRAY(m_hA);
	SAFE_DELETE_ARRAY(m_hB);
	SAFE_DELETE_ARRAY(m_hC);
	SAFE_DELETE_ARRAY(m_hGPUResult);

	/////////////////////////////////////////////////
	// Sect. 4.5., 4.6.	

	// TO DO: free resources on the GPU
	clReleaseMemObject(m_dA);
	clReleaseMemObject(m_dB);
	clReleaseMemObject(m_dC);
}

void CSimpleArraysTask::ComputeCPU()
{
	for(unsigned int i = 0; i < m_ArraySize; i++)
	{
		m_hC[i] = m_hA[i] + m_hB[m_ArraySize - i - 1];
	}
}

void CSimpleArraysTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	/////////////////////////////////////////////////
	// Sect. 4.5
	// TO DO: Write input data to the GPU
	cl_int clErr;
	clErr = clEnqueueWriteBuffer(CommandQueue, m_dA, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hA, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error copying data from host (m_hA) to device (m_dA)!");
	clErr = clEnqueueWriteBuffer(CommandQueue, m_dB, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hB, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error copying data from host (m_hB) to device (m_dB)!");


	/////////////////////////////////////////
	// Sect. 4.6.
	
	size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_ArraySize, LocalWorkSize[0]);
	size_t nGroups = globalWorkSize / LocalWorkSize[0];
	cout << "Executing " << globalWorkSize << " threads in " << nGroups << " groups of size " << LocalWorkSize[0] << endl;

	//execute the kernel: one thread for each element!
	//clErr = clEnqueueNDRangeKernel(CommandQueue, m_Kernel, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
	//V_RETURN_CL(clErr, "Error executing kernel");

	double ms;
	unsigned int iterations = 10000;

	ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, &LocalWorkSize[0], iterations);
	cout << "kernel run with " << nGroups << " groups of size " << LocalWorkSize[0] << " and " << iterations << " iterations " << " executed in " << ms << " miliseconds." << endl;
	nGroups = globalWorkSize / LocalWorkSize[1];
	ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, &LocalWorkSize[1], iterations);
	cout << "kernel run with " << nGroups << " groups of size " << LocalWorkSize[1] << " and " << iterations << " iterations " << " executed in " << ms << " miliseconds." << endl;
	nGroups = globalWorkSize / LocalWorkSize[2];
	ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, &LocalWorkSize[2], iterations);
	cout << "kernel run with " << nGroups << " groups of size " << LocalWorkSize[2] << " and " << iterations << " iterations " << " executed in " << ms << " miliseconds." << endl;
	nGroups = globalWorkSize / LocalWorkSize[3];
	ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, &LocalWorkSize[3], iterations);
	cout << "kernel run with " << nGroups << " groups of size " << LocalWorkSize[3] << " and " << iterations << " iterations " << " executed in " << ms << " miliseconds." << endl;
	nGroups = globalWorkSize / LocalWorkSize[4];
	ms = CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, &LocalWorkSize[4], iterations);
	cout << "kernel run with " << nGroups << " groups of size " << LocalWorkSize[4] << " and " << iterations << " iterations " << " executed in " << ms << " miliseconds." << endl;

	

			// Sect. 4.7.: rewrite the kernel call to use our ProfileKernel()
			//				utility function to measure execution time.
			//				Also print out the execution time.
			
	// TO DO: Determine number of thread groups and launch kernel



	// TO DO: read back results synchronously.
	//This command has to be blocking, since we need the data
	clErr = clEnqueueReadBuffer(CommandQueue, m_dC, CL_TRUE, 0, m_ArraySize * sizeof(int), m_hGPUResult, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error reading data back from device");
}

bool CSimpleArraysTask::ValidateResults()
{
	return (memcmp(m_hC, m_hGPUResult, m_ArraySize * sizeof(float)) == 0);
}

///////////////////////////////////////////////////////////////////////////////
