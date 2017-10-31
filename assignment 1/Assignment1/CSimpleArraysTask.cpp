/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CSimpleArraysTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

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

	/////////////////////////////////////////
	// Sect. 4.6.
	
	//TO DO: load and compile kernels

	//TO DO: bind kernel arguments


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


	/////////////////////////////////////////
	// Sect. 4.6.
	
	//execute the kernel: one thread for each element!

			// Sect. 4.7.: rewrite the kernel call to use our ProfileKernel()
			//				utility function to measure execution time.
			//				Also print out the execution time.
			
	// TO DO: Determine number of thread groups and launch kernel



	// TO DO: read back results synchronously.
	//This command has to be blocking, since we need the data
}

bool CSimpleArraysTask::ValidateResults()
{
	return (memcmp(m_hC, m_hGPUResult, m_ArraySize * sizeof(float)) == 0);
}

///////////////////////////////////////////////////////////////////////////////
