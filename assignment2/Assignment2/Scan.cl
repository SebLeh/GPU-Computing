


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	int GID = get_global_id(0);
	if (GID < offset)
		outArray[GID] = inArray[GID];
	else
		outArray[GID] = inArray[GID] + inArray[GID - offset];
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	int LID = get_local_id(0);
	int grp = get_group_id(0);
	int lSize = get_local_size(0);

	if (LID < get_local_size(0))
	{
		// writing 512 global values to local memory
		localBlock[LID] = array[LID + grp * lSize];
	}
	// TO DO: Kernel implementation
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 14)
}