
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint offset, uint stride, uint numElements) 
{
	int GID = get_global_id(0);

	if (GID < numElements)
	{
		array[GID * stride] = array[GID * stride] + array[GID * stride + offset];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride, uint numElements)
{
	int GID = get_global_id(0);
	if (GID < numElements)
	{
		array[GID] = array[GID] + array[GID + stride];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, /*uint N,*/ __local uint* localBlock)
{
	int LID = get_local_id(0);
	int grp = get_group_id(0);
	int lSize = get_local_size(0);

	if (LID < get_local_size(0))
	{
		// writing 512 global values to local memory
		localBlock[LID] = inArray[LID + grp * lSize];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int currentSize = lSize;
	bool isReduced = false;
	while(!isReduced)
	{
		//iterations on local array
		currentSize = currentSize / 2;			//scales down with iterations
		if (LID < currentSize)
		{
			localBlock[LID] = localBlock[LID] + localBlock[LID + currentSize];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (currentSize == 1) isReduced = true;
	}

	outArray[grp] = localBlock[0];

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, /*uint N,*/ __local uint* localBlock)
{
	int LID = get_local_id(0);
	int grp = get_group_id(0);
	int lSize = get_local_size(0);

	if (LID < get_local_size(0))
	{
		// writing 512 global values to local memory
		localBlock[LID] = inArray[LID + grp * lSize];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int currentSize = lSize;
	//unsigned int warpSize = WARP_SZ;
	bool isReduced = false;
	while (!isReduced)
	{
		//iterations on local array
		currentSize = currentSize / 2;			//scales down with iterations
		if (LID < currentSize)
		{
			localBlock[LID] = localBlock[LID] + localBlock[LID + currentSize];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (currentSize == 64) isReduced = true;
	}
	
	if (currentSize == 64)
	{
		for (unsigned int i = 0; i < 32; i++)
			localBlock[i] = localBlock[i] + localBlock[i + 32];
		currentSize = 32;
	}
	if (currentSize == 32)
	{
		for (unsigned int i = 0; i < 16; i++)
			localBlock[i] = localBlock[i] + localBlock[i + 16];
		currentSize = 16;
	}
	if (currentSize == 16)
	{
		for (unsigned int i = 0; i < 8; i++)
			localBlock[i] = localBlock[i] + localBlock[i + 8];
		currentSize = 8;
	}
	if (currentSize == 8)
	{
		for (unsigned int i = 0; i < 4; i++)
			localBlock[i] = localBlock[i] + localBlock[i + 4];
		currentSize = 4;
	}
	if (currentSize == 4)
	{
		localBlock[0] = localBlock[0] + localBlock[2];
		localBlock[1] = localBlock[1] + localBlock[3];
	}
	if (currentSize == 2)
		localBlock[0] = localBlock[0] + localBlock[1];


	outArray[grp] = localBlock[0];
	// TO DO: Kernel implementation
}
