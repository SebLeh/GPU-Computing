
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
__kernel void Reduction_SequentialAddressing(__global const uint* arrayA, __global uint* arrayB, uint numElements)
{
	int GID = get_global_id(0);
	if (GID < numElements)
	{
		arrayB[GID] = arrayA[GID] + arrayA[GID + numElements / 2];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
}
