package miopen

/*
#include <miopen/miopen.h>

void MakeAlgoBwdWeights(miopenConvAlgoPerf_t *input, miopenConvBwdWeightsAlgorithm_t algo){
	input->bwd_weights_algo=algo;
}


//perfBwdWeightsAlgo - Helper function to access union inside of miopenConvAlgoPerf_t
miopenConvBwdWeightsAlgorithm_t perfBwdWeightsAlgo(miopenConvAlgoPerf_t *input){
	return input->bwd_weights_algo;
}

*/
import "C"
import "github.com/dereklstinson/cutil"

//ConvBwdWeightAlgoPerf binding for miopenConvAlgoPerf_t because of lack of union type in go
type ConvBwdWeightAlgoPerf C.miopenConvAlgoPerf_t

//Get gets the values of ConvBwdWeightAlgoPerf
func (c *ConvBwdWeightAlgoPerf) Get() (algo ConvBwdWeightsAlgorithm, time float32, wspaceSIB uint) {
	algo = (ConvBwdWeightsAlgorithm)(C.perfBwdWeightsAlgo(c.cptr()))
	time = (float32)(c.time)
	wspaceSIB = (uint)(c.memory)
	return algo, time, wspaceSIB
}
func (c ConvBwdWeightAlgoPerf) c() C.miopenConvAlgoPerf_t      { return (C.miopenConvAlgoPerf_t)(c) }
func (c *ConvBwdWeightAlgoPerf) cptr() *C.miopenConvAlgoPerf_t { return (*C.miopenConvAlgoPerf_t)(c) }

//GetBwdWeightsWorkspaceSize - Get the GPU memory required for the backward weights convolution algorithm.
//
//For a provided tensor descriptors and algorithm selection, this function calculates and returns
//the workspace size required for back propagation on data. This call is required and must be
//executed once before running (*ConvolutionD)FindBwdWeightsAlgorithm() in order to
//determine
//the largest required allocation for the algorithm search; i.e., the maximum size of the memory
//needed from the set of potential backward weights convolution algorithm is returned.
//
//If using Group/Depthwise convolution mode, call (*ConvolutionD)SetGroupCount() before running
//this.
//
//	h		MIOpen handle (input)
//	dyD		Tensor descriptor for data input tensor dy (input)
//	xD		Tensor descriptor for data tensor x (input)
//	dwD		Tensor descriptor for output weights tensor dw (input)
func (c *ConvolutionD) GetBwdWeightsWorkspaceSize(h *Handle, dyD, xD, dwD *TensorD) (wspaceSIB uint, err error) {
	var wsp C.size_t
	err = Status(C.miopenConvolutionBackwardWeightsGetWorkSpaceSize(h.x, dyD.d, xD.d, c.d, dwD.d, &wsp)).error("(c *ConvolutionD) GetBwdWeightsWorkSpaceSize")
	wspaceSIB = (uint)(wsp)
	return wspaceSIB, err
}

//FindBwdWeightsAlgorithm - Search and run the backwards weights convolutional algorithms and return a list of kernel times.
//
//This function attempts all MIOpen backward weights convolution algorithms, and returns a slice of
//type ConvBwdWeightAlgoPerf. These metrics are written in sorted fashion where the first element has
//the lowest compute time.  This function is mandatory before using backwards weight convolutions.
//Users can chose the top-most algorithm if they only care about the fastest algorithm.
//
//This function is mandatory before using (*ConvolutionD)BackwardWeights(). In order to
//execute this function, (*ConvolutionD)GetBwdWeightsWorkSpaceSize() must be run to
//determine the required memory for this search.
//
// MIOpen will look for the best kernel for the provided configuration.
// If a match is not found, an exhaustive search is performed by running individual algorithms.
//
// If using Group/Depthwise convolution mode, call (*Convolution)SetGroupCount() before running
// this.
//
//h		MIOpen handle (input)
//dyD		Tensor descriptor for data input tensor dy (input)
//dy		Data delta tensor dy (input)
//xD		Tensor descriptor for output data tensor x (input)
//x		Data delta tensor dx (input)
//dwD		Tensor descriptor for weight tensor dw (input)
//dw		Weights delta tensor dw (input)
//workSpace		Pointer to workspace required for the search (input)
//workSpaceSize		Size in bytes of the memory needed for find (input)
func (c *ConvolutionD) FindBwdWeightsAlgorithm(h *Handle,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	dwD *TensorD, dw cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint) (results []ConvBwdWeightAlgoPerf, err error) {
	request := (C.int)(4)
	var actual C.int
	results = make([]ConvBwdWeightAlgoPerf, request)
	err = Status(C.miopenFindConvolutionBackwardWeightsAlgorithm(h.x,
		dyD.d, dy.Ptr(),
		xD.d, x.Ptr(),
		c.d,
		dwD.d, dw.Ptr(),
		request, &actual, results[0].cptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		true)).error("FindForwardAlgorithm")
	return results[:actual], err
}
