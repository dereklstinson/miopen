package miopen

/*
#include <miopen/miopen.h>
void MakeAlgoBwdData(miopenConvAlgoPerf_t *input, miopenConvBwdDataAlgorithm_t algo){
	input->bwd_data_algo=algo;
}

//perfBwdDataAlgo - Helper function to access union inside of miopenConvAlgoPerf_t
miopenConvBwdDataAlgorithm_t perfBwdDataAlgo(miopenConvAlgoPerf_t *input){
	return input->bwd_data_algo;
}
*/
import "C"
import (
	"github.com/dereklstinson/cutil"
)

//ConvBwdDataAlgoPerf binding for miopenConvAlgoPerf_t because of lack of union type in go
type ConvBwdDataAlgoPerf C.miopenConvAlgoPerf_t

//Get gets the values of ConvBwdDataAlgoPerf
func (c *ConvBwdDataAlgoPerf) Get() (algo ConvBwdDataAlgorithm, time float32, wspaceSIB uint) {
	algo = (ConvBwdDataAlgorithm)(C.perfBwdDataAlgo(c.cptr()))
	time = (float32)(c.time)
	wspaceSIB = (uint)(c.memory)
	return algo, time, wspaceSIB
}
func (c ConvBwdDataAlgoPerf) c() C.miopenConvAlgoPerf_t      { return (C.miopenConvAlgoPerf_t)(c) }
func (c *ConvBwdDataAlgoPerf) cptr() *C.miopenConvAlgoPerf_t { return (*C.miopenConvAlgoPerf_t)(c) }

//GetBwdDataWorkspaceSize -  Get the GPU memory required for the backward data convolution algorithm.
//
//For a provided tensor descriptors and algorithm selection, this function calculates and returns
//the workspace size required for back propagation on data. This call is required and must be
//executed once before running (*ConvolutionD)FindBwdDataAlgorithm() in order to determine
//the largest required allocation for the algorithm search; i.e., the maximum size of the memory
//needed from the set of potential backward convolution algorithm is returned.
//
//If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
//this.
//
//	h         MIOpen handle (input)
//	dyD         Tensor descriptor for data input tensor dy (input)
//	wD          Tensor descriptor for weight tensor w (input)
//	dxD         Tensor descriptor for output data tensor dx (input)
//
func (c *ConvolutionD) GetBwdDataWorkspaceSize(h *Handle, dyD, wD, dxD *TensorD) (wspaceSIB uint, err error) {
	var wspace C.size_t
	err = Status(C.miopenConvolutionBackwardDataGetWorkSpaceSize(h.x, dyD.d, wD.d, c.d, dxD.d, &wspace)).error("(c *ConvolutionD)GetBackwardDataWorkSpaceSize")
	wspaceSIB = (uint)(wspace)
	return wspaceSIB, err
}

//FindBwdDataAlgorithm - Search and run the backwards data convolution algorithms and return a list of kernel times.
//
//This function attempts all MIOpen backward data convolution algorithms, and returns a slice of
//type ConvBwdDataAlgoPerf.  These metrics are written in sorted fashion where the first
//element has the lowest compute time.  This function is mandatory before using backwards
//convolutions. Users can chose the top-most algorithm if they only care about the fastest algorithm.
//
//This function is mandatory before using (*ConvolutionD)BackwardData(). In order to
//execute this function, (*ConvolutionD)GetBackwardDataWorkSpaceSize() must be run to determine
//the required memory for this search.
//
// MIOpen will look for the best kernel for the provided configuration.
// If a match is not found, an exhaustive search is performed by running individual algorithms.
//
//If using Group/Depthwise convolution mode, call (*ConvolutionD)SetGroupCount() before running
//this.
//
//	h			MIOpen handle (input)
//	dyD			Tensor descriptor for data input tensor dy (input)
//	dy			Data delta tensor dy (input)
//	wD			Tensor descriptor for weight tensor w (input)
//	w			Weights tensor w (input)
//	dxD			Tensor descriptor for output data tensor dx (input)
//	dx			Data delta tensor dx (input)
//	wspace			Pointer to workspace required for the search (output)
//	wspaceSIB		Size in bytes of the memory needed for find (output)
func (c *ConvolutionD) FindBwdDataAlgorithm(
	h *Handle,
	dyD *TensorD, dy cutil.Mem,
	wD *TensorD, w cutil.Mem,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
) (results []ConvBwdDataAlgoPerf, err error) {
	request := (C.int)(4)
	var actual C.int
	results = make([]ConvBwdDataAlgoPerf, request)
	err = Status(C.miopenFindConvolutionBackwardDataAlgorithm(h.x,
		dyD.d, dy.Ptr(),
		wD.d, w.Ptr(),
		c.d,
		dxD.d, dx.Ptr(),
		request, &actual, results[0].cptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		true)).error("(c *ConvolutionD)FindBwdDataAlgorithm")
	return results[:actual], err
}
