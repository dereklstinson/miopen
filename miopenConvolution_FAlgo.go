package miopen

/*
#include <miopen/miopen.h>


void MakeAlgoFwd(miopenConvAlgoPerf_t *input, miopenConvFwdAlgorithm_t algo){
	input->fwd_algo=algo;
}

//perfFwdAlgo - Helper function to access union inside of miopenConvAlgoPerf_t
miopenConvFwdAlgorithm_t perfFwdAlgo(miopenConvAlgoPerf_t *input){
	return input->fwd_algo;
}


*/
import "C"
import (
	"github.com/dereklstinson/cutil"
)

//ConvFwdAlgoPerf binding for miopenConvAlgoPerf_t because of lack of union type in go
type ConvFwdAlgoPerf C.miopenConvAlgoPerf_t

func (c ConvFwdAlgoPerf) c() C.miopenConvAlgoPerf_t      { return (C.miopenConvAlgoPerf_t)(c) }
func (c *ConvFwdAlgoPerf) cptr() *C.miopenConvAlgoPerf_t { return (*C.miopenConvAlgoPerf_t)(c) }

//Get gets the values of ConvFowdAlgoPerf
func (c *ConvFwdAlgoPerf) Get() (algo ConvFwdAlgorithm, time float32, wspaceSIB uint) {
	algo = (ConvFwdAlgorithm)(C.perfFwdAlgo(c.cptr()))
	time = (float32)(c.time)
	wspaceSIB = (uint)(c.memory)
	return algo, time, wspaceSIB
}

//GetFwdWorkspaceSize - Query the workspace size required for a forward convolution layer
//
//This call is required and must be executed once before running
//(*ConvolutionD)FindForwardAlgorithm()
//in order to determine the largest required allocation for the algorithm search; i.e., the maximum
//size of the memory needed from the set of potential forward convolution algorithm is returned.
//
//If using Group/Depthwise convolution mode, call miopenSetConvolutionGroupCount() before running
//this.
//
//	h		MIOpen handle (input)
//
//	wD		Tensor descriptor for weight tensor w (input)
//
//	xD		Tensor descriptor for input data tensor x (input)
//
//	yD		Tensor descriptor for output data tensor y (input)
func (c *ConvolutionD) GetFwdWorkspaceSize(h *Handle, wD, xD, yD *TensorD) (wspaceSIB uint, err error) {
	var sib C.size_t
	err = Status(C.miopenConvolutionForwardGetWorkSpaceSize(h.x, wD.d, xD.d, c.d, yD.d, &sib)).error("GetFwdWorkSpaceSize")
	wspaceSIB = (uint)(sib)
	return wspaceSIB, err
}

//FindForwardAlgorithm - Search and run the forward convolutional algorithms and return a list of kernel times.
//
// This function attempts all MIOpen forward convolution algorithms based on
// the input configuration, and outputs performance metrics to a
// slice of type ConvFwdAlgoPerf. These metrics are written
// in a sorted fashion where the first element has the lowest compute time.
// Users can chose the top-most algorithm if they only care about the fastest
// algorithm.
//
// This function is mandatory before using (*ConvolutionD)Forward(). In order
// to execute this function, (*ConvolutionD)GetFwdWorkSpaceSize() must be
// run to determine the required memory for this search.
//
// MIOpen will look for the best kernel for the provided configuration.
// If a match is not found, an exhaustive search is performed by running individual algorithms.
//
// If using Group/Depthwise convolution mode, call (*ConvolutionD)SetGroupCount() before running
// this.
//
//	h			MIOpen handle (input)
//
//	xD			Tensor descriptor for data input tensor x (input)
//
//	x			Data tensor x (input)
//
//	wD			Tensor descriptor for weight tensor w (input)
//
//	w			Weights tensor w (input)
//
//	yD			Tensor descriptor for output data tensor y (input)
//
//	y			Data tensor y (output)
//
//	wspace			Pointer to workspace required for the search (input)
//
//	wspaceSIB		Size in bytes of the memory needed for find (input)
func (c *ConvolutionD) FindForwardAlgorithm(
	h *Handle,
	xD *TensorD, x cutil.Mem,
	wD *TensorD, w cutil.Mem,
	yD *TensorD, y cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
) (results []ConvFwdAlgoPerf, err error) {
	request := (C.int)(4)
	var actual C.int
	results = make([]ConvFwdAlgoPerf, request)
	err = Status(C.miopenFindConvolutionForwardAlgorithm(h.x,
		xD.d, x.Ptr(),
		wD.d, w.Ptr(),
		c.d,
		yD.d, y.Ptr(),
		request, &actual, results[0].cptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		true)).error("FindForwardAlgorithm")
	return results[:actual], err
}
