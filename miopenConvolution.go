package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/cutil"
)

//ConvolutionD - Convolution descriptor is an object that allows the user to specify a layer's padding, stride,
//and dilation of the convolutional filter. Parameters must all be non-negative.
type ConvolutionD struct {
	dims C.int
	d    C.miopenConvolutionDescriptor_t
}

//CreateConvolutionDescriptor -  Creates a convolution layer descriptor
func CreateConvolutionDescriptor() (*ConvolutionD, error) {
	x := new(ConvolutionD)

	err := Status(C.miopenCreateConvolutionDescriptor(&x.d)).error("CreateConvolutionDescriptor")
	if err != nil {
		return nil, err
	}

	runtime.SetFinalizer(x, miopenDestroyConvolutionDescriptor)

	return x, nil
}

func miopenDestroyConvolutionDescriptor(c *ConvolutionD) error {
	return Status(C.miopenDestroyConvolutionDescriptor(c.d)).error("miopenDestroyConvolutionDescriptor")

}

//Set sets the N-dimensional convolution layer descriptor
//
// pad           Array of input data padding (input)
//
// stride        Array of convolution stride (input)
//
// dilation      Array of convolution dilation (input)
//
// c_mode        Convolutional mode (input)
//
// len(pad) ==len(stride) ==len(dilation)
func (c *ConvolutionD) Set(pad, stride, dilation []int32, mode ConvolutionMode) error {

	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdilation := int32Tocint(dilation)
	c.dims = C.int(len(pad))

	return Status(C.miopenInitConvolutionNdDescriptor(c.d, c.dims, &cpad[0], &cstride[0], &cdilation[0], mode.c())).error(" (*ConvolutionD)Set()")
}

//Get - Retrieves a N-dimensional convolution layer descriptor's details
//
//
func (c *ConvolutionD) Get() (pad, stride, dilation []int32, mode ConvolutionMode, err error) {
	padding := make([]C.int, c.dims)
	striding := make([]C.int, c.dims)
	dilationing := make([]C.int, c.dims)
	var actual C.int
	err = Status(C.miopenGetConvolutionNdDescriptor(c.d, c.dims, &actual, &padding[0], &striding[0], &dilationing[0], mode.cptr())).error("(*ConvolutionD)Get()")
	return cintToint32(padding[:actual]), cintToint32(striding[:actual]), cintToint32(dilationing[:actual]), mode, err
}

//SetGroupCount -- Set the number of groups to be used in Group/Depthwise convolution
//
//Must be called before all computational APIs of group/depthwise convolution, it is preferable to
//call miopenInitConvolutionDescriptor() first, then miopenSetConvolutionGroupCount() to fully
//initialize group convolutions. Both Convolution Mode and Transpose Convolution Mode support
//group/depthwise convolution. To run depthwise convolution, set groupCount value equal to number of
//channels.
//
//	groupCount		number of groups, in depthwise conv using filter_number/channel_multiplier
func (c *ConvolutionD) SetGroupCount(groupCount int32) error {
	return Status(C.miopenSetConvolutionGroupCount(c.d, (C.int)(groupCount))).error("SetGroupCount")
}

//SetTransposeOutputPadding - Set the output padding to be used in N-dimensional Transpose convolution
//
// This function is optional for initialization of Transpose convolution. If applicable, it must be
// called before all computational APIs of Transpose convolution. It is preferable to call
// miopenInitConvolutionNdDescriptor() first, then miopenSetTransposeConvNdOutputPadding() to fully
// initialize transpose convolutions. Currently, 2-D and 3-D convolutions are supported.
//
//	adjA		array of output padding for output data (input)
func (c *ConvolutionD) SetTransposeOutputPadding(adjA []int32) error {
	dims := (C.int)(len(adjA))
	cadjA := int32Tocint(adjA)
	return Status(C.miopenSetTransposeConvNdOutputPadding(c.d, dims, &cadjA[0])).error("SetTransposeOutputPadding")
}

//ForwardOutputDim - Get the shape of a resulting N-dimensional tensor from a (N-2)-dimensional convolution
//
//This function returns the dimensions of the resulting N-dimensional tensor of a (N-2)-dimensional
//convolution, given the convolution descriptor, the input tensor descriptor
//and the filter descriptor. It is used to setup the output tensor descriptor prior to executing
//the convolution layer.
//
//	xD		Input data tensor descriptor (input)
//
//	wD		Weight descriptor (input)
//
func (c *ConvolutionD) ForwardOutputDim(xD, wD *TensorD) (outputdims []int32, err error) {
	var dims C.int
	odims := make([]C.int, c.dims)
	err = Status(C.miopenGetConvolutionNdForwardOutputDim(c.d, xD.d, wD.d, &dims, &odims[0])).error("SetTransposeOutputPadding")
	outputdims = cintToint32(odims[:dims])
	return outputdims, err
}

//ConvFwdAlgorithm - Used as flags.
//Convolutional algorithm mode for forward propagation. MIOpen use cross-correlation for its
//convolution implementation.
//
type ConvFwdAlgorithm C.miopenConvFwdAlgorithm_t

func (c ConvFwdAlgorithm) c() C.miopenConvFwdAlgorithm_t      { return (C.miopenConvFwdAlgorithm_t)(c) }
func (c *ConvFwdAlgorithm) cptr() *C.miopenConvFwdAlgorithm_t { return (*C.miopenConvFwdAlgorithm_t)(c) }

//GEMM sets c and returns GEMM flag
func (c *ConvFwdAlgorithm) GEMM() ConvFwdAlgorithm {
	*c = (ConvFwdAlgorithm)(C.miopenConvolutionFwdAlgoGEMM)
	return *c
}

//Direct sets c and returns Direct flag
func (c *ConvFwdAlgorithm) Direct() ConvFwdAlgorithm {
	*c = (ConvFwdAlgorithm)(C.miopenConvolutionFwdAlgoDirect)
	return *c
}

//FFT sets c and returns FFT flag
func (c *ConvFwdAlgorithm) FFT() ConvFwdAlgorithm {
	*c = (ConvFwdAlgorithm)(C.miopenConvolutionFwdAlgoFFT)
	return *c
}

//WinoGrad sets c and returns WinoGrad flag
func (c *ConvFwdAlgorithm) WinoGrad() ConvFwdAlgorithm {
	*c = (ConvFwdAlgorithm)(C.miopenConvolutionFwdAlgoWinograd)
	return *c
}

//ConvBwdWeightsAlgorithm - Used for flags
//Convolutional algorithm mode for back propagation on weights
type ConvBwdWeightsAlgorithm C.miopenConvBwdWeightsAlgorithm_t

func (c ConvBwdWeightsAlgorithm) c() C.miopenConvBwdWeightsAlgorithm_t {
	return (C.miopenConvBwdWeightsAlgorithm_t)(c)
}

//GEMM sets c and returns GEMM flag
func (c *ConvBwdWeightsAlgorithm) GEMM() ConvBwdWeightsAlgorithm {
	*c = (ConvBwdWeightsAlgorithm)(C.miopenConvolutionBwdWeightsAlgoDirect)
	return *c
}

//Direct sets c and returns Direct flag
func (c *ConvBwdWeightsAlgorithm) Direct() ConvBwdWeightsAlgorithm {
	*c = (ConvBwdWeightsAlgorithm)(C.miopenConvolutionBwdWeightsAlgoDirect)
	return *c
}

//WinoGrad sets c and returns WinoGrad flag
func (c *ConvBwdWeightsAlgorithm) WinoGrad() ConvBwdWeightsAlgorithm {
	*c = (ConvBwdWeightsAlgorithm)(C.miopenConvolutionBwdWeightsAlgoWinograd)
	return *c
}

//ConvBwdDataAlgorithm - Used as flags.
// Convolutional algorithm mode for back propagation on data.
//
type ConvBwdDataAlgorithm C.miopenConvBwdDataAlgorithm_t

func (c ConvBwdDataAlgorithm) c() C.miopenConvBwdDataAlgorithm_t {
	return (C.miopenConvBwdDataAlgorithm_t)(c)
}

//GEMM sets c and returns GEMM flag
func (c *ConvBwdDataAlgorithm) GEMM() ConvBwdDataAlgorithm {
	*c = (ConvBwdDataAlgorithm)(C.miopenConvolutionBwdDataAlgoGEMM)
	return *c
}

//Direct sets c and returns Direct flag
func (c *ConvBwdDataAlgorithm) Direct() ConvBwdDataAlgorithm {
	*c = (ConvBwdDataAlgorithm)(C.miopenConvolutionBwdDataAlgoDirect)
	return *c
}

//FFT sets c and returns FFT flag
func (c *ConvBwdDataAlgorithm) FFT() ConvBwdDataAlgorithm {
	*c = (ConvBwdDataAlgorithm)(C.miopenConvolutionBwdDataAlgoFFT)
	return *c
}

//WinoGrad sets c and returns WinoGrad flag
func (c *ConvBwdDataAlgorithm) WinoGrad() ConvBwdDataAlgorithm {
	*c = (ConvBwdDataAlgorithm)(C.miopenConvolutionBwdDataAlgoWinograd)
	return *c
}

//Forward - Execute a forward convolution layer
//
//Runs the forward convolution layer based on the selected algorithm. The function
//(*ConvolutionD) FindForwardAlgorithm() must have been executed previously to
//determine the required memory needed for the workspace and the best convolutional algorithm.
//
//If using Group/Depthwise convolution mode, call (*ConvolutionD)SetGroupCount() before running
//this.
//
//	h				MIOpen handle (input)
//	alpha			Floating point scaling factor, allocated on the host (input)
//	xD				Tensor descriptor for data input tensor x (input)
//	x				Data tensor x (input)
//	wD				Tensor descriptor for weight tensor w (input)
//	w				Weights tensor w (inputs)
//	algo			Algorithm selected (inputs)
//	beta			Floating point shift factor, allocated on the host (input)
//	yD				Tensor descriptor for output data tensor y (input)
//	y				Data tensor y (output)
//	wspace			Pointer to workspace required (input)
//	wspaceSIB		Size in bytes of the memory determined by the find step (input)
func (c *ConvolutionD) Forward(
	h *Handle,
	alpha float64,
	xD *TensorD, x cutil.Mem,
	wD *TensorD, w cutil.Mem,
	algo *ConvFwdAlgorithm,
	beta float64,
	yD *TensorD, y cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenConvolutionForward(h.x, a1.CPtr(), xD.d, x.Ptr(), wD.d, w.Ptr(), c.d, algo.c(), b1.CPtr(), yD.d, y.Ptr(), wspace.Ptr(), (C.size_t)(wspaceSIB))).error("(c *ConvolutionD)Forward()")
}

//ForwardBias - Calculate element-wise scale and shift of a tensor via a bias tensor
//
//This function applies an element-wise bias to a data tensor from an input bias tensor.
//	handle         MIOpen handle (input)
//	alpha          Floating point scaling factor, allocated on the host (input)
//	bDesc          Tensor descriptor for bias tensor b (input)
//	b              Bias tensor b (input)
//	beta           Floating point shift factor, allocated on the host (input)
//	yDesc          Tensor descriptor for data tensor y (input)
//	y              Data tensor y (input and output)
func (c *ConvolutionD) ForwardBias(h *Handle,
	alpha float64,
	bD *TensorD, b cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
) error {
	dtype, _, _, err := bD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenConvolutionForwardBias(
		h.x,
		a1.CPtr(),
		bD.d, b.Ptr(),
		b1.CPtr(),
		yD.d, y.Ptr())).error("(c *ConvolutionD)ForwardBias()")
}

//BackwardData -Execute a backward data convolution layer
// Runs the backward data convolution layer based on the selected algorithm. The function
// (*ConvolutionD)GetBwdDataWorkspaceSize() must have been executed previously to
// determine the required memory needed for the workspace and the (*ConvolutionD) FindBwdDataAlgo() for the best convolutional algorithm.
//
//If using Group/Depthwise convolution mode, call  (*ConvolutionD)SetGroupCount() before running this.
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	dyD		Tensor descriptor for data input tensor dy (input)
//	dy		Data delta tensor dy (input)
//	wD		Tensor descriptor for weight tensor w (input)
//	w		Weights tensor w (input)
//	algo		Algorithm selected (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	dxD		Tensor descriptor for output data tensor dx (input)
//	dx		Data delta tensor dx (output)
//	wspace		Pointer to workspace required for the search (input)
//	wspaceSIB		Size in bytes of the memory needed for find (input)
func (c *ConvolutionD) BackwardData(h *Handle,
	alpha float64,
	dyD *TensorD, dy cutil.Mem,
	wD *TensorD, w cutil.Mem,
	algo ConvBwdDataAlgorithm,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint) error {
	dtype, _, _, err := dyD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenConvolutionBackwardData(h.x,
		a1.CPtr(),
		dyD.d, dy.Ptr(),
		wD.d, w.Ptr(),
		c.d,
		algo.c(),
		b1.CPtr(),
		dxD.d, dx.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB))).error("(c *ConvolutionD)BackwardWeights()")
}

//BackwardWeights - Execute a backward weights convolution layer
//
//Runs the backward weights convolution layer based on the selected algorithm. The function
//(*ConvolutionD)GetBwdWeightsWorkspaceSize() must have been executed previously to determine the required memory needed
//for the workspace and the (*ConvolutionD) FindBwdWeightsAlgorithm() for the best convolutional algorithm.
//
//If using Group/Depthwise convolution mode, call (*ConvolutionD)SetGroupCount() before running this.
//
//handle		MIOpen handle (input)
//alpha		Floating point scaling factor, allocated on the host (input)
//dyD		Tensor descriptor for data tensor dy (input)
//dy		Data delta tensor dy (input)
//xD		Tensor descriptor for data tensor x (input)
//x		Data tensor x (input)
//algo		Algorithm selected (input)
//beta		Floating point shift factor, allocated on the host (input)
//dwD		Tensor descriptor for weight tensor dw (input)
//dw		Weights delta tensor dw (output)
//wspace		Pointer to workspace required for the search (input)
//wspaceSIB		Size in bytes of the memory needed for find (input)
//
func (c *ConvolutionD) BackwardWeights(h *Handle,
	alpha float64,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	algo ConvBwdWeightsAlgorithm,
	beta float64,
	dwD *TensorD, dw cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenConvolutionBackwardWeights(h.x,
		a1.CPtr(),
		dyD.d, dy.Ptr(),
		xD.d, x.Ptr(),
		c.d,
		algo.c(),
		b1.CPtr(),
		dwD.d, dw.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB))).error("(c *ConvolutionD)BackwardWeights()")
}

//BackwardBias - Calculates the gradient with respect to the bias.
//
//Compute the convolution backwards gradient with respect to the bias tensor.
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	dyD		Tensor descriptor for data input tensor dy (input)
//	dy		Data delta tensor dy (input)
//	beta		point shift factor, allocated on the host (input)
//	dbD		Tensor descriptor for input bias tensor db (input)
//	db		Bias delta tensor db (output)
func (c *ConvolutionD) BackwardBias(
	h *Handle,
	alpha float64,
	dyD *TensorD, dy cutil.Mem,
	beta float64,
	dbD *TensorD, db cutil.Mem) error {
	dtype, _, _, err := dyD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenConvolutionBackwardBias(
		h.x,
		a1.CPtr(),
		dyD.d, dy.Ptr(),
		b1.CPtr(),
		dbD.d, db.Ptr())).error("(c *ConvolutionD)BackwardBias()")
}

//ConvolutionMode is the type to describe the convolution mode flags
type ConvolutionMode C.miopenConvolutionMode_t

func (c ConvolutionMode) c() C.miopenConvolutionMode_t { return C.miopenConvolutionMode_t(c) }

func (c *ConvolutionMode) cptr() *C.miopenConvolutionMode_t { return (*C.miopenConvolutionMode_t)(c) }

//Convolution sets and returns value of c to ConvolutionMode(C.miopenConvolution)
//
//Cross-Correlation convolution
func (c *ConvolutionMode) Convolution() ConvolutionMode {
	*c = ConvolutionMode(C.miopenConvolution)
	return *c
}

// Transpose sets and returns value of c to  ConvolutionMode(C.miopenTranspose)
//
//Transpose convolutions -- deconvolution
func (c *ConvolutionMode) Transpose() ConvolutionMode {
	*c = ConvolutionMode(C.miopenTranspose)
	return *c
}

//PaddingMode is used for flags for the PaddingMode. Flags are set through its methods
type PaddingMode C.miopenPaddingMode_t

func (p PaddingMode) c() C.miopenPaddingMode_t      { return (C.miopenPaddingMode_t)(p) }
func (p *PaddingMode) cptr() *C.miopenPaddingMode_t { return (*C.miopenPaddingMode_t)(p) }

//Default sets p and returns PaddingMode(C.miopenPaddingDefault) flag
//
//MIOPEN Default Padding
func (p *PaddingMode) Default() PaddingMode { *p = (PaddingMode)(C.miopenPaddingDefault); return *p }

//Same sets p and returns PaddingMode(C.miopenPaddingSame) flag
//
// Tensorflow SAME Padding
func (p *PaddingMode) Same() PaddingMode { *p = (PaddingMode)(C.miopenPaddingSame); return *p }

//Valid sets p and returns PaddingMode(C.miopenPaddingValid) flag
//
//MIOPEN VALID Padding
func (p *PaddingMode) Valid() PaddingMode { *p = (PaddingMode)(C.miopenPaddingValid); return *p }
