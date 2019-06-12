package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import "unsafe"

//Pointer - For miopen a Pointer needs to have a function Ptr() Which points to memory used by amd. (usually on device)
type Pointer interface {
	Ptr() unsafe.Pointer
}

//FusionOpD - Fusion Operator Descriptor contains the meta-data associated with an operator
type FusionOpD struct {
	d C.miopenFusionOpDescriptor_t
}

//ConvolutionD - Convolution descriptor is an object that allows the user to specify a layer's padding, stride,
//and dilation of the convolutional filter. Parameters must all be non-negative.
type ConvolutionD struct {
	dims C.int
	d    C.miopenConvolutionDescriptor_t
}


//LRND - LRN descriptor is an object that allows the user to specify the LRN mode, the number of elements
//in the normalization window, and the LRN k-parameter.
type LRND struct {
	d C.miopenLRNDescriptor_t
}

//ActivationD - Activation descriptor is an object that allows the user to specify the activation mode.
type ActivationD struct {
	d C.miopenActivationDescriptor_t
}

//RNND - Recurrent Neural Network descriptor
type RNND struct {
	d C.miopenRNNDescriptor_t
}

//DataType is used for flags for the tensor layer structs
type DataType C.miopenDataType_t

// Float sets d to Float and returns the changed value
func (d *DataType) Float() DataType { *d = DataType(C.miopenFloat); return *d }

// Int8 sets d to Int8 and returns the changed value
//
//Partial Support
func (d *DataType) Int8() DataType { *d = DataType(C.miopenInt8); return *d }

// Int32 sets d to Int32 and returns the changed value
//
//Not Supported
func (d *DataType) Int32() DataType { *d = DataType(C.miopenInt32); return *d }

//Half sets d to Half and returns the changed value
func (d *DataType) Half() DataType { *d = DataType(C.miopenHalf); return *d }

//Int8x4 sets d to  Int8x4 and returns the changed value
//
//Partial Support
func (d *DataType) Int8x4() DataType { *d = DataType(C.miopenInt8x4); return *d }

func (d DataType) c() C.miopenDataType_t      { return C.miopenDataType_t(d) }
func (d *DataType) cptr() *C.miopenDataType_t { return (*C.miopenDataType_t)(d) }

//ToString will return a human readable string that can be printed for debugging.
func (d DataType) ToString() string {
	var flg DataType
	switch d {
	case flg.Float():
		return "Float"
	case flg.Int8():
		return "Int8"
	case flg.Int32():
		return "Int32"
	case flg.Half():
		return "Half"
	case flg.Int8x4():
		return "Int8x4"
	}
	return "ERROR no such flag"
}

//IndexType MIOpen index datatypes.
type IndexType C.miopenIndexType_t

func (i IndexType) c() C.miopenIndexType_t      { return C.miopenIndexType_t(i) }
func (i *IndexType) cptr() *C.miopenIndexType_t { return (*C.miopenIndexType_t)(i) }

//Uint8 sets i to Uint8 and returns the changed value
func (i *IndexType) Uint8() IndexType { *i = IndexType(C.miopenIndexUint8); return *i }

//Uint16 sets i to Uint16 and returns the changed value
func (i *IndexType) Uint16() IndexType { *i = IndexType(C.miopenIndexUint16); return *i }

//Uint32 sets i to Uint32 and returns the changed value
func (i *IndexType) Uint32() IndexType { *i = IndexType(C.miopenIndexUint32); return *i }

//Uint64 sets i to Uint64 and returns the changed value
func (i *IndexType) Uint64() IndexType { *i = IndexType(C.miopenIndexUint64); return *i }

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

//PoolingMode is used for flags in pooling
type PoolingMode C.miopenPoolingMode_t

func (p PoolingMode) c() C.miopenPoolingMode_t      { return C.miopenPoolingMode_t(p) }
func (p *PoolingMode) cptr() *C.miopenPoolingMode_t { return (*C.miopenPoolingMode_t)(p) }

//Max sets p and returns PoolingMode(C.CUDNN_POOLING_MAX) flag
//
//The maximum value inside the pooling window is used.
func (p *PoolingMode) Max() PoolingMode { *p = PoolingMode(C.miopenPoolingMax); return *p }

//Average sets p and returns PoolingMode(C.miopenPoolingAverage) flag
//
//Average
func (p *PoolingMode) Average() PoolingMode {
	*p = PoolingMode(C.miopenPoolingAverage)
	return *p
}

//AverageInclusive returns PoolingMode(C.miopenPoolingAverageInclusive) flag
//
//AverageInclusive
func (p *PoolingMode) AverageInclusive() PoolingMode {
	*p = PoolingMode(C.miopenPoolingAverageInclusive)
	return *p
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

//LRNMode is used for flags for the LRNMode. Flags are set through its methods
//
//Local Response Normalization layer mode
type LRNMode C.miopenLRNMode_t

func (l LRNMode) c() C.miopenLRNMode_t      { return (C.miopenLRNMode_t)(l) }
func (l *LRNMode) cptr() *C.miopenLRNMode_t { return (*C.miopenLRNMode_t)(l) }

//WithinChannel sets l and returns LRNMode(C.miopenLRNWithinChannel) flag
//
// Channel independent
func (l *LRNMode) WithinChannel() LRNMode { *l = (LRNMode)(C.miopenLRNWithinChannel); return *l }

//CrossChannel sets l and returns LRNMode(C.miopenLRNCrossChannel) flag
//
// Cross Channel
func (l *LRNMode) CrossChannel() LRNMode { *l = (LRNMode)(C.miopenLRNCrossChannel); return *l }

//BatchNormMode is used for flags. Flags are set through its methods
//
//Batch Normalization layer mode
type BatchNormMode C.miopenBatchNormMode_t

func (b BatchNormMode) c() C.miopenBatchNormMode_t      { return (C.miopenBatchNormMode_t)(b) }
func (b *BatchNormMode) cptr() *C.miopenBatchNormMode_t { return (*C.miopenBatchNormMode_t)(b) }

//PerActivation sets b and returns BatchNormMode(C.miopenBNPerActivation) flag
//
//Element-wise normalization for fully connected layer
func (b *BatchNormMode) PerActivation() BatchNormMode {
	*b = (BatchNormMode)(C.miopenBNPerActivation)
	return *b
}

//Spatial sets b and returns BatchNormMode(C.miopenBNSpatial) flag
//
//Mini-batch spatial normalization for convolutional layers
func (b *BatchNormMode) Spatial() BatchNormMode { *b = (BatchNormMode)(C.miopenBNSpatial); return *b }

//ActivationMode is used for flags. Flags are set through its methods
//
//Activation layer modes
type ActivationMode C.miopenActivationMode_t

func (a ActivationMode) c() C.miopenBatchNormMode_t      { return (C.miopenBatchNormMode_t)(a) }
func (a *ActivationMode) cptr() *C.miopenBatchNormMode_t { return (*C.miopenBatchNormMode_t)(a) }

//PasThru sets a and returns ActivationMode(C.miopenActivationPASTHRU) flag
//
//No activation, pass through the data
func (a *ActivationMode) PasThru() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationPASTHRU)
	return *a
}

//Logistic sets a and returns ActivationMode(C.miopenActivationLOGISTIC) flag
//
// Sigmoid function: 1 / (1 + e^{-x})
func (a *ActivationMode) Logistic() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationLOGISTIC)
	return *a
}

//Tanh sets a and returns ActivationMode(C.miopenActivationTANH) flag
//
//Tanh activation: beta * tanh(alpha * x)
func (a *ActivationMode) Tanh() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationTANH)
	return *a
}

//Relu sets a and returns ActivationMode(C.miopenActivationRELU) flag
//
//Rectified Linear Unit:  max(0, x)
func (a *ActivationMode) Relu() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationRELU)
	return *a
}

//SoftRelu sets a and returns ActivationMode(C.miopenActivationSOFTRELU) flag
//
//SoftRelu activation: log(1 + e^x)
func (a *ActivationMode) SoftRelu() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationSOFTRELU)
	return *a
}

//Abs sets a and returns ActivationMode(C.miopenActivationABS) flag
//
//Absolute value abs(x)
func (a *ActivationMode) Abs() ActivationMode { *a = (ActivationMode)(C.miopenActivationABS); return *a }

//Power sets a and returns ActivationMode(C.miopenActivationPOWER) flag
//
//Scaled and shifted power (alpha + beta * x)^{gamma}
func (a *ActivationMode) Power() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationPOWER)
	return *a
}

//ClippedRelu sets a and returns ActivationMode(C.miopenActivationCLIPPEDRELU) flag
//
//Clipped Rectified Linear Unit: min(alpha, max(0,x))
func (a *ActivationMode) ClippedRelu() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationCLIPPEDRELU)
	return *a
}

//LeakyRelu sets a and returns ActivationMode(C.miopenActivationLEAKYRELU) flag
//
//Leaky Rectified Linear Unit: alpha * x | x <= 0; x | x > 0
func (a *ActivationMode) LeakyRelu() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationLEAKYRELU)
	return *a
}

//Elu sets a and returns ActivationMode(C.miopenActivationELU) flag
//
//Exponential Rectified Linear Unit: alpha * (e^{x} - 1) | x <= 0; x | x > 0
func (a *ActivationMode) Elu() ActivationMode {
	*a = (ActivationMode)(C.miopenActivationELU)
	return *a
}
