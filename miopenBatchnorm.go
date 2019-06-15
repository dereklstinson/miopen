package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"errors"

	"github.com/dereklstinson/cutil"
)

//BatchNormD is an original to these bindings.  This is to make the batchnorm operation similar to the majority of these bindings.
type BatchNormD struct {
	mode C.miopenBatchNormMode_t
	set  bool
	gogc bool
}

//CreateBatchNormDescriptor creates a new BatchNormD
func CreateBatchNormDescriptor() (*BatchNormD, error) {
	x := new(BatchNormD)
	x.gogc = true
	return x, nil
}

//Set sets the values used in the batchnorm descriptor
func (b *BatchNormD) Set(mode BatchNormMode) error {
	b.mode = mode.c()
	b.set = true
	return nil
}

//Get gets the values stored in BatchNormMode
func (b *BatchNormD) Get() (mode BatchNormMode, err error) {
	if !b.set {
		return 0, errors.New("BatchNormD not set")
	}
	return BatchNormMode(b.mode), nil
}

//DeriveBNTensorDescriptor Derives a BN Tensor Descriptor from the one passed.
/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
 */
func (b *BatchNormD) DeriveBNTensorDescriptor(xDesc *TensorD) (bndesc *TensorD, err error) {
	if !b.set {
		return nil, errors.New("BatchNormD not set")
	}
	return miopenDeriveBNTensorDescriptor(xDesc, BatchNormMode(b.mode), b.gogc)
}

//Private Func
func miopenDeriveBNTensorDescriptor(xDesc *TensorD, mode BatchNormMode, gogc bool) (descriptor *TensorD, err error) {
	if xDesc.dims > 5 || xDesc.dims < 4 {
		return nil, errors.New("dims for descriptor must be 4 or 5")
	}

	descriptor, err = createtensordescriptor()

	if err != nil {
		return nil, err
	}
	err = Status(C.miopenDeriveBNTensorDescriptor(descriptor.d, xDesc.d, mode.c())).error("DeriveBNTensorDescriptor-Derive")
	descriptor.dims = xDesc.dims
	return descriptor, err
}

func (b *BatchNormD) ForwardTraining(h *Handle, alpha, beta float64,
	xD *TensorD, x cutil.Mem,
	yD *TensorD, y cutil.Mem,
	scalbiasmeanvarD *TensorD,
	scale, bias cutil.Mem, //returned values
	avgfactor float64,
	mean, variance cutil.Mem, //returned values
	epsilon float64,
	saveMean, saveInvariance cutil.Mem, //returned vallues
) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenBatchNormalizationForwardTraining(h.x, b.mode, a1.CPtr(), b1.CPtr(),
		xD.d, x.Ptr(),
		yD.d, y.Ptr(),
		scalbiasmeanvarD.d,
		scale.Ptr(), bias.Ptr(),
		(C.double)(avgfactor),
		mean.Ptr(), variance.Ptr(),
		(C.double)(epsilon),
		saveMean.Ptr(), saveInvariance.Ptr())).error("(b *BatchNormD)ForwardTraining")

}

func (b *BatchNormD) Backward(h *Handle, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff float64,
	xD *TensorD, x cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	dxD *TensorD, dx cutil.Mem,
	scalebiasdiffD *TensorD,
	scale, scalediff, biasdiff cutil.Mem,
	epsilon float64,
	savedMean, savedInvVariance cutil.Mem) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alphaDataDiff)
	b1 := cscalarbydatatype(dtype, betaDataDiff)
	a2 := cscalarbydatatype(dtype, alphaParamDiff)
	b2 := cscalarbydatatype(dtype, betaParamDiff)
	return Status(C.miopenBatchNormalizationBackward(h.x, b.mode, a1.CPtr(), b1.CPtr(), a2.CPtr(), b2.CPtr(), xD.d, x.Ptr(), dyD.d, dy.Ptr(), dxD.d, dx.Ptr(), scalebiasdiffD.d,
		scale.Ptr(), scalediff.Ptr(), biasdiff.Ptr(), (C.double)(epsilon), savedMean.Ptr(), savedInvVariance.Ptr())).error("(b *BatchNormD)Backward()")
}

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
