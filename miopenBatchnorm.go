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

//DeriveBNTensorDescriptor - Derive tensor for gamma and beta (scale and bias) from input tensor descriptor
//
//This function takes the input tensor descriptor and outputs a derived tensor for the
//normalization scale (gamma) and shift (beta) tensors.
//
//For an input tensor NCHW and spatial mode, the output derived tensor is 1C11, while for
//per-activation the derived tensor is 1CHW.
//
//For an input tensor NCDHW and spatial mode, the output derived tensor is 1C111, while for
//per-activation the derived tensor is 1CDHW.
//
//	xDesc		Input tensor descriptor (input)
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

//ForwardInference -  Execute forward inference layer for batch normalization
//
//Batch normalization pass for forward inference pass.
//Takes in batch normalization mode bn_mode and input tensor x, output tensor y, bnBias and bnScale
//with their descriptor.
//
//If either estimatedMean, or estimatedVariance are null pointers then the values for the mean and
//variance will not be used.
//
//handle				MIOpen handle (input)
//alpha				Floating point scaling factor, allocated on the host (input)
//beta				Floating point shift factor, allocated on the host (input)
//sD				Tensor descriptor for data input tensor x (input)
//x				Data tensor x (input)
//yD				Tensor descriptor for output data tensor y (input)
//y				Data tensor y (output)
//scalbiasmeanvarD				Tensor descriptor for BN scaling, shifting, saved variance and mean (input)
//scale				Batch norm scaling, gamma, tensor (input)
//bias				Batch norm bias, beta, tensor (input)
//mean				Running average saved during forward training (input)
//variance				Running variance saved during forward training (input)
//epsilon				Value to stabilize inverse variance calculation (input)
func (b *BatchNormD) ForwardInference(h *Handle, alpha, beta float64,
	xD *TensorD, x cutil.Mem,
	yD *TensorD, y cutil.Mem,
	scalbiasmeanvarD *TensorD,
	scale, bias cutil.Mem, //returned values
	mean, variance cutil.Mem, //returned values
	epsilon float64,
) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenBatchNormalizationForwardInference(h.x, b.mode, a1.CPtr(), b1.CPtr(),
		xD.d, x.Ptr(),
		yD.d, y.Ptr(),
		scalbiasmeanvarD.d,
		scale.Ptr(), bias.Ptr(),
		mean.Ptr(), variance.Ptr(),
		(C.double)(epsilon))).error("(b *BatchNormD)ForwardTraining")

}

//ForwardTraining - Execute forward training layer for batch normalization
//
//Batch normalization pass for forward training pass.
//Takes in batch normalization mode bn_mode and input tensor x, output tensor y, bnBias and bnScale
//with their descriptor.
//
//If either resultSaveMean, or resultSaveInvVariance are null pointers then the values for the mean
//and inverse variance will not be used.
//
//Likewise, if either resultRunningMean, or resultRunningVariance are null pointers then the values
//for the running mean and variance will not be saved.
//
//Running averages and variances are scaled using an exponential averaging factor:
//
//	M.old =M.new*factor + M.old*(1-factor)
//	where factor=1/(1+iteration)
//
//Params:
//	h			MIOpen handle (input)
//	alpha			Floating point scaling factor, allocated on the host (input)
//	beta			Floating point shift factor, allocated on the host (input)
//	xD			Tensor descriptor for data input tensor x (input)
//	x			Data tensor x (input)
//	yD			Tensor descriptor for output data tensor y (input)
//	y			Data tensor y (output)
//	scalbiasmeanvarD	Tensor descriptor for BN scaling, shifting, saved variance and mean (input)
//	scale			Batch norm scaling, gamma, tensor (input)
//	bias			Batch norm bias, beta, tensor (input)
//	avgfactor		Exponential averaging factor (input)
//	mean			Running average saved for inference (output)
//	variance		Running variance saved for inference (output)
//	epsilon			Value to stablize inverse variance calculation (input)
//	saveMean		Saved mini-batch mean for backwards pass (output)
//	saveInvariance		Saved mini-batch inverse variance for backwards pass (output)
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

//Backward - Execute backwards propagation layer for batch normalization
//
//Batch normalization pass for backwards propagation training pass.
//The method for backwards propagation batch normalization.
//
//Takes in batch normalization mode bn_mode and input tensor data x, input activation tensor dy,
//output tensor dx, the learned tensors resultBNBiasDiff and resultBNScaleDiff with their
//descriptor.
//
//If BOTH savedMean, and savedVariance are not null pointers then the method will use the saved
//mean and variance calculated by the forward training phase.
//
//	h				MIOpen handle (input)
//	alphaDataDiff			Floating point scaling factor, allocated on the host (input)
//	betaDataDiff			Floating point shift factor, allocated on the host (input)
//	alphaParamDiff			Floating point scaling factor, allocated on the host (input)
//	betaParamDiff			Floating point shift factor, allocated on the host (input)
//	xD				Tensor descriptor for data input tensor x (input)
//	x				Data tensor x (input)
//	dyD				Tensor descriptor for output data tensor y (input)
//	dy				Data tensor y (input)
//	dxD				Tensor descriptor for output data tensor dx (input)
//	dx				Data delta tensor dx (output)
//	scalebiasdiffD 			Tensor descriptor for BN scaling, shifting, saved variance and mean (input)
//	bnScale				Batch norm scaling, gamma, tensor (input)
//	scalediff			Tensor for dscale (output)
//	biasdiff			Tensor for dbias (output)
//	epsilon				Value to stabilize inverse variance calculation (input)
//	savedMean			Saved mini-batch mean for backwards pass (input)
//	savedInvVariance		Saved mini-bathc inverse variance for backwards pass (input)
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
