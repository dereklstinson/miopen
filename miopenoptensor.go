package miopen

/*
#include <miopen/miopen.h>

*/
import (
	"C"
)

//OpTensorOp is used for flags for the Optensor functions
type OpTensorOp C.miopenTensorOp_t

func (o OpTensorOp) c() C.miopenTensorOp_t      { return C.miopenTensorOp_t(o) }
func (o *OpTensorOp) cptr() *C.miopenTensorOp_t { return (*C.miopenTensorOp_t)(o) }

//Add sets o to OpTensorOp(C.miopenTensorOpAdd) and returns the new value
func (o *OpTensorOp) Add() OpTensorOp { *o = OpTensorOp(C.miopenTensorOpAdd); return *o }

//Mul sets o to OpTensorOp(C.miopenTensorOpMul) and returns the new value
func (o *OpTensorOp) Mul() OpTensorOp { *o = OpTensorOp(C.miopenTensorOpMul); return *o }

//Min sets o to OpTensorOp(C.miopenTensorOpMin)  and returns the new value
func (o *OpTensorOp) Min() OpTensorOp { *o = OpTensorOp(C.miopenTensorOpMin); return *o }

//Max sets o to OpTensorOp(C.miopenTensorOpMax) and returns the new value
func (o *OpTensorOp) Max() OpTensorOp { *o = OpTensorOp(C.miopenTensorOpMax); return *o }

//OpTensor - This function implements: \f$ C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C \f$
//
//For Forward Bias one can also use, miopenConvolutionForwardBias()
func OpTensor(h *Handle,
	op OpTensorOp,
	alpha float64,
	aD *TensorD, a Pointer,
	alpha2 float64,
	bD *TensorD, b Pointer,
	beta float64,
	cD *TensorD, c Pointer) error {
	dtype, _, _, err := aD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	a2 := cscalarbydatatype(dtype, alpha2)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenOpTensor(h.x, op.c(), a1.CPtr(), aD.descriptor, a.Ptr(), a2.CPtr(), bD.descriptor, b.Ptr(), b1.CPtr(), cD.descriptor, c.Ptr())).error("OpTensor")
}
