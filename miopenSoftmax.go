package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import "github.com/dereklstinson/cutil"

//SoftMaxD holds the methods to call the soft max function. This is so it keeps uniform with the other descriptors
//
//MIOpen does not support Softmax modes. MIOpen implements the SOFTMAX_MODE_CHANNEL flavor.
type SoftMaxD struct {
}

//CreateSoftMax - Creates a soft max method holder
func CreateSoftMax() (*SoftMaxD, error) {
	return &SoftMaxD{}, nil
}

//Forward - Execute a softmax forward layer
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	xD		Tensor descriptor for data input tensor x (input)
//	x		Data tensor x (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	yD		Tensor descriptor for output data tensor y (input)
//	y		Data tensor y (output)
func (s *SoftMaxD) Forward(h *Handle, alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenSoftmaxForward(h.x, a1.CPtr(), xD.d, x.Ptr(), b1.CPtr(), yD.d, y.Ptr())).error("(s *SoftMaxD)Forward()")
}

//Backward - Execute a softmax backwards layer
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	yD		Tensor descriptor for input data tensor y (input)
//	y		Data tensor y (input)
//	dyD		Tensor descriptor for input data tensor dy (input)
//	dy		Data delta tensor dy (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	dxD		Tensor descriptor for data output tensor dx (input)
//	dx		Output data delta tensor dx (output)
func (s *SoftMaxD) Backward(h *Handle, alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem) error {
	dtype, _, _, err := yD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha).CPtr()
	b1 := cscalarbydatatype(dtype, beta).CPtr()
	return Status(C.miopenSoftmaxBackward(h.x, a1, yD.d, y.Ptr(), dyD.d, dy.Ptr(), b1, dxD.d, dx.Ptr())).error("(s *SoftMaxD)Backward()")
}
