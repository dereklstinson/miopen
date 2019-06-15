package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"runtime"

	"github.com/dereklstinson/cutil"
)

//ActivationD - Activation descriptor is an object that allows the user to specify the activation mode.
type ActivationD struct {
	d C.miopenActivationDescriptor_t
}

//CreateActivationDescriptor - Creates the Activation descriptor object
func CreateActivationDescriptor() (a *ActivationD, err error) {
	a = new(ActivationD)
	err = Status(C.miopenCreateActivationDescriptor(&a.d)).error("CreateActivationDescriptor")

	runtime.SetFinalizer(a, miopenDestroyActivationDescriptor)

	return a, err
}
func miopenDestroyActivationDescriptor(a *ActivationD) error {
	return Status(C.miopenDestroyActivationDescriptor(a.d)).error("miopenDestroyActivationDescriptor")

}

//Set - Sets the activation layer descriptor details
//
//Sets all of the descriptor details for the activation layer
//
//	mode         Activation mode enum (input)
//	alpha   Alpha value for some activation modes (input)
//	beta    Beta value for some activation modes (input)
//	gamma   Gamma value for some activation modes (input)
func (a *ActivationD) Set(mode ActivationMode, alpha, beta, gamma float64) error {
	return Status(C.miopenSetActivationDescriptor(a.d, mode.c(), (C.double)(alpha), (C.double)(beta), (C.double)(gamma))).error("(a *ActivationD)Set()")
}

//Get - Gets the activation layer descriptor details
//
//Retrieves all of the descriptor details for the activation layer.
func (a *ActivationD) Get() (mode ActivationMode, alpha, beta, gamma float64, err error) {
	err = Status(C.miopenGetActivationDescriptor(a.d, mode.cptr(), (*C.double)(&alpha), (*C.double)(&beta), (*C.double)(&gamma))).error("(a *ActivationD)Get()")
	return mode, alpha, beta, gamma, err
}

//Forward - Execute an activation forward layer
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	xD		Tensor descriptor for data input tensor x (input)
//	x		Data tensor x (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	yD		Tensor descriptor for output data tensor y (input)
//	y		Data tensor y (output)
func (a *ActivationD) Forward(h *Handle, alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenActivationForward(h.x, a.d, a1.CPtr(), xD.d, x.Ptr(), b1.CPtr(), yD.d, y.Ptr())).error("(a *Activation)Forward()")
}

//Backward - Execute a activation backwards layer
//
//	h		MIOpen handle (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	yD		Tensor descriptor for input data tensor y (input)
//	y		Data tensor y (input)
//	dyD		Tensor descriptor for input data tensor dy (input)
//	dy		Data delta tensor dy (input)
//	xD		Tensor descriptor for data input tensor x (input)
//	x		Data tensor x (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	dxD		Tensor descriptor for data output tensor dx (input)
//	dx		Output data delta tensor dx (output)
func (a *ActivationD) Backward(h *Handle, alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return err
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenActivationBackward(h.x, a.d, a1.CPtr(),
		yD.d, y.Ptr(),
		dyD.d, dy.Ptr(),
		xD.d, x.Ptr(), b1.CPtr(), dxD.d, dx.Ptr())).error("(a *Activation)Forward()")
}

//ActivationMode is used for flags. Flags are set through its methods
//
//Activation layer modes
type ActivationMode C.miopenActivationMode_t

func (a ActivationMode) c() C.miopenActivationMode_t      { return (C.miopenActivationMode_t)(a) }
func (a *ActivationMode) cptr() *C.miopenActivationMode_t { return (*C.miopenActivationMode_t)(a) }

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
