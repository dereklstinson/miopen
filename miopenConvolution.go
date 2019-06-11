package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"runtime"
)

//CreateConvolutionDescriptor -  Creates a convolution layer descriptor
//
//returns  ConvolutionD ,error
func CreateConvolutionDescriptor() (*ConvolutionD, error) {
	x := new(ConvolutionD)

	err := Status(C.miopenCreateConvolutionDescriptor(&x.descriptor)).error("CreateConvolutionDescriptor")
	if err != nil {
		return nil, err
	}
	if setfinalizer {
		runtime.SetFinalizer(x, miopenDestroyConvolutionDescriptor)
	}
	return x, nil
}

func miopenDestroyConvolutionDescriptor(c *ConvolutionD) error {
	err := Status(C.miopenDestroyConvolutionDescriptor(c.descriptor)).error("miopenDestroyConvolutionDescriptor")
	if err != nil {
		return err
	}
	c = nil
	return nil
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
//	returns error
//
// len(pad) ==len(stride) ==len(dilation)
func (c *ConvolutionD) Set(pad, stride, dilation []int32, mode ConvolutionMode) error {

	cpad := int32Tocint(pad)
	cstride := int32Tocint(stride)
	cdilation := int32Tocint(dilation)
	c.dims = C.int(len(pad))
	return Status(C.miopenInitConvolutionNdDescriptor(c.descriptor, c.dims, &cpad[0], &cstride[0], &cdilation[0], mode.c())).error(" (*ConvolutionD)Set()")
}
