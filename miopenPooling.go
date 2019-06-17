package miopen

//#include <miopen/miopen.h>
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/cutil"
)

//PoolingD - Pooling descriptor is an object that allows the user to specify the dimension sizes of the
//pooling windows, paddings, strides, and pooling mode.
type PoolingD struct {
	d    C.miopenPoolingDescriptor_t
	dims C.int
}

//CreatePoolingDescriptor - Creates a pooling layer descriptor
func CreatePoolingDescriptor() (p *PoolingD, err error) {
	p = new(PoolingD)
	err = Status(C.miopenCreatePoolingDescriptor(&p.d)).error("CreatePoolingDescriptor")

	runtime.SetFinalizer(p, miopenDestroyPoolingDescriptor)
	return p, err
}

func miopenDestroyPoolingDescriptor(p *PoolingD) error {
	return Status(C.miopenDestroyPoolingDescriptor(p.d)).error("miopenDestroyPoolingDescriptor")

}

//SetIndexType - Set index data type for pooling layer. The default indexing type is uint8_t.
//
//Users can set the index type to any of the miopenIndexType_t sizes; 8, 16, 32, or 64 bit
//unsigned integers.
//
//	index   Index type (input)
func (p *PoolingD) SetIndexType(index IndexType) error {
	return Status(C.miopenSetPoolingIndexType(p.d, index.c())).error("SetIndex")
}

//GetIndexType - Get the index data type for pooling layer. The index type to any of the
//IndexType sizes; 8, 16, 32, or 64 bit unsigned integers.
func (p *PoolingD) GetIndexType() (index IndexType, err error) {
	err = Status(C.miopenGetPoolingIndexType(p.d, index.cptr())).error("GetIndex")
	return index, err
}

//Set - Sets a pooling layer descriptor details. (2D only right now)
//
//Sets the window shape, padding, and stride for a previously created 2-D pooling descriptor.
//
//	mode		Pooling mode enum (input)
//	window	Input window dimension (input)
//	pad          Number of elements to pad (input)
//	stride       Number of elements to stride over (input)
func (p *PoolingD) Set(mode PoolingMode, window, pad, stride []int32) error {
	if len(window) != 2 || len(pad) != 2 || len(stride) != 2 {
		return errors.New("(*Pooling)Set() : len(window)!=2 || len(pad) !=2 ||len(stride)!=2")
	}
	p.dims = (C.int)(len(window))
	padding := int32Tocint(pad)
	s := int32Tocint(stride)
	w := int32Tocint(window)
	return Status(C.miopenSet2dPoolingDescriptor(p.d, mode.c(), w[0], w[1], padding[0], padding[1], s[0], s[1])).error("(p *Pooling)Set()")
}

//Get - Gets layer descriptor details. (2D only right now)
//
//Gets the window shape, padding, and stride for a previously created pooling descriptor.
func (p *PoolingD) Get() (mode PoolingMode, window, pad, stride []int32, err error) {
	cw := make([]C.int, p.dims)
	cp := make([]C.int, p.dims)
	cs := make([]C.int, p.dims)
	err = Status(C.miopenGet2dPoolingDescriptor(p.d, mode.cptr(), &cw[0], &cw[1], &cp[0], &cp[1], &cs[0], &cs[1])).error("(p *Pooling)Get()")
	window = cintToint32(cw)
	pad = cintToint32(cp)
	stride = cintToint32(cs)
	return mode, window, pad, stride, err
}

//GetForwardOutputDim - Gets the shape of the output tensor
//
//Retrieve the tensor dimensions for the forward 2-D pooling. This call is required for
//the forward if the output dimensions are different than the input tensor
//dimensions.
//
//	tD		Input tensor descriptor (input)
func (p *PoolingD) GetForwardOutputDim(tD *TensorD) (dims []int32, err error) {
	cdims := make([]C.int, 4)
	err = Status(C.miopenGetPoolingForwardOutputDim(p.d, tD.d, &cdims[0], &cdims[1], &cdims[2], &cdims[3])).error("(p *Pooling)GetForwardOutputDim()")
	dims = cintToint32(cdims)
	return dims, err
}

//GetWSpaceSize - Get the amount of GPU memory required for pooling
//
//Retrieves the amount of workspace in bytes require for pooling. This call is required to
//determine the amount of GPU memory needed for the backwards pooling algorithms. For max-
//pooling, there is no assumption on index data type. As the user can set the index datatype
//size using miopenSetPoolingIndexType().
//
//yD		Descriptor for pooling layer (input)
func (p *PoolingD) GetWSpaceSize(yD *TensorD) (wspaceSIB uint, err error) {
	var ws C.size_t
	err = Status(C.miopenPoolingGetWorkSpaceSizeV2(p.d, yD.d, &ws)).error("(*PoolingD) GetForwardOutputDim")
	wspaceSIB = (uint)(ws)
	return wspaceSIB, err
}

//Forward - Execute a forward pooling layer
//
//Runs forward pooling. miopenGetPoolingForwardOutputDim() should be called before
//miopenPoolingForward().
//If the parameter do_backward == 0, then set workSpace = nullptr and workSpaceSize = 0. However,
//for back-propagation do_backwards must be set to 1 in miopenPoolingForward().
//
//h         MIOpen handle (input)
//alpha          Floating point scaling factor, allocated on the host (input)
//xD          Tensor descriptor for data input tensor x (input)
//x              Data tensor x (input)
//beta           Floating point shift factor, allocated on the host (input)
//yD          Tensor descriptor for output data tensor y (input)
//y              Data tensor y (output)
//do_backward    Boolean to toggle save data in workspace for backwards pass (input)
//wspace      Pointer user allocated memory (input)
//wspaceSIB  Size in bytes of the memory needed (input)
func (p *PoolingD) Forward(h *Handle, alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
	dobackwards bool, wspace cutil.Mem, wspaceSIB uint) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return errors.New(err.Error() + " in (*Pooling)Forward()")
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenPoolingForward(h.x, p.d, a1.CPtr(),
		xD.d, x.Ptr(),
		b1.CPtr(),
		yD.d, y.Ptr(),
		(C.bool)(dobackwards), wspace.Ptr(), (C.size_t)(wspaceSIB))).error("(*Pooling)Forward()")
}

//Backward - Execute a backward pooling layer
//
//Runs backward pooling. (p *PoolingD) GetWSpaceSize() must be called before
//(p *PoolingD) Backward() to determine the amount of workSpace to be allocated.
//
//h         MIOpen handle (input)
//alpha          Floating point scaling factor, allocated on the host (input)
//yD          Tensor descriptor for output data tensor y (input)
//y              Data tensor y (input)
//dyD         Tensor descriptor for data input tensor dy (input)
//dy             Data delta tensor dy (input)
//xD          Tensor descriptor for output data tensor x (input)
//x              Data tensor x (output)
//beta           Floating point shift factor, allocated on the host (input)
//dxD         Tensor descriptor for tensor dx (input)
//dx             Weights delta tensor dx (output)
//wspace      Pointer to user allocated workspace (input)
func (p *PoolingD) Backward(h *Handle, alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return errors.New(err.Error() + " in (*Pooling)Backward()")
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenPoolingBackward(h.x, p.d, a1.CPtr(), yD.d, y.Ptr(), dyD.d, dy.Ptr(), xD.d, x.Ptr(), b1.CPtr(), dxD.d, dx.Ptr(), wspace.Ptr())).error("(*PoolingD)Backward()")
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
