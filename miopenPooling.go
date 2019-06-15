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

func CreatePoolingDescriptor() (p *PoolingD, err error) {
	p = new(PoolingD)
	err = Status(C.miopenCreatePoolingDescriptor(&p.d)).error("CreatePoolingDescriptor")

	runtime.SetFinalizer(p, miopenDestroyPoolingDescriptor)
	return p, err
}

func miopenDestroyPoolingDescriptor(p *PoolingD) error {
	return Status(C.miopenDestroyPoolingDescriptor(p.d)).error("miopenDestroyPoolingDescriptor")

}

func (p *PoolingD) SetIndex(index IndexType) error {
	return Status(C.miopenSetPoolingIndexType(p.d, index.c())).error("SetIndex")
}
func (p *PoolingD) GetIndex() (index IndexType, err error) {
	err = Status(C.miopenGetPoolingIndexType(p.d, index.cptr())).error("GetIndex")
	return index, err
}

//Set 2d pooling descriptor is only supported
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
func (p *PoolingD) GetForwardOutputDim(tD *TensorD) (dims []int32, err error) {
	cdims := make([]C.int, 4)
	err = Status(C.miopenGetPoolingForwardOutputDim(p.d, tD.d, &cdims[0], &cdims[1], &cdims[2], &cdims[3])).error("(p *Pooling)GetForwardOutputDim()")
	dims = cintToint32(cdims)
	return dims, err
}
func (p *PoolingD) GetWSpaceSize(yD *TensorD) (wspaceSIB uint, err error) {
	var ws C.size_t
	err = Status(C.miopenPoolingGetWorkSpaceSizeV2(p.d, yD.d, &ws)).error("(*PoolingD) GetForwardOutputDim")
	wspaceSIB = (uint)(ws)
	return wspaceSIB, err
}

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
