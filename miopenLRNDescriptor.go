package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/cutil"
)

//LRND - LRN descriptor is an object that allows the user to specify the LRN mode, the number of elements
//in the normalization window, and the LRN k-parameter.
type LRND struct {
	d    C.miopenLRNDescriptor_t
	gogc bool
}

func CreateLRNDescriptor() (lrnDesc *LRND, err error) {
	lrnDesc = new(LRND)
	lrnDesc.gogc = true
	err = Status(C.miopenCreateLRNDescriptor(&lrnDesc.d)).error("CreateLRNDescriptor")

	runtime.SetFinalizer(lrnDesc, miopenDestroyLRNDescriptor)

	return lrnDesc, err
}

func miopenDestroyLRNDescriptor(l *LRND) error {
	return Status(C.miopenDestroyLRNDescriptor(l.d)).error("miopenDestroyLRNDescriptor")
}
func (l *LRND) Set(mode LRNMode, n uint32, alpha, beta, k float64) error {
	return Status(C.miopenSetLRNDescriptor(l.d, mode.c(), (C.uint)(n), (C.double)(alpha), (C.double)(beta), (C.double)(k))).error("(l *LRND)Set()")
}
func (l *LRND) Get() (mode LRNMode, n uint32, alpha, beta, k float64, err error) {
	var cn C.uint

	err = Status(C.miopenGetLRNDescriptor(l.d, mode.cptr(), &cn, (*C.double)(&alpha), (*C.double)(&beta), (*C.double)(&k))).error("(l *LRND)Get()")
	n = (uint32)(cn)
	return mode, n, alpha, beta, k, err
}
func (l *LRND) GetWorkSpaceSize(yD *TensorD) (wspaceSIB uint, err error) {
	var wsib C.size_t
	err = Status(C.miopenLRNGetWorkSpaceSize(yD.d, &wsib)).error("(l *LRND)GetWorkSpaceSize()")
	wspaceSIB = (uint)(wsib)
	return wspaceSIB, err
}
func (l *LRND) Forward(h *Handle, alpha float64,
	xD *TensorD, x cutil.Mem,
	beta float64,
	yD *TensorD, y cutil.Mem,
	doBackwards bool,
	wspace cutil.Mem) error {
	dtype, _, _, err := xD.Get()
	if err != nil {
		return errors.New(err.Error() + " in (*Pooling)Backward()")
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenLRNForward(h.x, l.d, a1.CPtr(), xD.d, x.Ptr(), b1.CPtr(), yD.d, y.Ptr(), (C.bool)(doBackwards), wspace.Ptr())).error(" (l *LRND)Forward()")

}
func (l *LRND) Backward(h *Handle, alpha float64,
	yD *TensorD, y cutil.Mem,
	dyD *TensorD, dy cutil.Mem,
	xD *TensorD, x cutil.Mem,
	beta float64,
	dxD *TensorD, dx cutil.Mem,
	wspace cutil.Mem) error {
	dtype, _, _, err := yD.Get()
	if err != nil {
		return errors.New(err.Error() + " in (*Pooling)Backward()")
	}
	a1 := cscalarbydatatype(dtype, alpha)
	b1 := cscalarbydatatype(dtype, beta)
	return Status(C.miopenLRNBackward(h.x, l.d, a1.CPtr(), yD.d, y.Ptr(), dyD.d, dy.Ptr(), xD.d, x.Ptr(), b1.CPtr(), dxD.d, dx.Ptr(), wspace.Ptr())).error("(l *LRND)Backward()")
}

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
