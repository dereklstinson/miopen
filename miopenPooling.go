package miopen
//#include <miopen/miopen.h>
import "C"
import "errors"

//PoolingD - Pooling descriptor is an object that allows the user to specify the dimension sizes of the
//pooling windows, paddings, strides, and pooling mode.
type PoolingD struct {
	d C.miopenPoolingDescriptor_t
	dims C.int
}

func CreatePoolingDescriptor()(p *PoolingD,err error){
	p=new(PoolingD)
err=Status(C.miopenCreatePoolingDescriptor(&p.d)).error("CreatePoolingDescriptor")
return p,err
}

func (p *PoolingD)SetIndex(index IndexType)error{
return	Status(C.miopenSetPoolingIndexType(p.d,index.c())).error("SetIndex")
}
func (p *PoolingD)GetIndex()(index IndexType, err error){
	err=Status(C.miopenGetPoolingIndexType(p.d,index.cptr())).error("GetIndex")
	return index,err
}

//Set 2d pooling descriptor is only supported
func (p *PoolingD)Set(mode PoolingMode, window,pad,stride []int32)error{
	if len(window)!=2 || len(pad) !=2 ||len(stride)!=2{
	return	errors.New("(*Pooling)Set() : len(window)!=2 || len(pad) !=2 ||len(stride)!=2")
	}
	p.dims=(C.int)(len(window))
padding:=int32Tocint(pad)
s:=int32Tocint(stride)
w:=int32Tocint(window)
return Status(C.miopenSet2dPoolingDescriptor(p.d,mode.c(),w[0], w[1],padding[0],padding[1],s[0],s[1])).error("(p *Pooling)Set()")
}