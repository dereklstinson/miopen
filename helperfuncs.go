package miopen

import "C"
import "github.com/dereklstinson/cutil"

func int32Tocint(x []int32) []C.int {
	y := make([]C.int, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = C.int(x[i])
	}
	return y
}
func cintToint32(x []C.int) []int32 {
	y := make([]int32, len(x))
	for i := 0; i < len(x); i++ {
		y[i] = int32(x[i])
	}
	return y
}
func comparedims(dims ...[]int32) bool {
	totallength := len(dims)
	if totallength == 1 {
		return true
	}
	for i := 1; i < totallength; i++ {
		if len(dims[0]) != len(dims[i]) {
			return false
		}
		for j := 0; j < len(dims[0]); j++ {
			if dims[0][j] != dims[i][j] {
				return false
			}
		}
	}
	return true
}
func findvolume(dims []int32) int32 {
	mult := int32(1)
	for i := range dims {
		mult *= dims[i]
	}
	return mult
}
func stridecalc(dims []int32) []int32 {
	strides := make([]int32, len(dims))
	stride := int32(1)
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return strides
}

//CScalarByDataType takes the DataType flag and puts num into a CScalar interface. The value of num will be bound by what is passed for DataType.
//If a DataType isn't supported by the function it will return nil.
func cscalarbydatatype(dtype DataType, num float64) cutil.CScalar {
	var x DataType //CUDNN_DATATYPE_FLOAT
	switch dtype {
	case x.Float():
		return cutil.CFloat(num)
	case x.Int32():
		y := float32(num)
		return cutil.CFloat(y)
	case x.Int8():
		y := float32(num)
		return cutil.CFloat(y)
	case x.Half():
		y := float32(num)
		return cutil.CFloat(y)
	default:
		return nil
	}

}
