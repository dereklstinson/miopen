package miopen

/*
#include <miopen/miopen.h>

*/
import "C"

//RNND - Recurrent Neural Network descriptor
type RNND struct {
	d C.miopenRNNDescriptor_t
}

//DataType is used for flags for the tensor layer structs
type DataType C.miopenDataType_t

// Float sets d to Float and returns the changed value
func (d *DataType) Float() DataType { *d = DataType(C.miopenFloat); return *d }

// Int8 sets d to Int8 and returns the changed value
//
//Partial Support
func (d *DataType) Int8() DataType { *d = DataType(C.miopenInt8); return *d }

// Int32 sets d to Int32 and returns the changed value
//
//Not Supported
func (d *DataType) Int32() DataType { *d = DataType(C.miopenInt32); return *d }

//Half sets d to Half and returns the changed value
func (d *DataType) Half() DataType { *d = DataType(C.miopenHalf); return *d }

//Int8x4 sets d to  Int8x4 and returns the changed value
//
//Partial Support
func (d *DataType) Int8x4() DataType { *d = DataType(C.miopenInt8x4); return *d }

func (d DataType) c() C.miopenDataType_t      { return C.miopenDataType_t(d) }
func (d *DataType) cptr() *C.miopenDataType_t { return (*C.miopenDataType_t)(d) }

//ToString will return a human readable string that can be printed for debugging.
func (d DataType) ToString() string {
	var flg DataType
	switch d {
	case flg.Float():
		return "Float"
	case flg.Int8():
		return "Int8"
	case flg.Int32():
		return "Int32"
	case flg.Half():
		return "Half"
	case flg.Int8x4():
		return "Int8x4"
	}
	return "ERROR no such flag"
}

//IndexType MIOpen index datatypes.
type IndexType C.miopenIndexType_t

func (i IndexType) c() C.miopenIndexType_t      { return C.miopenIndexType_t(i) }
func (i *IndexType) cptr() *C.miopenIndexType_t { return (*C.miopenIndexType_t)(i) }

//Uint8 sets i to Uint8 and returns the changed value
func (i *IndexType) Uint8() IndexType { *i = IndexType(C.miopenIndexUint8); return *i }

//Uint16 sets i to Uint16 and returns the changed value
func (i *IndexType) Uint16() IndexType { *i = IndexType(C.miopenIndexUint16); return *i }

//Uint32 sets i to Uint32 and returns the changed value
func (i *IndexType) Uint32() IndexType { *i = IndexType(C.miopenIndexUint32); return *i }

//Uint64 sets i to Uint64 and returns the changed value
func (i *IndexType) Uint64() IndexType { *i = IndexType(C.miopenIndexUint64); return *i }
