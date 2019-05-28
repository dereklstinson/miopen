package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import "errors"

//Status is the error return used in miopen
type status C.miopenStatus_t

func (s status) error(comment string) error {
	x := (C.miopenStatus_t)(s)
	switch x {
	case C.miopenStatusSuccess:
		return nil
	case C.miopenStatusNotInitialized:
		return errors.New("miopenStatusNotInitialized : " + comment)
	case C.miopenStatusInvalidValue:
		return errors.New("miopenStatusInvalidValue : " + comment)
	case C.miopenStatusBadParm:
		return errors.New("miopenStatusBadParm : " + comment)
	case C.miopenStatusAllocFailed:
		return errors.New("miopenStatusAllocFailed : " + comment)
	case C.miopenStatusInternalError:
		return errors.New("miopenStatusInternalError : " + comment)
	case C.miopenStatusNotImplemented:
		return errors.New("miopenStatusNotImplemented : " + comment)
	case C.miopenStatusUnknownError:
		return errors.New("miopenStatusUnknownError : " + comment)
	case C.miopenStatusUnsupportedOp:
		return errors.New("miopenStatusUnsupportedOp : " + comment)
	}
	return errors.New("go miopen new error not in go binding")
}
