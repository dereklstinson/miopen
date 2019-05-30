package miopen

//#include "miopen/miopen.h"
//#include <hip/hip_runtime_api.h>
import "C"
import (
	"errors"
	"unsafe"
)

//Streamer allowes streams from other packages to be used with this package
type Streamer interface {
	Ptr() unsafe.Pointer
	Sync() error
}

//Stream for miopen
type stream struct {
	s C.hipStream_t
}

//Ptr returns an unsafepointer of stream
func (s *stream) Ptr() unsafe.Pointer {
	return (unsafe.Pointer)(s)
}

//Sync syncs the stream
func (s *stream) Sync() error {
	x := C.hipStreamSynchronize((s.s))
	if x == 0 {
		return nil
	}
	return errors.New("Error with HIP stream ")
}

//Streamer allows other libraries with amd to pass its stream
