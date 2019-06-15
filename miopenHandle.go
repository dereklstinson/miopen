package miopen

//#include "miopen/miopen.h"
//#include <hip/hip_runtime_api.h>
import "C"
import "runtime"

//Handle handles the functions for miopen
type Handle struct {
	x C.miopenHandle_t
}

func init() {
	x := C.hipInit(0)
	if x != 0 {
		panic(x)
	}
}

//CreateHandle creates a handle.
func CreateHandle() *Handle {
	handle := new(Handle)
	err := Status(C.miopenCreate(&handle.x)).error("NewHandle")
	if err != nil {
		panic(err)
	}

	runtime.SetFinalizer(handle, miopenDestroy)

	return handle
}

func miopenDestroy(h *Handle) error {
	return Status(C.miopenDestroy(h.x)).error("(*Handle).Destroy")
}

//SetStream passes a stream to sent in the cuda handle
func (h *Handle) SetStream(s Streamer) error {

	y := C.miopenSetStream(h.x, C.miopenAcceleratorQueue_t(s.Ptr()))

	return Status(y).error("(*Handle).SetStream")
}

//GetStream will return a stream that the handle is using
func (h *Handle) GetStream() (Streamer, error) {

	s := new(stream)
	err := Status(C.miopenGetStream(h.x, &s.s)).error("(*Handle).GetStream")
	return s, err
}

//GetKernelTime -This function is used only when profiling mode has been enabled.
//
//Kernel timings are based on the MIOpen handle and is not thread-safe.
//In order to use multi-threaded profiling, create an MIOpen handle for each
//concurrent thread.
func (h *Handle) GetKernelTime() (time float32, err error) {
	err = Status(C.miopenGetKernelTime(h.x, (*C.float)(&time))).error("GetKernelTime")
	return time, err
}

//EnableProfiling - Enables profiling to retrieve kernel time
//
//Enable or disable kernel profiling. This profiling is only for kernel time.
func (h *Handle) EnableProfiling(enable bool) (err error) {
	return Status(C.miopenEnableProfiling(h.x, (C.bool)(enable))).error("EnableProfiling")

}

/*TODO:
MIOPEN_EXPORT miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                                miopenAllocatorFunction allocator,
                                                miopenDeallocatorFunction deallocator,
                                                void* allocatorContext);



*/
