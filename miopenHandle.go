package miopen

//#include "miopen/miopen.h"
//#include <hip/hip_runtime_api.h>
import "C"

type Handle struct {
	x    C.miopenHandle_t
	gogc bool
}

func Init() {
	x := C.hipInit(0)
	if x != 0 {
		panic(x)
	}
}

func CreateHandle(usegogc bool) *Handle {
	handle := new(Handle)
	err := status(C.miopenCreate(&handle.x)).error("NewHandle")
	if err != nil {
		panic(err)
	}

	if setfinalizer {
		handle.gogc = true
		//	runtime.SetFinalizer(handle, destroycudnnhandle)
	} else {
		if usegogc {
			handle.gogc = true
			//		runtime.SetFinalizer(handle, destroycudnnhandle)
		}
	}

	return handle
}

const setfinalizer = true
