package miopen

/*
#cgo CFLAGS: -D__HIP_PLATFORM_HCC__ -D__HIP_VDI__
#cgo CFLAGS: -I/opt/rocm/hip/include -I/opt/rocm/miopen/include
#cgo LDFLAGS: "-L/opt/rocm/hip/lib" "-L/opt/rocm/miopen/lib" -lhip_hcc -lMIOpen

*/
import "C"
