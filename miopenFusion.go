package miopen

//#include <miopen/miopen.h>
import "C"
import "runtime"

//FusionPlanD - MIOpen fusion interface
type FusionPlanD struct {
	d C.miopenFusionPlanDescriptor_t
}

//FusionOpD - Fusion Operator Descriptor contains the meta-data associated with an operator
type FusionOpD struct {
	d C.miopenFusionOpDescriptor_t
}

//CreateFusionPlan - Creates the kenrel fusion plan descriptor object
//
//direction		Horizontal or Vertical fusion (input)
//inputD		Descriptor to tensor for the input (input)
func CreateFusionPlan(direction FusionDirection, inputD *TensorD) (fpD *FusionPlanD, err error) {
	fpD = new(FusionPlanD)
	err = Status(C.miopenCreateFusionPlan(&fpD.d, direction.c(), inputD.d)).error("CreateFusionPlan")
	runtime.SetFinalizer(fpD, miopenDestroyFusionPlan)
	return fpD, err
}
func miopenDestroyFusionPlan(f *FusionPlanD) error {
	return Status(C.miopenDestroyFusionPlan(f.d)).error("miopenDestroyFusionPlan")
}

func (f *FusionPlanD) Compile(h *Handle) error {
	return Status(C.miopenCompileFusionPlan(h.x, f.d)).error("(f *FusionPlanD) Compile()")
}

func (f *FusionPlanD) GetOp(opIdx int32) (op *FusionOpD, err error) {
	op = new(FusionOpD)
	err = Status(C.miopenFusionPlanGetOp(f.d, (C.int)(opIdx), &op.d)).error("f *FusionPlanD)GetOp()")
	return op, err
}
func (f *FusionPlanD) GetWorkSpaceSize(h *Handle, algo ConvFwdAlgorithm) (wspaceSIB uint, err error) {
	var ws C.size_t
	err = Status(C.miopenFusionPlanGetWorkSpaceSize(h.x, f.d, &ws, algo.c())).error("(f *FusionPlanD)GetWorkSpaceSize()")
	wspaceSIB = (uint)(ws)
	return wspaceSIB, err
}
func (f *FusionPlanD) ConvolutionGetAlgo() (algos []ConvFwdAlgorithm, err error) {
	var actual C.int
	request := C.int(4)
	algos = make([]ConvFwdAlgorithm, request)
	err = Status(C.miopenFusionPlanConvolutionGetAlgo(f.d, request, &actual, algos[0].cptr())).error("(f *FusionPlanD)ConvolutionGetAlgo()")
	return algos[:actual], err
}
func (f *FusionPlanD) ConvolutionSetAlgo(algo ConvFwdAlgorithm) error {
	return Status(C.miopenFusionPlanConvolutionSetAlgo(f.d, algo.c())).error("(f *FusionPlanD)ConvolutionSetAlgo()")
}
func (f *FusionPlanD)CreateOpConvForward(convD *ConvolutionD, wD *TensorD)(fop *FusionOpD,err error){
	fop=new(FusionOpD)
err=	Status(C.miopenCreateOpConvForward(f.d,&fop.d,convD.d,wD.d)).error("(f *FusionPlanD)CreateOpConvForward()")
return fop,err
}
//FusionDirection is used as flags
//
//Kernel fusion direction in the network
type FusionDirection C.miopenFusionDirection_t

func (f *FusionDirection) cptr() *C.miopenFusionDirection_t { return (*C.miopenFusionDirection_t)(f) }
func (f FusionDirection) c() C.miopenFusionDirection_t      { return (C.miopenFusionDirection_t)(f) }

//Vertical sets f and returns (FusionDirection)(C.miopenVerticalFusion) flag
//
//fuses layers vertically, current the only supported mode
func (f *FusionDirection) Vertical() FusionDirection {
	*f = (FusionDirection)(C.miopenVerticalFusion)
	return *f
}

//Horizontal sets f and returns (FusionDirection)(C.miopenHorizontalFusion) flag
//
//fuses layers horizontally, this is unimplemented
func (f *FusionDirection) Horizontal() FusionDirection {
	*f = (FusionDirection)(C.miopenHorizontalFusion)
	return *f
}
