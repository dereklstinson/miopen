package miopen

//#include <miopen/miopen.h>
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/cutil"
)

//FusionPlanD - MIOpen fusion interface
type FusionPlanD struct {
	d     C.miopenFusionPlanDescriptor_t
	dtype DataType
}

//FusionOpD - Fusion Operator Descriptor contains the meta-data associated with an operator
type FusionOpD struct {
	d     C.miopenFusionOpDescriptor_t
	dtype DataType
}

//CreateFusionPlan - Creates the kernel fusion plan descriptor object
//
//direction		Horizontal or Vertical fusion (input)
//inputD		Descriptor to tensor for the input (input)
func CreateFusionPlan(direction FusionDirection, inputD *TensorD) (fpD *FusionPlanD, err error) {
	fpD = new(FusionPlanD)
	fpD.dtype, _, _, err = inputD.Get()
	if err != nil {
		return nil, errors.New("From CreateFusionPlan :" + err.Error())
	}
	err = Status(C.miopenCreateFusionPlan(&fpD.d, direction.c(), inputD.d)).error("CreateFusionPlan")

	runtime.SetFinalizer(fpD, miopenDestroyFusionPlan)
	return fpD, err
}
func miopenDestroyFusionPlan(f *FusionPlanD) error {
	return Status(C.miopenDestroyFusionPlan(f.d)).error("miopenDestroyFusionPlan")
}

//Compile - Compiles the fusion plan
//
//h		MIOpen handle (input)
func (f *FusionPlanD) Compile(h *Handle) error {
	return Status(C.miopenCompileFusionPlan(h.x, f.d)).error("(f *FusionPlanD) Compile()")
}

//GetOp - Allows access to the operators in a fusion plan
//
//This api call does bounds checking on the supplied op_idx and would
//return err !=nil if the index is out of bounds
//
//	opIdx		 Index of the required operator in the fusion plan, in the order of insertion
func (f *FusionPlanD) GetOp(opIdx int32) (op *FusionOpD, err error) {
	op = new(FusionOpD)
	err = Status(C.miopenFusionPlanGetOp(f.d, (C.int)(opIdx), &op.d)).error("f *FusionPlanD)GetOp()")
	return op, err
}

//GetWorkSpaceSize - Query the workspace size required for the fusion plan
//
//	h		Handle for miopen (input)
//	algo		Convolution forward algorithm (input)
func (f *FusionPlanD) GetWorkSpaceSize(h *Handle, algo ConvFwdAlgorithm) (wspaceSIB uint, err error) {
	var ws C.size_t
	err = Status(C.miopenFusionPlanGetWorkSpaceSize(h.x, f.d, &ws, algo.c())).error("(f *FusionPlanD)GetWorkSpaceSize()")
	wspaceSIB = (uint)(ws)
	return wspaceSIB, err
}

//ConvolutionGetAlgo - Returns the supported algorithms for the convolution operator in the Fusion Plan
//
//A Convolution operator in a fusion plan may be implemented by different algorithms
//representing different tradeoffs of memory and performance. The returned list of algorithms
//is sorted in decreasing order of priority. Therefore, if the user does not request an
//algorithm to be set using the (f *FusionPlanD)ConvolutionSetAlgo() call, the first algorithm
//in the list would be used to execute the convolution in the fusion plan. Moreover this call
//must be immediately preceded by the (f *FusionPlanD)CreateConvForward() call for the op in question.
//
func (f *FusionPlanD) ConvolutionGetAlgo() (algos []ConvFwdAlgorithm, err error) {
	var actual C.int
	request := C.int(4)
	algos = make([]ConvFwdAlgorithm, request)
	err = Status(C.miopenFusionPlanConvolutionGetAlgo(f.d, request, &actual, algos[0].cptr())).error("(f *FusionPlanD)ConvolutionGetAlgo()")
	return algos[:actual], err
}

//ConvolutionSetAlgo - Requests the fusion runtime to choose a particular algorithm for the added convolution operation
//
//Please see the description for (f *FusionPlanD) ConvolutionGetAlgo()
//
//	algo		 Requested algorithm for the convolution operator (input)
func (f *FusionPlanD) ConvolutionSetAlgo(algo ConvFwdAlgorithm) error {
	return Status(C.miopenFusionPlanConvolutionSetAlgo(f.d, algo.c())).error("(f *FusionPlanD)ConvolutionSetAlgo()")
}

//CreateConvForward - Creates forward convolution operator.
//
//	convOp		Pointer to an operator type (output)
//	convDesc		Convolution layer descriptor (input)
//	wDesc		Descriptor for the weights tensor (input)
func (f *FusionPlanD) CreateConvForward(convD *ConvolutionD, wD *TensorD) (convOp *FusionOpD, err error) {
	convOp = new(FusionOpD)
	convOp.dtype = f.dtype
	err = Status(C.miopenCreateOpConvForward(f.d, &convOp.d, convD.d, wD.d)).error("(f *FusionPlanD)CreateOpConvForward()")
	return convOp, err
}

//CreateActivationForward - Creates a forward activation operator.
//
//	activFwdOp		Pointer to an operator type (output)
//	mode		Activation version (input)
func (f *FusionPlanD) CreateActivationForward(mode ActivationMode) (activFwdOp *FusionOpD, err error) {
	activFwdOp = new(FusionOpD)
	activFwdOp.dtype = f.dtype
	err = Status(C.miopenCreateOpActivationForward(f.d, &activFwdOp.d, mode.c())).error("(f *FusionPlanD)CreateActivationForward()")
	return activFwdOp, err
}

//CreateActivationBwd - Creates a backward activation operator.
//
//	activBwdOp		Pointer to an operator type (output)
//	mode		Activation version (input)
func (f *FusionPlanD) CreateActivationBwd(mode ActivationMode) (activBwdOp *FusionOpD, err error) {
	activBwdOp = new(FusionOpD)
	activBwdOp.dtype = f.dtype
	err = Status(C.miopenCreateOpActivationBackward(f.d, &activBwdOp.d, mode.c())).error("(f *FusionPlanD)CreateActivationBwd()")
	return activBwdOp, err
}

//CreateBiasForward - Creates a forward bias operator.
//
//	biasOp		Pointer to an operator type (output)
//	bDesc		bias tensor descriptor (input)
func (f *FusionPlanD) CreateBiasForward(bD *TensorD) (biasOp *FusionOpD, err error) {
	biasOp = new(FusionOpD)
	biasOp.dtype = f.dtype
	err = Status(C.miopenCreateOpBiasForward(f.d, &biasOp.d, bD.d)).error("(f *FusionPlanD)CreateBiasForward()")
	return biasOp, err
}

//CreateBatchNormInference - Creates a forward inference batch normalization operator.
//
//	bnOp		Pointer to an operator type (output)
//	bnD		Batch normalization descriptor (input)
//	scaleBiasMeanVarD		Gamma, beta, mean, variance tensor descriptor (input)
func (f *FusionPlanD) CreateBatchNormInference(bnD *BatchNormD, scaleBiasMeanVarD *TensorD) (bnOp *FusionOpD, err error) {
	bnOp = new(FusionOpD)
	bnOp.dtype = f.dtype
	err = Status(C.miopenCreateOpBatchNormInference(f.d, &bnOp.d, bnD.mode, scaleBiasMeanVarD.d)).error("(f *FusionPlanD)CreateBatchNormInference()")
	return bnOp, err
}

//CreateBatchNormForward - Creates a forward training batch normalization operator.
//
//	bnFwdOp		Pointer to an operator type (output)
//	bnD		Batch normalization descriptor (input)
//	saveMeanVariance		Toggles whether or not to save population statistics for inferencebatch statistic are required (input)
func (f *FusionPlanD) CreateBatchNormForward(bnD *BatchNormD, saveMeanVariance bool) (bnOp *FusionOpD, err error) {
	bnOp = new(FusionOpD)
	bnOp.dtype = f.dtype
	err = Status(C.miopenCreateOpBatchNormForward(f.d, &bnOp.d, bnD.mode, (C.bool)(saveMeanVariance))).error("(f *FusionPlanD)CreateBatchNormForward()")
	return bnOp, err
}

//CreateBatchNormBackward -Creates a back propagation batch normalization operator.
//
//	bnBwdOp		Pointer to an operator type (output)
//	bnD		Batch normalization descriptor (input)
func (f *FusionPlanD) CreateBatchNormBackward(bnD *BatchNormD) (bnOp *FusionOpD, err error) {
	bnOp = new(FusionOpD)
	bnOp.dtype = f.dtype
	err = Status(C.miopenCreateOpBatchNormBackward(f.d, &bnOp.d, bnD.mode)).error("(f *FusionPlanD)CreateBatchNormForward()")
	return bnOp, err
}

//OperatorArgs is an operator argument opbject
type OperatorArgs struct {
	args C.miopenOperatorArgs_t
}

//CreateOperatorArgs - Creates an operator argument object
//
func CreateOperatorArgs() (args *OperatorArgs, err error) {
	args = new(OperatorArgs)
	err = Status(C.miopenCreateOperatorArgs(&args.args)).error("CreateOperatorArgs")
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(args, miopenDestroyOperatorArgs)
	return args, err
}

func miopenDestroyOperatorArgs(args *OperatorArgs) error {
	return Status(C.miopenDestroyOperatorArgs(args.args)).error("miopenDestroyOperatorArgs")
}

//SetConvForward - Sets the arguments for forward convolution op
//
//	convOp		Forward convolution operator (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	w		Pointer to tensor memory  (input)
func (o *OperatorArgs) SetConvForward(convOp *FusionOpD, alpha, beta float64, w cutil.Mem) error {

	a1 := cscalarbydatatype(convOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(convOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsConvForward(o.args, convOp.d, a1, b1, w.Ptr())).error("(o *OperatorArgs) SetArgsConvForward()")
}

//SetActivForward - Sets the arguments for forward activation op
//
//	activFwdOp		Activation backwards operator (input)
//	alpha		Floating point scaling factor, allocated on the host (input)
//	beta		Floating point shift factor, allocated on the host (input)
//	activAlpha		Double precision activation parameter which depends on activation mode (input)
//	activBeta		Double precision activation parameter which depends on activation mode (input)
//	activGamma		Double precision activation parameter which depends on activation mode (input)
func (o *OperatorArgs) SetActivForward(activFwdOp *FusionOpD, alpha, beta, activeAlpha, activBeta, activGamma float64) error {
	a1 := cscalarbydatatype(activFwdOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(activFwdOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsActivForward(o.args, activFwdOp.d, a1, b1, (C.double)(activeAlpha), (C.double)(activBeta), (C.double)(activGamma))).error("(o *OperatorArgs)SetActivForward()")
}

//SetActivBackward - Sets the arguments for backward activation op
//
//	activBwdOp   Activation backwards operator (input)
//	alpha   Floating point scaling factor, allocated on the host (input)
//	beta    Floating point shift factor, allocated on the host (input)
//	y        Data tensor y, output of activations in the forward direction (input)
//	reserved    Data tensor reserved memory space; currently should be nullptr (input)
//	activAlpha  Double precision activation parameter which depends on activation mode (input)
//	activBeta   Double precision activation parameter which depends on activation mode (input)
//	activGamma  Double precision activation parameter which depends on activation mode (input)
func (o *OperatorArgs) SetActivBackward(activBwdOp *FusionOpD, alpha, beta float64, y, reserved cutil.Mem, activeAlpha, activBeta, activGamma float64) error {
	a1 := cscalarbydatatype(activBwdOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(activBwdOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsActivBackward(o.args, activBwdOp.d, a1, b1, y.Ptr(), nil, (C.double)(activeAlpha), (C.double)(activBeta), (C.double)(activGamma))).error("(o *OperatorArgs)SetActivForward()")
}

//SetBatchNormInference - Sets the arguments for inference batch normalization op
//
//	bnOp               Batch normalization inference operator (input)
//	alpha              Floating point scaling factor, allocated on the host (input)
//	beta               Floating point shift factor, allocated on the host (input)
//	bnScale            Pointer to the gamma tensor memory  (input)
//	bnBias             Pointer to the beta tensor memory  (input)
//	estimatedMean      Pointer to population mean memory  (input)
//	estimatedVariance  Pointer to population variance memory  (input)
//	epsilon            Scalar value for numerical stability (input)
func (o *OperatorArgs) SetBatchNormInference(bnOp *FusionOpD, alpha, beta float64, scale, bias, estimatedMean, estimatedVariance cutil.Mem, epsilon float64) error {
	a1 := cscalarbydatatype(bnOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(bnOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsBatchNormInference(o.args, bnOp.d, a1, b1, scale.Ptr(), bias.Ptr(), estimatedMean.Ptr(), estimatedVariance.Ptr(), (C.double)(epsilon))).error("(o *OperatorArgs)SetBatchNormInference()")
}

//SetBatchNormForward - Sets the arguments for forward batch normalization op
//
//	bnOp               Batch normalization forward operator (input)
//	alpha              Floating point scaling factor, allocated on the host (input)
//	beta               Floating point shift factor, allocated on the host (input)
//	bnScale            Pointer to the gamma tensor memory  (input)
//	bnBias             Pointer to the beta tensor memory  (input)
//	savedMean          Pointer to batch mean memory  (input)
//	savedInvVariance   Pointer to batch inverse variance memory  (input)
//	runningMean        Pointer to population mean memory  (input)
//	runningVariance    Pointer to population variance memory  (input)
//	expAvgFactor       Scalar value for control of population statistics (input)
//	epsilon            Scalar value for numerical stability (input)
func (o *OperatorArgs) SetBatchNormForward(bnOp *FusionOpD, alpha, beta float64,
	scale, bias, savedMean, savedVariance, runningMean, runningVariance cutil.Mem,
	expAvgFactor, epsilon float64) error {
	a1 := cscalarbydatatype(bnOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(bnOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsBatchNormForward(o.args, bnOp.d, a1, b1,
		scale.Ptr(), bias.Ptr(), savedMean.Ptr(), savedVariance.Ptr(), runningMean.Ptr(), runningVariance.Ptr(),
		(C.double)(expAvgFactor), (C.double)(epsilon))).error("(o *OperatorArgs)SetBatchNormForward()")
}

//SetBatchNormBackward - Sets the arguments for backward batch normalization op
//
//	bnOp               Batch normalization forward operator (input)
//	alpha              Floating point scaling factor, allocated on the host (input)
//	beta               Floating point shift factor, allocated on the host (input)
//	x                  Pointer to the forward input tensor memory  (input)
//	bnScale            Pointer to the gamma tensor memory  (input)
//	bnBias             Pointer to the beta tensor memory  (input)
//	resultBnScaleDiff  Pointer to the gamma gradient tensor memory  (output)
//	resultBnBiasDiff   Pointer to the beta gradient tensor memory  (output)
//	savedMean          Pointer to batch mean memory  (input)
//	savedInvVariance   Pointer to batch inverse variance memory  (input)
func (o *OperatorArgs) SetBatchNormBackward(bnOp *FusionOpD, alpha, beta float64,
	x, scale, bias, resultScale, resultBias, savedMean, savedVariance cutil.Mem) error {
	a1 := cscalarbydatatype(bnOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(bnOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsBatchNormBackward(o.args, bnOp.d, a1, b1,
		x.Ptr(), scale.Ptr(), bias.Ptr(), resultScale.Ptr(), resultBias.Ptr(), savedMean.Ptr(), savedVariance.Ptr())).error("(o *OperatorArgs)SetBatchNormBackward()")
}

//SetBiasForward - Sets the arguments for forward bias op
//
//	biasOp         Forward bias operator (input)
//	alpha          Floating point scaling factor, allocated on the host (input)
//	beta           Floating point shift factor, allocated on the host (input)
//	bias           Pointer to the forward bias input tensor memory  (input)
func (o *OperatorArgs) SetBiasForward(biasOp *FusionOpD, alpha, beta float64, bias cutil.Mem) error {
	a1 := cscalarbydatatype(biasOp.dtype, alpha).CPtr()
	b1 := cscalarbydatatype(biasOp.dtype, beta).CPtr()
	return Status(C.miopenSetOpArgsBiasForward(o.args, biasOp.d, a1, b1, bias.Ptr())).error("(o *OperatorArgs)SetBiasForward()")

}

//Execute - Executes the fusion plan
//
//	h		MIOpen handle (input)
//	inputD		Descriptor of the input tensor (input)
//	input		Source data tensor  (input)
//	outputD		Decriptor of the output tensor (input)
//	output		Destination data tensor  (output)
//	args		An argument object of the fused kernel (input)
func (f *FusionPlanD) Execute(h *Handle,
	inputD *TensorD, input cutil.Mem,
	outputD *TensorD, output cutil.Mem, args *OperatorArgs) error {
	return Status(C.miopenExecuteFusionPlan(h.x, f.d, inputD.d, input.Ptr(), outputD.d, output.Ptr(), args.args)).error("(f *FusionPlanD)Execute()")
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
