package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import (
	"errors"
	"runtime"

	"github.com/dereklstinson/cutil"
)

//RNNMode is used for flags for the RNNMode. Flags are set through its methods
//
//RNN mode selection for rnn layer preference
type RNNMode C.miopenRNNMode_t

func (r RNNMode) c() C.miopenRNNMode_t      { return (C.miopenRNNMode_t)(r) }
func (r *RNNMode) cptr() *C.miopenRNNMode_t { return (*C.miopenRNNMode_t)(r) }

//RELU - RNN with ReLU activation
func (r *RNNMode) RELU() RNNMode { *r = (RNNMode)(C.miopenRNNRELU); return *r }

//Tanh -  RNN with tanh activation
func (r *RNNMode) Tanh() RNNMode { *r = (RNNMode)(C.miopenRNNTANH); return *r }

//LSTM - set to LSTM
func (r *RNNMode) LSTM() RNNMode { *r = (RNNMode)(C.miopenLSTM); return *r }

//GRU - Set to GRU
func (r *RNNMode) GRU() RNNMode { *r = (RNNMode)(C.miopenGRU); return *r }

//RNNInputMode is used for flags for the RNNInputMode. Flags are set through its methods
//
//Recurrent Neural Network layer initial input mode
type RNNInputMode C.miopenRNNInputMode_t

func (r RNNInputMode) c() C.miopenRNNInputMode_t      { return (C.miopenRNNInputMode_t)(r) }
func (r *RNNInputMode) cptr() *C.miopenRNNInputMode_t { return (*C.miopenRNNInputMode_t)(r) }

//Linear - Matrix multiplication at the input of the first layer
func (r *RNNInputMode) Linear() RNNInputMode { *r = (RNNInputMode)(C.miopenRNNlinear); return *r }

//Skip - No operation is performed at the input of the first layer.
func (r *RNNInputMode) Skip() RNNInputMode { *r = (RNNInputMode)(C.miopenRNNskip); return *r }

//RNNAlgo is used for flags for the RNNAlgo. Flags are set through its methods
//
// Recurrent Neural Network algorithm mode
type RNNAlgo C.miopenRNNAlgo_t

func (r RNNAlgo) c() C.miopenRNNAlgo_t      { return (C.miopenRNNAlgo_t)(r) }
func (r *RNNAlgo) cptr() *C.miopenRNNAlgo_t { return (*C.miopenRNNAlgo_t)(r) }

//Default is default.
func (r *RNNAlgo) Default() RNNAlgo { *r = (RNNAlgo)(C.miopenRNNdefault); return *r }

//RNNDirectionMode - Recurrent Neural Network bi-directional behavior
type RNNDirectionMode C.miopenRNNDirectionMode_t

func (r RNNDirectionMode) c() C.miopenRNNDirectionMode_t      { return (C.miopenRNNDirectionMode_t)(r) }
func (r *RNNDirectionMode) cptr() *C.miopenRNNDirectionMode_t { return (*C.miopenRNNDirectionMode_t)(r) }

//UNI -  Forward in time only
func (r *RNNDirectionMode) UNI() RNNDirectionMode {
	*r = (RNNDirectionMode)(C.miopenRNNunidirection)
	return *r
}

//BI - Forward and backwards in time
func (r *RNNDirectionMode) BI() RNNDirectionMode {
	*r = (RNNDirectionMode)(C.miopenRNNbidirection)
	return *r
}

//RNNBiasMode -  Recurrent Neural Network add on bias
type RNNBiasMode C.miopenRNNBiasMode_t

func (r RNNBiasMode) c() C.miopenRNNBiasMode_t      { return (C.miopenRNNBiasMode_t)(r) }
func (r *RNNBiasMode) cptr() *C.miopenRNNBiasMode_t { return (*C.miopenRNNBiasMode_t)(r) }

//NoBias - No Biases will be applied to GEMM operations
func (r *RNNBiasMode) NoBias() RNNBiasMode { *r = (RNNBiasMode)(C.miopenRNNNoBias); return *r }

//WithBias - Biases will be applied to GEMM operations
func (r *RNNBiasMode) WithBias() RNNBiasMode { *r = (RNNBiasMode)(C.miopenRNNwithBias); return *r }

//RNNGEMMalgoMode - RNN algo mode
type RNNGEMMalgoMode C.miopenRNNGEMMalgoMode_t

func (r RNNGEMMalgoMode) c() C.miopenRNNGEMMalgoMode_t      { return (C.miopenRNNGEMMalgoMode_t)(r) }
func (r *RNNGEMMalgoMode) cptr() *C.miopenRNNGEMMalgoMode_t { return (*C.miopenRNNGEMMalgoMode_t)(r) }

//AlgoGEMM selects algo to GEMM
func (r *RNNGEMMalgoMode) AlgoGEMM() RNNGEMMalgoMode {
	*r = (RNNGEMMalgoMode)(C.miopenRNNAlgoGEMM)
	return *r
}

//CreateRNNDescriptor - Create a RNN layer Descriptor
func CreateRNNDescriptor() (rnnD *RNND, err error) {
	rnnD = new(RNND)
	err = Status(C.miopenCreateRNNDescriptor(&rnnD.d)).error("CreateRNNDescriptor")
	runtime.SetFinalizer(rnnD, miopenDestroyRNNDescriptor)
	return rnnD, err
}
func miopenDestroyRNNDescriptor(r *RNND) error {
	return Status(C.miopenDestroyRNNDescriptor(r.d)).error("miopenDestroyRNNDescriptor")
}

//Set - Set the details of the RNN descriptor
//
//Interface for setting the values of the RNN descriptor object. This function requires specific
//algorithm selection.
//
//hsize        Hidden layer size (input)
//nlayers      Number of layers (input)
//inMode       RNN first layer input mode (input)
//direction    RNN direction (input)
//mode      RNN model type (input)
//biasmode     RNN bias included (input)
//algo         RNN algorithm selected (input)
//dtype     Only fp32 currently supported for RNNs (input)
func (r *RNND) Set(hsize, nlayers int32,
	inMode RNNInputMode,
	direction RNNDirectionMode,
	mode RNNMode,
	biasmode RNNBiasMode,
	algo RNNAlgo,
	dtype DataType) error {
	return Status(C.miopenSetRNNDescriptor(r.d, (C.int)(hsize), (C.int)(nlayers), inMode.c(), direction.c(), mode.c(), biasmode.c(), algo.c(), dtype.c())).error("(r *RNND) Set()")
}

//Get - Retrieves a RNN layer descriptor's details
//
func (r *RNND) Get() (hsize, nlayers int32,
	inMode RNNInputMode,
	direction RNNDirectionMode,
	mode RNNMode,
	biasmode RNNBiasMode,
	algo RNNAlgo,
	err error) {
	err = Status(C.miopenGetRNNDescriptor(r.d,
		mode.cptr(),
		algo.cptr(),
		inMode.cptr(),
		direction.cptr(),
		biasmode.cptr(),
		(*C.int)(&hsize),
		(*C.int)(&nlayers))).error("(r *RNND) Get()")
	return hsize, nlayers, inMode, direction, mode, biasmode, algo, err
}

//GetWorkspaceSize - Query the amount of memory required to execute the RNN layer
//
//This function calculates the amount of memory required to run the RNN layer given an RNN
//descriptor and a tensor descriptor.
//
//	h				MIOpen handle (input)
//
//	xD				An array of tensor descriptors. These are the
//					input descriptors to each time step. The first dimension of each descriptor is the
//					size and may decrease from element n to element n+1 and not increase in size.
//					second dimension is the same for all descriptors in the array and is the input
//					length. (input)
//
//	wspacesib        Number of bytes required for RNN layer execution (output)
func (r *RNND) GetWorkspaceSize(h *Handle, xD []*TensorD) (wspacesib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := tensorDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNWorkspaceSize(h.x, r.d, seqLen, &xDs[0], &sizet)).error("GetWorkspaceSize")
	wspacesib = (uint)(sizet)
	return wspacesib, err
}

//GetTrainingReserveSize - Query the amount of memory required for RNN training
//
//This function calculates the amount of memory required to train the RNN layer given an
//RNN descriptor and a tensor descriptor.
//
//	h          		MIOpen handle (input)
//
//	xD      		An array of tensor descriptors. These are the
//					input descriptors to each time step. The first dimension of each descriptor is the
//					batch size and may decrease from element n to element n+1 and not increase in size.
//					The second dimension is the same for all descriptors in the array and is the input
//					vector length. (input)
//
//	reservesib		Number of bytes required for RNN layer execution (output)
func (r *RNND) GetTrainingReserveSize(h *Handle, xD []*TensorD) (reservesib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := tensorDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNTrainingReserveSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("GetTrainingReserveSize")
	reservesib = (uint)(sizet)
	return reservesib, err
}

//GetParamSize - Query the amount of parameter memory required for RNN training
//
//This function calculates the amount of parameter memory required to train the RNN layer given an
//RNN descriptor and a tensor descriptor.
//
//	h			MIOpen handle (input)
//	xD			A tensor descriptor (input)
//	dtype		MIOpen data type enum (input)
func (r *RNND) GetParamSize(h *Handle, xD *TensorD, dtype DataType) (paramSIB uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNParamsSize(h.x, r.d, xD.d, &sizet, dtype.c())).error("GetTrainingReserveSize")
	paramSIB = (uint)(sizet)
	return paramSIB, err
}

//GetRNNDParamDescriptor - Obtain a weight tensor descriptor for RNNs
//
//This function populates a weight descriptor that describes the memory layout of the
//weight matrix.
//
//	h			MIOpen handle (input)
//	xD			A previously populated tensor descriptor (input)
//	dtype		MIOpen data type enum, currently only fp32 is supported (input)
func (r *RNND) GetRNNDParamDescriptor(h *Handle, xD *TensorD, dtype DataType) (wD *TensorD, err error) {
	wD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	err = Status(C.miopenGetRNNParamsDescriptor(h.x, r.d, xD.d, wD.d, dtype.c())).error("(r *RNND)GetRNNDParamDescriptor()")
	return wD, err
}

//GetInputTensorSize - Obtain a the size in bytes of the RNN input tensor
//
//This function determines the size in bytes of the allocation needed for the input data
//tensor for an RNN layer. The number of bytes is derived from the array of
//tensor descriptors.
//
//	h		MIOpen handle (input)
//
//	xD		An array of tensor descriptors. These are the
//			input descriptors to each time step. The first dimension of each descriptor is the
//			batch size and may decrease from element n to element n+1 and not increase in size.
//			The second dimension is the same for all descriptors in the array and is the input
//			vector length. (input)
func (r *RNND) GetInputTensorSize(h *Handle, xD []*TensorD) (sib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := tensorDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNInputTensorSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("(r *RNND)miopenGetRNNInputTensorSize()")
	sib = (uint)(sizet)
	return sib, err
}

//GetHiddenTensorSize - Obtain a the size in bytes of the RNN hidden tensor
//
//This function determines the size in bytes of the allocation needed for the
//hidden tensor over all layers
//
//	handle		MIOpen handle (input)
//	xD			An array of previously populated tensor descriptors (input)
func (r *RNND) GetHiddenTensorSize(h *Handle, xD []*TensorD) (sib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := tensorDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNHiddenTensorSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("(r *RNND)GetHiddenTensorSize()")
	sib = (uint)(sizet)
	return sib, err
}

//GetLayerParamSize - Gets the number of bytes of a parameter matrix
//
//For RNN vanilla ELU and TANH, paramID == 0 retrieves the
//weight matrix associated with the in input GEMM, while paramID == 1 retrieves
//the weight matrix associated with the hidden state GEMM.
//
//For LSTM paramID 0 to 3 refer to the weight matrices associated
//with the input GEMM, 4-7 are associated with matrices associated with the
//hidden state GEMM.
//	paramID 0 and 4 are for the input gate.
//	paramID 1 and 5 are for the forget gate.
//	paramID 2 and 6 are for the output gate.
//	paramID 3 and 7 are for the new memory gate.
//
//For GRU paramID 0 to 2 refer to the weight matrix offset associated
//with the input GEMM, while 3 through 5 are associated with the hidden state
//GEMM.
//	paramID 0 and 3 are for the update gate.
//	paramID 1 and 4 are for the reset gate.
//	paramID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//	h		MIOpen handle (input)
//	layer		layer number in the RNN stack (input)
//	xDesc		tensor descriptor to input (input)
//	paramID		ID of the internal parameter tensor (input)
func (r *RNND) GetLayerParamSize(h *Handle, layer int32, xD *TensorD, paramID int32) (sib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNLayerParamSize(h.x, r.d, (C.int)(layer), xD.d, (C.int)(paramID), &sizet)).error("(r *RNND)GetLayerParamSize()")
	sib = (uint)(sizet)
	return sib, err
}

//GetLayerBiasSize - Gets the number of bytes of a bias
//
//For RNN vanilla ELU and TANH, biasID == 0 retrieves the
//weight matrix associated with the in input GEMM, while biasID == 1 retrieves
//the weight matrix associated with the hidden state GEMM.
//
//For LSTM biasID 0 to 3 refer to the weight matrices associated
//with the input GEMM, 4-7 are associated with matrices associated with the
//hidden state GEMM.
//	biasID 0 and 4 are for the input gate.
//	biasID 1 and 5 are for the forget gate.
//	biasID 2 and 6 are for the output gate.
//	biasID 3 and 7 are for the new memory gate.
//
//For GRU biasID 0 to 2 refer to the weight matrix offset associated
//with the input GEMM, while 3 through 5 are associated with the hidden state
//GEMM.
//	biasID 0 and 3 are for the update gate.
//	biasID 1 and 4 are for the reset gate.
//	biasID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//	h		MIOpen handle (input)
//	layer		layer number in the RNN stack (input)
//	biasID		ID of the internal parameter tensor (input)
func (r *RNND) GetLayerBiasSize(h *Handle, layer, biasID int32) (sib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNLayerBiasSize(h.x, r.d, (C.int)(layer), (C.int)(biasID), &sizet)).error("(r *RNND)GetLayerBiasSize()")
	sib = (uint)(sizet)
	return sib, err
}

//GetLayerParam - Gets a weight matrix for a specific layer in an RNN stack
//
//This function retrieves the weight matrix data for a specific layer and parameter ID
//and copies the data into previously allocated device memory.
//
//For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
//weight matrix associated with the in input GEMM, while paramID == 1 retrieves
//the weight matrix associated with the hidden state GEMM.
//
//For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
//with the input GEMM, 4-7 are associated with matrices associated with the
//hidden state GEMM.
//	paramID 0 and 4 are for the input gate.
//	paramID 1 and 5 are for the forget gate.
//	paramID 2 and 6 are for the output gate.
//	paramID 3 and 7 are for the new memory gate.
//
//For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
//with the input GEMM, while 3 through 5 are associated with the hidden state
//GEMM.
//	paramID 0 and 3 are for the update gate.
//	paramID 1 and 4 are for the reset gate.
//	paramID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The output argument paramDesc is a previously created tensor descriptor that is populated
//to describe the memory layout of the parameter matrix. It is full packed and is used when
//calling to (r *RNND) SetLayerParam()
//
//The argument layerParam should either be nullptr, or have device memory allocated
//to allow copying of the entire layer parameter matrix into it. If layerParam is
//nullptr then only the paramDesc is populated and returned. The size in bytes of the
//layer parameter matrix can be determined by using (r *RNND) GetLayerParamSize(().
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case (r *RNND) GetLayerParam() will return
//a error status miopenStatusBadParm for input paramID associated with the input GEMM.
//
//	h			MIOpen handle (input)
//	layer		The layer number in the RNN stack (input)
//	xD			A tensor descriptor to input (input)
//	wD			A tensor descriptor to the parameter tensor (input)
//	w			Pointer to memory containing parameter tensor (input)
//	paramID		ID of the internal parameter tensor (input)
func (r *RNND) GetLayerParam(h *Handle, layer int32, xD, wD *TensorD, w cutil.Mem, paramID int32) (paramD *TensorD, param cutil.Mem, err error) {
	paramD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, nil, err
	}
	param = new(mem)

	err = Status(C.miopenGetRNNLayerParam(h.x,
		r.d,
		(C.int)(layer),
		xD.d,
		wD.d,
		w.Ptr(),
		(C.int)(paramID),
		paramD.d,
		param.Ptr())).error("(r *RNND)GetLayerParam()")
	return paramD, param, err
}

//GetLayerBias - Gets a bias for a specific layer in an RNN stack
//
//This function retrieves the bias data for a specific layer and bias ID and copies
//the data into previously allocated device memory.
//
//For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
//bias associated with the in input GEMM, while biasID == 1 retrieves
//the bias associated with the hidden state GEMM.
//
//For miopenLSTM biasID 0 to 3 refer to the biases associated
//with the input GEMM, 4-7 are associated with biases associated with the
//hidden state GEMM.
//	biasID 0 and 4 are for the input gate.
//	biasID 1 and 5 are for the forget gate.
//	biasID 2 and 6 are for the output gate.
//	biasID 3 and 7 are for the new memory gate.
//
//For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
//while 3 through 5 are associated with the hidden state GEMM.
//	biasID 0 and 3 are for the update gate.
//	biasID 1 and 4 are for the reset gate.
//	biasID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The output argument biasDesc is a previously created tensor descriptor that is populated
//to describe the memory layout of the bias. It is full packed and is used when
//calling to (r *RNND) SetLayerBias()
//
//The argument layerBias should either be nullptr, or have device memory allocated
//to allow copying of the entire layer bias into it. If layerBias is
//nullptr then only the biasDesc is populated and returned. The size in bytes of the
//layer bias can be determined by using (r *RNND) GetLayerBiasSize().
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case  (r *RNND) GetLayerBias() will return
//a error status miopenStatusBadParm for input biasID associated with the input GEMM.
//
//	h			MIOpen handle (input)
//	layer		The layer number in the RNN stack (input)
//	xD			A tensor descriptor to input (input)
//	wD			A tensor descriptor to the parameter tensor (input)
//	w			Pointer to memory containing parameter tensor (input)
//	biasID		ID of the internal parameter tensor (input)
func (r *RNND) GetLayerBias(h *Handle, layer int32, xD, wD *TensorD, w cutil.Mem, biasID int32) (biasD *TensorD, bias cutil.Mem, err error) {
	biasD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, nil, err
	}
	bias = new(mem)
	err = Status(C.miopenGetRNNLayerBias(h.x,
		r.d,
		(C.int)(layer),
		xD.d,
		wD.d,
		w.Ptr(),
		(C.int)(biasID),
		biasD.d,
		bias.Ptr())).error("(r *RNND)GetLayerBias()")
	return biasD, bias, err
}

//GetLayerParamOffset -Gets an index offset for a specific weight matrix for a layer in the RNN stack
//
//This function retrieves the index offset for a weight matrix in a layer.
//
//For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
//weight matrix offset associated with the in input GEMM, while paramID == 1
//retrieves the weight matrix offset associated with the hidden state GEMM.
//
//For miopenLSTM paramID 0 to 3 refer to the weight matrix offsets associated
//with the input GEMM, 4-7 are associated with matrix offset associated with the
//hidden state GEMM.
//	paramID 0 and 4 are for the input gate.
//	paramID 1 and 5 are for the forget gate.
//	paramID 2 and 6 are for the output gate.
//	paramID 3 and 7 are for the new memory gate.
//
//For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
//with the input GEMM, while 3 through 5 are associated with the hidden state
//GEMM.
//	paramID 0 and 3 are for the update gate.
//	paramID 1 and 4 are for the reset gate.
//	paramID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The output argument paramDesc is a previously created tensor descriptor that is populated
//to describe the memory layout of the parameter matrix. It is full packed and is used when
//calling to (r *RNND) SetLayerParam().
//
//The argument layerParamOffset should either be nullptr, or an address to place the
//offset. If layerParamOffset is nullptr then only the paramDesc is populated and returned.
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case (r *RNND)GetLayerParamOffset() will return
//a error status miopenStatusBadParm for input paramID associated with the input GEMM.
//
//	layer		The layer number in the RNN stack (input)
//	xD			A tensor descriptor to input (input)
//	paramID		ID of the internal parameter tensor (input)
func (r *RNND) GetLayerParamOffset(layer int32, xD *TensorD, paramID int32) (paramD *TensorD, offset uint, err error) {
	paramD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, 0, err
	}
	var coffset C.size_t
	err = Status(C.miopenGetRNNLayerParamOffset(
		r.d,
		(C.int)(layer),
		xD.d,
		(C.int)(paramID),
		paramD.d,
		&coffset)).error("(r *RNND)GetLayerParamOffset()")
	offset = (uint)(coffset)
	return paramD, offset, err
}

//GetLayerBiasOffset - Gets a bias index offset for a specific layer in an RNN stack
//
//This function retrieves the bias index offset for a specific layer and bias ID.
//
//For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
//bias associated with the in input GEMM, while biasID == 1 retrieves
//the weight matrix associated with the hidden state GEMM.
//
//For miopenLSTM biasID 0 to 3 refer to the bias offset associated
//with the input GEMM, 4-7 are the bias offsets associated with the hidden state GEMM.
//	biasID 0 and 4 are for the input gate.
//	biasID 1 and 5 are for the forget gate.
//	biasID 2 and 6 are for the output gate.
//	biasID 3 and 7 are for the new memory gate.
//
//For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
//while 3 through 5 are associated with the hidden state GEMM.
//	biasID 0 and 3 are for the update gate.
//	biasID 1 and 4 are for the reset gate.
//	biasID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The output argument biasDesc is a previously created tensor descriptor that is populated
//to describe the memory layout of the bias. It is full packed and is used when
//calling to (r *RNND) SetLayerBias()
//
//The argument layerBiasOffset should either be nullptr, or point to an output address.
//If layerBias is nullptr then only the biasDesc is populated and returned.
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case miopenGetRNNLayerBiasOffset() will return
//a error status miopenStatusBadParm for input biasID associated with the input GEMM.
//
//	layer           The layer number in the RNN stack (input)
//	xD           A tensor descriptor to input (input)
//	biasID          ID of the internal parameter tensor (input)
func (r *RNND) GetLayerBiasOffset(layer int32, xD *TensorD, paramID int32) (biasD *TensorD, offset uint, err error) {
	biasD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, 0, err
	}
	var coffset C.size_t
	err = Status(C.miopenGetRNNLayerBiasOffset(
		r.d,
		(C.int)(layer),
		xD.d,
		(C.int)(paramID),
		biasD.d,
		&coffset)).error("(r *RNND)GetLayerParamOffset()")
	offset = (uint)(coffset)
	return biasD, offset, err
}

//SetLayerParam - Sets a weight matrix for a specific layer in an RNN stack
//
// This function sets the weight matrix data for a specific layer and parameter ID.
//
// For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 sets the
// weight matrix associated with the in input GEMM, while paramID == 1 sets
// the weight matrix associated with the hidden state GEMM.
//
// For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
// with the input GEMM, 4-7 are associated with matrices associated with the
// hidden state GEMM.
//	paramID 0 and 4 are for the input gate.
//	paramID 1 and 5 are for the forget gate.
//	paramID 2 and 6 are for the output gate.
//	paramID 3 and 7 are for the new memory gate.
//
// For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
// with the input GEMM, while 3 through 5 are associated with the hidden state
// GEMM.
//	paramID 0 and 3 are for the update gate.
//	paramID 1 and 4 are for the reset gate.
//	paramID 2 and 5 are for the new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The input argument paramDesc is a previously populated tensor descriptor typically
//by first calling (r *RNND) GetLayerParam().
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case (r *RNND) SetLayerParam() will return
//a error status miopenStatusBadParm for input paramID associated with the input GEMM.
//
//	h				MIOpen handle (input)
//	layer			The layer number in the RNN stack (input)
//	xD				A tensor descriptor to input (input)
//	wD				A tensor descriptor to the parameter tensor (input)
//	w				Pointer to memory containing parameter tensor (input)
//	paramID			ID of the internal parameter tensor (input)
//	paramD			Descriptor of the parameter tensor (input)
//	layerParam		Pointer to the memory location of the parameter tensor (input)
func (r *RNND) SetLayerParam(h *Handle, layer int32, xD,
	wD *TensorD, w cutil.Mem,
	paramID int32, paramD *TensorD, layerParam cutil.Mem) error {
	return Status(C.miopenSetRNNLayerParam(h.x, r.d, (C.int)(layer), xD.d, wD.d, w.Ptr(), (C.int)(paramID), paramD.d, layerParam.Ptr())).error("SetLayerParam")
}

//SetLayerBias - Sets a bias for a specific layer in an RNN stack
//
//This function sets the bias data for a specific layer and bias ID.
//
//For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
//weight matrix associated with the in input GEMM, while biasID == 1 retrieves
//the bias associated with the hidden state GEMM.
//
// For miopenLSTM biasID 0 to 3 refer to the biases associated
// with the input GEMM, 4-7 are associated with the biases associated with the
// hidden state GEMM.
//	biasID 0 and 4 are for the input gate.
//	biasID 1 and 5 are for the forget gate.
//	biasID 2 and 6 are for the output gate.
//	biasID 3 and 7 are for the new memory gate.
//
// For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
// while 3 through 5 are associated with the hidden state GEMM.
//	biasID 0 and 3 are for the update gate.
//	biasID 1 and 4 are for the reset gate.
//	biasID 2 and 5 are for the new new memory gate.
//
//For bi-directional RNNs the backwards in time direction is numbered as the layer
//directly after the forward in time direction.
//
//The input argument biasDesc is a previously populated tensor descriptor typically
//by first calling miopenGetRNNLayeBias().
//
//Note: When inputSkip mode is selected there is no input layer matrix operation,
//and therefore no associated memory. In this case (r *RNND) SetLayerBias will return
//a error status miopenStatusBadParm for input biasID associated with the input GEMM.
//
//	h			MIOpen handle (input)
//	layer		The layer number in the RNN stack (input)
//	xD			A tensor descriptor to input (input)
//	wD			A tensor descriptor to the bias tensor (input)
//	w			Pointer to memory containing bias tensor (input)
//	biasID		ID of the internal bias tensor (input)
//	biasD		Descriptor of the bias tensor (output)
//	layerBias	Pointer to the memory location of the bias tensor (output)
func (r *RNND) SetLayerBias(h *Handle, layer int32, xD,
	wD *TensorD, w cutil.Mem,
	biasID int32, biasD *TensorD, layerBias cutil.Mem) error {
	return Status(C.miopenSetRNNLayerBias(h.x, r.d, (C.int)(layer), xD.d, wD.d, w.Ptr(), (C.int)(biasID), biasD.d, layerBias.Ptr())).error("SetLayerParam")
}

//ForwardTraining - Execute forward training for recurrent layer
//
//Interface for executing the forward training pass on a RNN.
//
//	h           	MIOpen handle (input)
//
//	xD          	An array of tensor descriptors. These are the
//		 	descriptors to each time step. The first dimension of each descriptor is the
//			batch size and may decrease from element n to element n+1 and not increase in size.
//			The second dimension is the same for all descriptors in the array and is the input
//			vector length. (input)
//
//	x		Pointer to input tensor (input)
//
//	hxD		A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hx          	Pointer to the hidden layer input tensor. If hx is NULL,
//			then the initial hidden state will be zero initialized. (input)
//
//	cxD         	A cell tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	cx          	Pointer to the cell layer input tensor. If cx is NULL,
//			then the initial cell state will be zero initialized. (input)
//
//	wD          	A weights tensor descriptor (input)
//
//	w           	Pointer to input weights tensor (input)
//
//	yD          	An array of fully packed tensor descriptors associated
//			with the output from each time step. The first dimension of the tensor descriptors
//			must equal the first dimension of the first descriptor (batch size) in the xDesc
//			tensor array. The second dimension of the element of the descriptor array
//			depends on the direction mode selected. If the direction mode is unidirectional,
//			the second dimension is the hiddenSize. If direction mode is bidirectional
//			the second dimension is twice the hiddenSize. (input)
//
//	y           	Pointer to output tensor (output)
//
//	hyD         	A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hy          	Pointer to the hidden layer output tensor. If hy is NULL,
//			then the final hidden state will not be saved. (output)
//
//	cyD         	A cell tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	cy		Pointer to the cell layer output tensor. If hy is NULL,
//			then the final cell state will not be saved. (output)
//
//	wspace      	Pointer to memory allocated for forward training (input)
//
//	wspaceSIB   	Number of allocated bytes in memory for the workspace (input)
//
//	rspace      	Pointer to memory allocated for random states (input / output)
//
//	rspaceSIB	Number of allocated bytes in memory for use in the forward  (input)
func (r *RNND) ForwardTraining(h *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *TensorD, w cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	hyD *TensorD, hy cutil.Mem,
	cyD *TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
	rspace cutil.Mem, rspaceSIB uint) error {

	xDc, seqenceLen1 := tensorDarraytomiopenTensorDescriptorArray(xD)
	yDc, seqenceLen2 := tensorDarraytomiopenTensorDescriptorArray(yD)
	if seqenceLen1 != seqenceLen2 {
		return errors.New("(r *RNND)ForwardTraining(): len(xD)!=len(yD)")
	}

	return Status(C.miopenRNNForwardTraining(h.x, r.d,
		seqenceLen1, &xDc[0], x.Ptr(),
		hxD.d, hx.Ptr(),
		cxD.d, cx.Ptr(),
		wD.d, w.Ptr(),
		&yDc[0], y.Ptr(),
		hyD.d, hy.Ptr(),
		cyD.d, cy.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		rspace.Ptr(), (C.size_t)(rspaceSIB))).error("(r *RNND)ForwardTraining()")
}

//BackwardData - Execute forward training for recurrent layer
//
//Interface for executing the forward training pass on a RNN.
//
//
//	h		MIOpen handle (input)
//
//	rnnD		descriptor type (input)
//
//	yD		An array of tensor descriptors (input)
//
//	y		Pointer to input tensor (input)
//
//	dyD		An array of fully packed tensor descriptors associated
//			with the output from each time step. The first dimension of the tensor descriptors
//			must equal the first dimension of the first descriptor (batch size) in the xDesc
//			tensor array. The second dimension of the element of the descriptor array
//			depends on the direction mode selected. If the direction mode is unidirectional,
//			the second dimension is the hiddenSize. If direction mode is bidirectional
//			the second dimension is twice the hiddenSize. (input)
//
//	dy		Pointer to the hidden layer input tensor (input)
//
//	dhyD		hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	dhy		Pointer to the cell layer input tensor (input)
//
//	dcyD		A cell tensor descriptor that has as its first dimension
//			the number of layers if the direction mode is unidirectional and twice the
//			of layers if the direction mode is bidirectional. The second dimension of
//			must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	dcy		Pointer to the cell layer input tensor. If dcy is NULL,
//			then the initial delta cell state will be zero initialized. (input)
//
//	wD		A weights tensor descriptor (input)
//
//	w		Pointer to input weights tensor (input)
//
//	hxD		An input hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hx		Pointer to the hidden layer input tensor. If hx is NULL,
//			then the initial hidden state will be zero initialized. (input)
//
//	cxD		A input cell tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	cx		Pointer to the hidden layer input tensor. If cx is NULL,
// 			then the initial cell state will be zero initialized. (input)
//
// 	dxD		An array of tensor descriptors. These are the
//			input descriptors to each time step. The first dimension of each descriptor is the
//			batch size and may decrease from element n to element n+1 and not increase in size.
//			The second dimension is the same for all descriptors in the array and is the input
//			vector length. (input)
//
//
//	dx		Pointer to the cell layer output tensor (output)
//
//	dhxD		A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	dhx		Pointer to the delta hidden layer output tensor. If dhx is NULL
//			the hidden gradient will not ouput. (output)
//
//	dcxD		A tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	dcx		Pointer to the cell layer output tensor. If dcx is NULL
//			the cell gradient will not ouput. (output)
//
//	wspace		Pointer to memory allocated for forward training (input)
//
//	wspaceSIB	Number of allocated bytes in memory for the workspace (input)
//
//	rspace		Pointer to memory allocated for random states (input / output)
//
//	rspaceSIB	Number of allocated bytes in memory for use in the forward (input)
func (r *RNND) BackwardData(h *Handle,
	yD []*TensorD, y cutil.Mem,
	dyD []*TensorD, dy cutil.Mem,
	dhyD *TensorD, dhy cutil.Mem,
	dcyD *TensorD, dcy cutil.Mem,
	wD *TensorD, w cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	dxD []*TensorD, dx cutil.Mem,
	dhxD *TensorD, dhx cutil.Mem,
	dcxD *TensorD, dcx cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
	rspace cutil.Mem, rspaceSIB uint) error {

	dxDc, seqenceLen1 := tensorDarraytomiopenTensorDescriptorArray(dxD)
	dyDc, seqenceLen2 := tensorDarraytomiopenTensorDescriptorArray(dyD)
	yDc, seqenceLen3 := tensorDarraytomiopenTensorDescriptorArray(yD)
	if seqenceLen1 != seqenceLen2 || seqenceLen1 != seqenceLen3 {
		return errors.New("(r *RNND)BackwardData(): len(dxD)!=len(dyD) || len(dxD)!= len(yD)")
	}

	return Status(C.miopenRNNBackwardData(h.x, r.d,
		seqenceLen1, &yDc[0], y.Ptr(),
		&dyDc[0], dy.Ptr(),
		dhyD.d, dhy.Ptr(),
		dcyD.d, dcy.Ptr(),
		wD.d, w.Ptr(),
		hxD.d, hx.Ptr(),
		cxD.d, cx.Ptr(),
		&dxDc[0], dx.Ptr(),
		dhxD.d, dhx.Ptr(),
		dcxD.d, dcx.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		rspace.Ptr(), (C.size_t)(rspaceSIB))).error("(r *RNND)BackwardData()")
}

//BackwardWeights - Execute forward training for recurrent layer
//
//Interface for executing the forward training pass on a RNN.
//
//
//	h		MIOpen handle (input)
//
//	rnnD		RNN layer descriptor type (input)
//
//	xD		An array of tensor descriptors. These are the
//			input descriptors to each time step. The first dimension of each descriptor is the
//			size and may decrease from element n to element n+1 and not increase in size.
//			second dimension is the same for all descriptors in the array and is the input
//			length. (input)
//
//	x		to input tensor (input)
//
//	hxD		A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hx		to the hidden layer input tensor. If hx is NULL,
//			then the initial hidden state will be zero initialized. (input)
//
//	yD		An array of fully packed tensor descriptors associated
//			with the output from each time step. The first dimension of the tensor descriptors
//			must equal the first dimension of the first descriptor (batch size) in the xDesc
//			tensor array. The second dimension of the element of the descriptor array
//			depends on the direction mode selected. If the direction mode is unidirectional,
//			the second dimension is the hiddenSize. If direction mode is bidirectional
//			the second dimension is twice the hiddenSize. (input)
//
//	y		Pointer to the output tensor (input)
//
//	dwD		A weights tensor descriptor (input)
//
//	dw		Pointer to input weights tensor (input / output)
//
//	wspace		Pointer to memory allocated for forward training (input)
//
//	wspaceSIB	Number of allocated bytes in memory for the workspace (input)
//
//	rspace		Pointer to memory allocated for random states (input)
//
//	rspaceSIB	Number of allocated bytes in memory for use in the forward (input)
func (r *RNND) BackwardWeights(h *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	dwD *TensorD, dw cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint,
	rspace cutil.Mem, rspaceSIB uint) error {

	xDc, seqenceLen1 := tensorDarraytomiopenTensorDescriptorArray(xD)
	yDc, seqenceLen2 := tensorDarraytomiopenTensorDescriptorArray(yD)

	if seqenceLen1 != seqenceLen2 {
		return errors.New("(r *RNND)BackwardWeights(): len(xD)!=len(yD)")
	}

	return Status(C.miopenRNNBackwardWeights(h.x, r.d,
		seqenceLen1, &xDc[0], x.Ptr(),
		hxD.d, hx.Ptr(),
		&yDc[0], y.Ptr(),
		dwD.d, dw.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB),
		rspace.Ptr(), (C.size_t)(rspaceSIB))).error("(r *RNND)BackwardWeights()")
}

//ForwardInference - Execute forward inference for RNN layer
//
//Interface for executing the forward inference pass on a RNN.
//
//	h           	MIOpen handle (input)
//
//	xD          	An array of tensor descriptors. These are the
//		 	descriptors to each time step. The first dimension of each descriptor is the
//			batch size and may decrease from element n to element n+1 and not increase in size.
//			The second dimension is the same for all descriptors in the array and is the input
//			vector length. (input)
//
//	x		Pointer to input tensor (input)
//
//	hxD		A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hx          	Pointer to the hidden layer input tensor. If hx is NULL,
//			then the initial hidden state will be zero initialized. (input)
//
//	cxD         	A cell tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	cx          	Pointer to the cell layer input tensor. If cx is NULL,
//			then the initial cell state will be zero initialized. (input)
//
//	wD          	A weights tensor descriptor (input)
//
//	w           	Pointer to input weights tensor (input)
//
//	yD          	An array of fully packed tensor descriptors associated
//			with the output from each time step. The first dimension of the tensor descriptors
//			must equal the first dimension of the first descriptor (batch size) in the xDesc
//			tensor array. The second dimension of the element of the descriptor array
//			depends on the direction mode selected. If the direction mode is unidirectional,
//			the second dimension is the hiddenSize. If direction mode is bidirectional
//			the second dimension is twice the hiddenSize. (input)
//
//	y           	Pointer to output tensor (output)
//
//	hyD         	A hidden tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	hy          	Pointer to the hidden layer output tensor. If hy is NULL,
//			then the final hidden state will not be saved. (output)
//
//	cyD         	A cell tensor descriptor that has as its first dimension
//			of the number of layers if the direction mode is unidirectional and twice the
//			number of layers if the direction mode is bidirectional. The second dimension of
//			the descriptor must equal the largest first dimension of the xDesc tensor descriptor
//			array. The third dimension equals the hiddenSize. (input)
//
//	cy		Pointer to the cell layer output tensor. If hy is NULL,
//			then the final cell state will not be saved. (output)
//
//	wspace      	Pointer to memory allocated for forward training (input)
//
//	wspaceSIB   	Number of allocated bytes in memory for the workspace (input)
func (r *RNND) ForwardInference(h *Handle,
	xD []*TensorD, x cutil.Mem,
	hxD *TensorD, hx cutil.Mem,
	cxD *TensorD, cx cutil.Mem,
	wD *TensorD, w cutil.Mem,
	yD []*TensorD, y cutil.Mem,
	hyD *TensorD, hy cutil.Mem,
	cyD *TensorD, cy cutil.Mem,
	wspace cutil.Mem, wspaceSIB uint) error {

	xDc, seqenceLen1 := tensorDarraytomiopenTensorDescriptorArray(xD)
	yDc, seqenceLen2 := tensorDarraytomiopenTensorDescriptorArray(yD)
	if seqenceLen1 != seqenceLen2 {
		return errors.New("(r *RNND)ForwardInference(): len(xD)!=len(yD)")
	}

	return Status(C.miopenRNNForwardInference(h.x, r.d,
		seqenceLen1, &xDc[0], x.Ptr(),
		hxD.d, hx.Ptr(),
		cxD.d, cx.Ptr(),
		wD.d, w.Ptr(),
		&yDc[0], y.Ptr(),
		hyD.d, hy.Ptr(),
		cyD.d, cy.Ptr(),
		wspace.Ptr(), (C.size_t)(wspaceSIB))).error("(r *RNND)ForwardInference()")
}
