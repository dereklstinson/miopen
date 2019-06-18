package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import "runtime"
import "unsafe"
import "github.com/dereklstinson/cutil"

type RNNMode C.miopenRNNMode_t

func (r RNNMode) c() C.miopenRNNMode_t      { return (C.miopenRNNMode_t)(r) }
func (r *RNNMode) cptr() *C.miopenRNNMode_t { return (*C.miopenRNNMode_t)(r) }

func (r *RNNMode) RELU() RNNMode { *r = (RNNMode)(C.miopenRNNRELU); return *r }
func (r *RNNMode) Tanh() RNNMode { *r = (RNNMode)(C.miopenRNNTANH); return *r }
func (r *RNNMode) LSTM() RNNMode { *r = (RNNMode)(C.miopenLSTM); return *r }
func (r *RNNMode) GRU() RNNMode  { *r = (RNNMode)(C.miopenGRU); return *r }

type RNNInputMode C.miopenRNNInputMode_t

func (r RNNInputMode) c() C.miopenRNNInputMode_t      { return (C.miopenRNNInputMode_t)(r) }
func (r *RNNInputMode) cptr() *C.miopenRNNInputMode_t { return (*C.miopenRNNInputMode_t)(r) }
func (r *RNNInputMode) Linear() RNNInputMode          { *r = (RNNInputMode)(C.miopenRNNlinear); return *r }
func (r *RNNInputMode) Skip() RNNInputMode            { *r = (RNNInputMode)(C.miopenRNNskip); return *r }

type RNNAlgo C.miopenRNNAlgo_t

func (r RNNAlgo) c() C.miopenRNNAlgo_t      { return (C.miopenRNNAlgo_t)(r) }
func (r *RNNAlgo) cptr() *C.miopenRNNAlgo_t { return (*C.miopenRNNAlgo_t)(r) }
func (r *RNNAlgo) Default() RNNAlgo         { *r = (RNNAlgo)(C.miopenRNNdefault); return *r }

type RNNDirectionMode C.miopenRNNDirectionMode_t

func (r RNNDirectionMode) c() C.miopenRNNDirectionMode_t      { return (C.miopenRNNDirectionMode_t)(r) }
func (r *RNNDirectionMode) cptr() *C.miopenRNNDirectionMode_t { return (*C.miopenRNNDirectionMode_t)(r) }
func (r *RNNDirectionMode) UNI() RNNDirectionMode {
	*r = (RNNDirectionMode)(C.miopenRNNunidirection)
	return *r
}
func (r *RNNDirectionMode) BI() RNNDirectionMode {
	*r = (RNNDirectionMode)(C.miopenRNNbidirection)
	return *r
}

type RNNBiasMode C.miopenRNNBiasMode_t

func (r RNNBiasMode) c() C.miopenRNNBiasMode_t      { return (C.miopenRNNBiasMode_t)(r) }
func (r *RNNBiasMode) cptr() *C.miopenRNNBiasMode_t { return (*C.miopenRNNBiasMode_t)(r) }
func (r *RNNBiasMode) NoBias() RNNBiasMode          { *r = (RNNBiasMode)(C.miopenRNNNoBias); return *r }
func (r *RNNBiasMode) WithBias() RNNBiasMode        { *r = (RNNBiasMode)(C.miopenRNNwithBias); return *r }

type RNNGEMMalgoMode C.miopenRNNGEMMalgoMode_t

func (r RNNGEMMalgoMode) c() C.miopenRNNGEMMalgoMode_t      { return (C.miopenRNNGEMMalgoMode_t)(r) }
func (r *RNNGEMMalgoMode) cptr() *C.miopenRNNGEMMalgoMode_t { return (*C.miopenRNNGEMMalgoMode_t)(r) }
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

/*! @brief Set the details of the RNN descriptor
 *
 * Interface for setting the values of the RNN descriptor object. This function requires specific
 * algorithm selection.
 * @param rnnDesc      RNN layer descriptor type (input)
 * @param hsize        Hidden layer size (input)
 * @param nlayers      Number of layers (input)
 * @param inMode       RNN first layer input mode (input)
 * @param direction    RNN direction (input)
 * @param rnnMode      RNN model type (input)
 * @param biasMode     RNN bias included (input)
 * @param algo         RNN algorithm selected (input)
 * @param dataType     Only fp32 currently supported for RNNs (input)
 * @return             miopenStatus_t
 */
func (r *RNND) Set(hsize, nlayers int32,
	inMode RNNInputMode,
	direction RNNDirectionMode,
	mode RNNMode,
	biasmode RNNBiasMode,
	algo RNNAlgo,
	dtype DataType) error {
	return Status(C.miopenSetRNNDescriptor(r.d, (C.int)(hsize), (C.int)(nlayers), inMode.c(), direction.c(), mode.c(), biasmode.c(), algo.c(), dtype.c())).error("(r *RNND) Set()")
}

/*! @brief Retrieves a RNN layer descriptor's details
*
* @param rnnDesc    RNN layer descriptor (input)
* @param rnnMode    RNN mode (output)
* @param algoMode   RNN algorithm mode (output)
* @param inputMode  RNN data input mode (output)
* @param dirMode    Uni or bi direction mode (output)
* @param biasMode   Bias used (output)
* @param hiddenSize Size of hidden state (output)
* @param layer      Number of stacked layers (output)
* @return           miopenStatus_t
 */
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

/*! @brief Query the amount of memory required to execute the RNN layer
 *
 * This function calculates the amount of memory required to run the RNN layer given an RNN
 * descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param sequenceLen     Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetWorkspaceSize(h *Handle, xD []*TensorD) (wspacesib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := xDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNWorkspaceSize(h.x, r.d, seqLen, &xDs[0], &sizet)).error("GetWorkspaceSize")
	wspacesib = (uint)(sizet)
	return wspacesib, err
}

/*! @brief Query the amount of memory required for RNN training
 *
 * This function calculates the amount of memory required to train the RNN layer given an
 * RNN descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param sequenceLen     Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetTrainingReserveSize(h *Handle, xD []*TensorD) (reservesib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := xDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNTrainingReserveSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("GetTrainingReserveSize")
	reservesib = (uint)(sizet)
	return reservesib, err
}

/*! @brief Query the amount of parameter memory required for RNN training
 *
 * This function calculates the amount of parameter memory required to train the RNN layer given an
 * RNN descriptor and a tensor descriptor.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param xDesc           A tensor descriptor (input)
 * @param numBytes        Number of bytes required for RNN layer execution (output)
 * @param dtype           MIOpen data type enum (input)
 * @return                miopenStatus_t
 */
func (r *RNND) GetParamSize(h *Handle, xD *TensorD, dtype DataType) (paramSIB uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNParamsSize(h.x, r.d, xD.d, &sizet, dtype.c())).error("GetTrainingReserveSize")
	paramSIB = (uint)(sizet)
	return paramSIB, err
}

/*! @brief Obtain a weight tensor descriptor for RNNs
 *
 * This function populates a weight descriptor that describes the memory layout of the
 * weight matrix.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor type (input)
 * @param xDesc           A previously populated tensor descriptor (input)
 * @param wDesc           A previously allocated tensor descriptor (output)
 * @param dtype           MIOpen data type enum, currently only fp32 is supported (input)
 * @return                miopenStatus_t
 */
func (r *RNND) GetRNNDParamDescriptor(h *Handle, xD *TensorD, dtype DataType) (wD *TensorD, err error) {
	wD, err = CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	err = Status(C.miopenGetRNNParamsDescriptor(h.x, r.d, xD.d, wD.d, dtype.c())).error("(r *RNND)GetRNNDParamDescriptor()")
	return wD, err
}

/*! @brief Obtain a the size in bytes of the RNN input tensor
 *
 * This function determines the size in bytes of the allocation needed for the input data
 * tensor for an RNN layer. The number of bytes is derived from the array of
 * tensor descriptors.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor (input)
 * @param seqLen          Number of iteration unrolls (input)
 * @param xDesc           An array of tensor descriptors. These are the
 * input descriptors to each time step. The first dimension of each descriptor is the
 * batch size and may decrease from element n to element n+1 and not increase in size.
 * The second dimension is the same for all descriptors in the array and is the input
 * vector length. (input)
 * @param numBytes        Number of bytes required for input tensor (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetInputTensorSize(h *Handle, xD []*TensorD) (sib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := xDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNInputTensorSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("(r *RNND)miopenGetRNNInputTensorSize()")
	sib = (uint)(sizet)
	return sib, err
}

/*! @brief Obtain a the size in bytes of the RNN hidden tensor
 *
 * This function determines the size in bytes of the allocation needed for the
 * hidden tensor over all layers
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         Fully populated RNN layer descriptor type (input)
 * @param seqLen          Number of iteration unrolls (input)
 * @param xDesc           An array of previously populated tensor descriptors (input)
 * @param numBytes        Number of bytes required for input tensor (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetHiddenTensorSize(h *Handle, xD []*TensorD) (sib uint, err error) {
	var sizet C.size_t
	xDs, seqLen := xDarraytomiopenTensorDescriptorArray(xD)
	err = Status(C.miopenGetRNNHiddenTensorSize(h.x, r.d, (seqLen), &xDs[0], &sizet)).error("(r *RNND)GetHiddenTensorSize()")
	sib = (uint)(sizet)
	return sib, err
}

/*! @brief Gets the number of bytes of a parameter matrix
 *
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while paramID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param paramID         ID of the internal parameter tensor (input)
 * @param numBytes        The number of bytes of the layer's parameter matrix (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetLayerParamSize(h *Handle, layer int32, xD *TensorD, paramID int32) (sib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNLayerParamSize(h.x, r.d, (C.int)(layer), xD.d, (C.int)(paramID), &sizet)).error("(r *RNND)GetLayerParamSize()")
	sib = (uint)(sizet)
	return sib, err
}

/*! @brief Gets the number of bytes of a bias
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while biasID == 1 retrieves
 * the bias associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the biases associated
 * with the input GEMM, 4-7 are associated with biases associated with the
 * hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param numBytes        The number of bytes of the layer's bias (output)
 * @return                miopenStatus_t
 */
func (r *RNND) GetLayerBiasSize(h *Handle, layer, biasID int32) (sib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNLayerBiasSize(h.x, r.d, (C.int)(layer), (C.int)(biasID), &sizet)).error("(r *RNND)GetLayerBiasSize()")
	sib = (uint)(sizet)
	return sib, err
}

/*! @brief Gets a weight matrix for a specific layer in an RNN stack
 *
 * This function retrieves the weight matrix data for a specific layer and parameter ID
 * and copies the data into previously allocated device memory.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix associated with the in input GEMM, while paramID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrices associated
 * with the input GEMM, 4-7 are associated with matrices associated with the
 * hidden state GEMM.
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument paramDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the parameter matrix. It is full packed and is used when
 * calling to miopenSetRNNLayerParam()
 *
 * The argument layerParam should either be nullptr, or have device memory allocated
 * to allow copying of the entire layer parameter matrix into it. If layerParam is
 * nullptr then only the paramDesc is populated and returned. The size in bytes of the
 * layer parameter matrix can be determined by using miopenGetRNNLayerParamSize().
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerParam() will return
 * a error status miopenStatusBadParm for input paramID associated with the input GEMM.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the parameter tensor (input)
 * @param w               Pointer to memory containing parameter tensor (input)
 * @param paramID         ID of the internal parameter tensor (input)
 * @param paramDesc       Tensor descriptor for the fully packed output parameter tensor (output)
 * @param layerParam      Pointer to the memory location of the parameter tensor (output)
 * @return                miopenStatus_t
 */
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

/*! @brief Gets a bias for a specific layer in an RNN stack
 *
 * This function retrieves the bias data for a specific layer and bias ID and copies
 * the data into previously allocated device memory.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * bias associated with the in input GEMM, while biasID == 1 retrieves
 * the bias associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the biases associated
 * with the input GEMM, 4-7 are associated with biases associated with the
 * hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument biasDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the bias. It is full packed and is used when
 * calling to miopenSetRNNLayerBias()
 *
 * The argument layerBias should either be nullptr, or have device memory allocated
 * to allow copying of the entire layer bias into it. If layerBias is
 * nullptr then only the biasDesc is populated and returned. The size in bytes of the
 * layer bias can be determined by using miopenGetRNNLayerBiasSize().
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerBias() will return
 * a error status miopenStatusBadParm for input biasID associated with the input GEMM.
 *
 * @param handle          MIOpen handle (input)
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param wDesc           A tensor descriptor to the parameter tensor (input)
 * @param w               Pointer to memory containing parameter tensor (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param biasDesc        Descriptor of the parameter tensor (output)
 * @param layerBias       Pointer to the memory location of the bias tensor (output)
 * @return                miopenStatus_t
 */
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
/*! @brief Gets an index offset for a specific weight matrix for a layer in the
 *  RNN stack
 *
 * This function retrieves the index offset for a weight matrix in a layer.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, paramID == 0 retrieves the
 * weight matrix offset associated with the in input GEMM, while paramID == 1
 * retrieves the weight matrix offset associated with the hidden state GEMM.
 *
 * For miopenLSTM paramID 0 to 3 refer to the weight matrix offsets associated
 * with the input GEMM, 4-7 are associated with matrix offset associated with the
 * hidden state GEMM.
 *
 * * paramID 0 and 4 are for the input gate.
 *
 * * paramID 1 and 5 are for the forget gate.
 *
 * * paramID 2 and 6 are for the output gate.
 *
 * * paramID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU paramID 0 to 2 refer to the weight matrix offset associated
 * with the input GEMM, while 3 through 5 are associated with the hidden state
 * GEMM.
 *
 * * paramID 0 and 3 are for the update gate.
 *
 * * paramID 1 and 4 are for the reset gate.
 *
 * * paramID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument paramDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the parameter matrix. It is full packed and is used when
 * calling to miopenSetRNNLayerParam().
 *
 * The argument layerParamOffset should either be nullptr, or an address to place the
 * offset. If layerParamOffset is nullptr then only the paramDesc is populated and returned.
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerParamOffset() will return
 * a error status miopenStatusBadParm for input paramID associated with the input GEMM.
 *
 *
 * @param rnnDesc           RNN layer descriptor type (input)
 * @param layer             The layer number in the RNN stack (input)
 * @param xDesc             A tensor descriptor to input (input)
 * @param paramID           ID of the internal parameter tensor (input)
 * @param paramDesc         Tensor descriptor for the fully packed output parameter tensor (output)
 * @param layerParamOffset  Location for the parameter offset (output)
 * @return                  miopenStatus_t
*/
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
/*! @brief Gets a bias index offset for a specific layer in an RNN stack
 *
 * This function retrieves the bias index offset for a specific layer and bias ID.
 *
 * For RNN vanilla miopenRNNRELU and miopenRNNTANH, biasID == 0 retrieves the
 * bias associated with the in input GEMM, while biasID == 1 retrieves
 * the weight matrix associated with the hidden state GEMM.
 *
 * For miopenLSTM biasID 0 to 3 refer to the bias offset associated
 * with the input GEMM, 4-7 are the bias offsets associated with the hidden state GEMM.
 *
 * * biasID 0 and 4 are for the input gate.
 *
 * * biasID 1 and 5 are for the forget gate.
 *
 * * biasID 2 and 6 are for the output gate.
 *
 * * biasID 3 and 7 are for the new memory gate.
 *
 * For miopenGRU biasID 0 to 2 refer to the biases associated with the input GEMM,
 * while 3 through 5 are associated with the hidden state GEMM.
 *
 * * biasID 0 and 3 are for the update gate.
 *
 * * biasID 1 and 4 are for the reset gate.
 *
 * * biasID 2 and 5 are for the new memory gate.
 *
 * For bi-directional RNNs the backwards in time direction is numbered as the layer
 * directly after the forward in time direction.
 *
 * The output argument biasDesc is a previously created tensor descriptor that is populated
 * to describe the memory layout of the bias. It is full packed and is used when
 * calling to miopenSetRNNLayerBias()
 *
 * The argument layerBiasOffset should either be nullptr, or point to an output address.
 * If layerBias is nullptr then only the biasDesc is populated and returned.
 *
 * Note: When inputSkip mode is selected there is no input layer matrix operation,
 * and therefore no associated memory. In this case miopenGetRNNLayerBiasOffset() will return
 * a error status miopenStatusBadParm for input biasID associated with the input GEMM.
 *
 * @param rnnDesc         RNN layer descriptor type (input)
 * @param layer           The layer number in the RNN stack (input)
 * @param xDesc           A tensor descriptor to input (input)
 * @param biasID          ID of the internal parameter tensor (input)
 * @param biasDesc        Descriptor of the parameter tensor (output)
 * @param layerBiasOffset Pointer to the memory location of the bias tensor (output)
 * @return                miopenStatus_t
*/
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

func xDarraytomiopenTensorDescriptorArray(td []*TensorD) (ctd []C.miopenTensorDescriptor_t, seqlen C.int) {
	seqlen = (C.int)(len(td))
	ctd = make([]C.miopenTensorDescriptor_t, len(td))
	for i := range td {
		ctd[i] = td[i].d
	}
	return ctd, seqlen
}

type mem struct {
	x unsafe.Pointer
}

func (m *mem) Ptr() unsafe.Pointer {
	return m.x
}
func (m *mem) DPtr() *unsafe.Pointer {
	return &m.x
}
