package miopen

/*
#include <miopen/miopen.h>

*/
import "C"
import "runtime"

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
func (r *RNND) GetWorkspaceSize(h *Handle, sequenceLen int32, xD *TensorD) (wspacesib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNWorkspaceSize(h.x, r.d, (C.int)(sequenceLen), &xD.d, &sizet)).error("GetWorkspaceSize")
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
func (r *RNND) GetTrainingReserveSize(h *Handle, sequenceLen int32, xD *TensorD) (reservesib uint, err error) {
	var sizet C.size_t
	err = Status(C.miopenGetRNNTrainingReserveSize(h.x, r.d, (C.int)(sequenceLen), &xD.d, &sizet)).error("GetTrainingReserveSize")
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
