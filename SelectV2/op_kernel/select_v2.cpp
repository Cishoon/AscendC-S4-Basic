#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSelectV2 {
private:
    uint32_t tileDataNum; // 除了最后一次，tile里的数据数量
    uint32_t dataNum; // 这个核要计算的数据数量
    uint32_t tileNum; // 这个核要计算的tile数量
    uint32_t tailDataNum; // 这个核最后一次计算的数据数量
    uint32_t processDataNum; // 这次要处理的数据数量
    
public:
    __aicore__ inline KernelSelectV2() {}
    __aicore__ inline void Init(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                uint32_t smallDataNum, uint32_t finalSmallTileNum, 
                                uint32_t tileDataNum, uint32_t smallTailDataNum, 
                                AscendC::TPipe* pipeIn)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        ASSERT(blockNum != 0 && "GetBlockNum() is 0");
        
        this->tileDataNum = tileDataNum;
        
        this->dataNum = smallDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        
        conditionGm.SetGlobalBuffer((__gm__ DTYPE_CONDITION *)condition, this->dataNum);
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, this->dataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, this->dataNum);
        
        pipe = pipeIn;
        pipe->InitBuffer(inQueueCondition, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_CONDITION));
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X2));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        
        if constexpr (std::is_same_v<DTYPE_X1, half>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp3, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp4, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp5, this->tileDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(int32_t));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(int32_t));
            pipe->InitBuffer(tmp3, this->tileDataNum * sizeof(half));
        } else if constexpr (std::is_same_v<DTYPE_X1, float>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(uint8_t));
        }
    }
    
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum; // 这次要处理的数据数量
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
    
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_CONDITION> conditionLocal = inQueueCondition.AllocTensor<DTYPE_CONDITION>();
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();        
        
        int32_t baseIndex = progress * this->tileDataNum;
        
        AscendC::DataCopy(conditionLocal, conditionGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        
        inQueueCondition.EnQue(conditionLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        if constexpr (std::is_same_v<DTYPE_X1, half>) {
            AscendC::LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
            
            AscendC::LocalTensor<half> conditionLocal = tmp1.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp2.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            outQueueY.EnQue<half>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            AscendC::LocalTensor<int8_t> _x1Local = inQueueX1.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _x2Local = inQueueX2.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _yLocal = outQueueY.AllocTensor<int8_t>();
            
            AscendC::LocalTensor<half> x1Local = tmp1.Get<half>();
            AscendC::LocalTensor<half> x2Local = tmp2.Get<half>();
            AscendC::LocalTensor<half> conditionLocal = tmp3.Get<half>();
            AscendC::LocalTensor<half> yLocal = tmp4.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp5.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(x1Local, _x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(x2Local, _x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            AscendC::Cast(_yLocal, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            
            outQueueY.EnQue<int8_t>(_yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(_x1Local);
            inQueueX2.FreeTensor(_x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            AscendC::LocalTensor<int32_t> x1Local = inQueueX1.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> x2Local = inQueueX2.DeQue<int32_t>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<int32_t> yLocal = outQueueY.AllocTensor<int32_t>();
            
            AscendC::LocalTensor<int32_t> conditionLocal = tmp1.Get<int32_t>();
            AscendC::LocalTensor<int32_t> nonCondtionLocal = tmp2.Get<int32_t>();
            AscendC::LocalTensor<half> tmp = tmp3.Get<half>();
            
            AscendC::Cast(tmp, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(conditionLocal, tmp, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            
            AscendC::Duplicate(nonCondtionLocal, (int32_t)1, this->processDataNum);
            AscendC::Sub(nonCondtionLocal, nonCondtionLocal, conditionLocal, this->processDataNum);
            AscendC::Mul(x1Local, x1Local, conditionLocal, this->processDataNum); 
            AscendC::Mul(x2Local, x2Local, nonCondtionLocal, this->processDataNum);
            AscendC::Add(yLocal, x1Local, x2Local, this->processDataNum);
            
            outQueueY.EnQue<int32_t>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, float>) {
            AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            
            AscendC::LocalTensor<half> conditionLocal = tmp1.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp2.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            outQueueY.EnQue<float>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else {
            AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
            AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
            AscendC::LocalTensor<DTYPE_CONDITION> _conditionLocal = inQueueCondition.DeQue<DTYPE_CONDITION>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            
            outQueueY.EnQue<DTYPE_Y>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueCondition;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp4;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp5;
    
    AscendC::GlobalTensor<DTYPE_X1> x1Gm;
    AscendC::GlobalTensor<DTYPE_X2> x2Gm;
    AscendC::GlobalTensor<DTYPE_CONDITION> conditionGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

class KernelSelectV2BroadCast {
private:
    uint32_t tileDataNum; // 除了最后一次，tile里的数据数量
    uint32_t dataNum; // 这个核要计算的数据数量
    uint32_t tileNum; // 这个核要计算的tile数量
    uint32_t tailDataNum; // 这个核最后一次计算的数据数量
    uint32_t processDataNum; // 这次要处理的数据数量
private:
    uint16_t* yShape;
    uint8_t yDimNum;
    
    uint8_t condNeedBroadcast;
    uint8_t x1NeedBroadcast;
    uint8_t x2NeedBroadcast;
    
    uint32_t* condStrides;
    uint32_t* x1Strides;
    uint32_t* x2Strides;
    uint32_t* yStrides;
    
public:
    __aicore__ inline KernelSelectV2BroadCast() {}
    __aicore__ inline void Init(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                uint32_t smallDataNum, uint32_t finalSmallTileNum, 
                                uint32_t tileDataNum, uint32_t smallTailDataNum, 
                                uint16_t* yShape, uint8_t yDimNum, 
                                uint32_t* condStrides, uint32_t* x1Strides, uint32_t* x2Strides, uint32_t* yStrides, 
                                uint8_t condNeedBroadcast, uint8_t x1NeedBroadcast, uint8_t x2NeedBroadcast, 
                                AscendC::TPipe* pipeIn)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        ASSERT(blockNum != 0 && "GetBlockNum() is 0");
        
        this->tileDataNum = tileDataNum;
        
        this->dataNum = smallDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        
        conditionGm.SetGlobalBuffer((__gm__ DTYPE_CONDITION *)condition, this->dataNum);
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, this->dataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, this->dataNum);
        
        pipe = pipeIn;
        pipe->InitBuffer(inQueueCondition, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_CONDITION));
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X2));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        
        if constexpr (std::is_same_v<DTYPE_X1, half>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp3, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp4, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp5, this->tileDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(int32_t));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(int32_t));
            pipe->InitBuffer(tmp3, this->tileDataNum * sizeof(half));
        } else if constexpr (std::is_same_v<DTYPE_X1, float>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(uint8_t));
        }
        
        // 广播相关参数
        this->yShape = yShape;
        this->yDimNum = yDimNum;
        
        this->condStrides = condStrides;
        this->x1Strides = x1Strides;
        this->x2Strides = x2Strides;
        this->yStrides = yStrides;
        
        // 判断是否需要广播
        this->condNeedBroadcast = condNeedBroadcast;
        this->x1NeedBroadcast = x1NeedBroadcast;
        this->x2NeedBroadcast = x2NeedBroadcast;

    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum; // 这次要处理的数据数量
        uint32_t i = 0;
        for (i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    
private:
    __aicore__ inline uint32_t convertIndex(uint32_t srcIndex, uint32_t* strides)
    {
        uint32_t dstIndex = 0;
        for (int32_t i = 0; i < this->yDimNum; i++) {
            if (strides[i] != 0) {
                dstIndex += srcIndex / yStrides[i] % yShape[i] * strides[i];
            }
        }
        return dstIndex;
    }
    
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_CONDITION> conditionLocal = inQueueCondition.AllocTensor<DTYPE_CONDITION>();
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();        
        
        int32_t baseIndex = progress * this->tileDataNum;
        
        if (this->condNeedBroadcast) {
            uint32_t condIndex = 0;
            uint32_t r[8] {};
            uint32_t indices[8] {};
            uint32_t currentOffset = 0;
            
            uint32_t n = baseIndex;
            for (uint8_t i = 0; i < this->yDimNum; i++) {
                r[i] = n % yStrides[i];
                indices[i] = n / yStrides[i] % yShape[i];
                currentOffset += indices[i] * condStrides[i];
            }
            
            conditionLocal.SetValue(0, conditionGm.GetValue(currentOffset));
            
            for (int i = 1; i < this->processDataNum; i++) {
                n += 1;
                for (uint8_t dim = 0; dim < this->yDimNum; dim++) {
                    const uint32_t& stride = condStrides[dim];
                    if (stride == 0) {
                        continue;
                    }
                    uint32_t &rdim = r[dim];
                    if (rdim + 1 == yStrides[dim]) {
                        rdim = 0;
                        uint32_t &indice = indices[dim];
                        if (indice + 1 == yShape[dim]) {
                            currentOffset -= indice * stride;
                            indice = 0;
                        } else {
                            currentOffset += stride;
                            indice += 1;
                        }
                    } else {
                        ++rdim;
                    }
                }
                conditionLocal.SetValue(i, conditionGm.GetValue(currentOffset));
            }
        } else {
            AscendC::DataCopy(conditionLocal, conditionGm[progress * this->tileDataNum], this->processDataNum);
        }

        if (this->x1NeedBroadcast) {
            uint32_t x1Index = 0;
            uint32_t r[8] {};
            uint32_t indices[8] {};
            uint32_t currentOffset = 0;
            
            uint32_t n = baseIndex;
            for (uint8_t i = 0; i < this->yDimNum; i++) {
                r[i] = n % yStrides[i];
                indices[i] = n / yStrides[i] % yShape[i];
                currentOffset += indices[i] * x1Strides[i];
            }
            
            x1Local.SetValue(0, x1Gm.GetValue(currentOffset));
            
            for (int i = 1; i < this->processDataNum; i++) {
                n += 1;
                for (uint8_t dim = 0; dim < this->yDimNum; dim++) {
                    const uint32_t& stride = x1Strides[dim];
                    if (stride == 0) {
                        continue;
                    }
                    uint32_t &rdim = r[dim];
                    if (rdim + 1 == yStrides[dim]) {
                        rdim = 0;
                        uint32_t &indice = indices[dim];
                        if (indice + 1 == yShape[dim]) {
                            currentOffset -= indice * stride;
                            indice = 0;
                        } else {
                            currentOffset += stride;
                            indice += 1;
                        }
                    } else {
                        ++rdim;
                    }
                }
                x1Local.SetValue(i, x1Gm.GetValue(currentOffset));
            }
            
        } else {
            AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        }
        
        if (this->x2NeedBroadcast) {
            uint32_t x2Index = 0;
            uint32_t r[8] {};
            uint32_t indices[8] {};
            uint32_t currentOffset = 0;
            
            uint32_t n = baseIndex;
            for (uint8_t i = 0; i < this->yDimNum; i++) {
                r[i] = n % yStrides[i];
                indices[i] = n / yStrides[i] % yShape[i];
                currentOffset += indices[i] * x2Strides[i];
            }
            
            x2Local.SetValue(0, x2Gm.GetValue(currentOffset));
            
            for (int i = 1; i < this->processDataNum; i++) {
                n += 1;
                for (uint8_t dim = 0; dim < this->yDimNum; dim++) {
                    const uint32_t& stride = x2Strides[dim];
                    if (stride == 0) {
                        continue;
                    }
                    uint32_t &rdim = r[dim];
                    if (rdim + 1 == yStrides[dim]) {
                        rdim = 0;
                        uint32_t &indice = indices[dim];
                        if (indice + 1 == yShape[dim]) {
                            currentOffset -= indice * stride;
                            indice = 0;
                        } else {
                            currentOffset += stride;
                            indice += 1;
                        }
                    } else {
                        ++rdim;
                    }
                }
                x2Local.SetValue(i, x2Gm.GetValue(currentOffset));
            }
        } else {
            AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        }
        
        inQueueCondition.EnQue(conditionLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        if constexpr (std::is_same_v<DTYPE_X1, half>) {
            AscendC::LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
            
            AscendC::LocalTensor<half> conditionLocal = tmp1.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp2.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            outQueueY.EnQue<half>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            AscendC::LocalTensor<int8_t> _x1Local = inQueueX1.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _x2Local = inQueueX2.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> _yLocal = outQueueY.AllocTensor<int8_t>();
            
            AscendC::LocalTensor<half> x1Local = tmp1.Get<half>();
            AscendC::LocalTensor<half> x2Local = tmp2.Get<half>();
            AscendC::LocalTensor<half> conditionLocal = tmp3.Get<half>();
            AscendC::LocalTensor<half> yLocal = tmp4.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp5.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(x1Local, _x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(x2Local, _x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            AscendC::Cast(_yLocal, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            
            outQueueY.EnQue<int8_t>(_yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(_x1Local);
            inQueueX2.FreeTensor(_x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            AscendC::LocalTensor<int32_t> x1Local = inQueueX1.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> x2Local = inQueueX2.DeQue<int32_t>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<int32_t> yLocal = outQueueY.AllocTensor<int32_t>();
            
            AscendC::LocalTensor<int32_t> conditionLocal = tmp1.Get<int32_t>();
            AscendC::LocalTensor<int32_t> nonCondtionLocal = tmp2.Get<int32_t>();
            AscendC::LocalTensor<half> tmp = tmp3.Get<half>();
            
            AscendC::Cast(tmp, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(conditionLocal, tmp, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            
            AscendC::Duplicate(nonCondtionLocal, (int32_t)1, this->processDataNum);
            AscendC::Sub(nonCondtionLocal, nonCondtionLocal, conditionLocal, this->processDataNum);
            AscendC::Mul(x1Local, x1Local, conditionLocal, this->processDataNum); 
            AscendC::Mul(x2Local, x2Local, nonCondtionLocal, this->processDataNum);
            AscendC::Add(yLocal, x1Local, x2Local, this->processDataNum);
            
            outQueueY.EnQue<int32_t>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<DTYPE_X1, float>) {
            AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            AscendC::LocalTensor<int8_t> _conditionLocal = inQueueCondition.DeQue<int8_t>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            
            AscendC::LocalTensor<half> conditionLocal = tmp1.Get<half>();
            AscendC::LocalTensor<uint8_t> selMask = tmp2.Get<uint8_t>();
            
            AscendC::Cast(conditionLocal, _conditionLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::CompareScalar(selMask, conditionLocal, (half)0, AscendC::CMPMODE::GT, this->processDataNum);
            AscendC::Select(yLocal, selMask, x1Local, x2Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            
            outQueueY.EnQue<float>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else {
            AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
            AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
            AscendC::LocalTensor<DTYPE_CONDITION> _conditionLocal = inQueueCondition.DeQue<DTYPE_CONDITION>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            
            outQueueY.EnQue<DTYPE_Y>(yLocal);
            inQueueCondition.FreeTensor(_conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }
    
private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueCondition;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp4;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp5;
    
    AscendC::GlobalTensor<DTYPE_X1> x1Gm;
    AscendC::GlobalTensor<DTYPE_X2> x2Gm;
    AscendC::GlobalTensor<DTYPE_CONDITION> conditionGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void select_v2(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    
    if (tiling_data.needBroadcast) {
        KernelSelectV2BroadCast op;
        op.Init(condition, x1, x2, y, tiling_data.smallDataNum, tiling_data.finalSmallTileNum, 
                tiling_data.tileDataNum, tiling_data.smallTailDataNum, 
                tiling_data.yShape, tiling_data.yDimNum, 
                tiling_data.condStrides, tiling_data.x1Strides, tiling_data.x2Strides, tiling_data.yStrides, 
                tiling_data.condNeedBroadcast, tiling_data.x1NeedBroadcast, tiling_data.x2NeedBroadcast, 
                &pipe);
        op.Process();
    } else {
        KernelSelectV2 op;
        op.Init(condition, x1, x2, y, tiling_data.smallDataNum, tiling_data.finalSmallTileNum, 
                tiling_data.tileDataNum, tiling_data.smallTailDataNum, &pipe);
        op.Process();
    }
}