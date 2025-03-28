#include "select_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32; // block字节数，常量
const uint32_t BUFFER_NUM = 2;	// double buffer，常量
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SelectV2TilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    /// 广播相关tiling
    // 1. 获取输入输出shape
    auto condShape = context->GetInputShape(0)->GetOriginShape();
    auto x1Shape = context->GetInputShape(1)->GetOriginShape();
    auto x2Shape = context->GetInputShape(2)->GetOriginShape();
    auto yShape = context->GetOutputShape(0)->GetOriginShape();
    int32_t yDimNum = static_cast<int32_t>(yShape.GetDimNum());
    int32_t condDimNum = static_cast<int32_t>(condShape.GetDimNum());
    int32_t x1DimNum = static_cast<int32_t>(x1Shape.GetDimNum());
    int32_t x2DimNum = static_cast<int32_t>(x2Shape.GetDimNum());
    if (yDimNum > 20 || condDimNum > 20 || x1DimNum > 20 || x2DimNum > 20) {
        return ge::GRAPH_FAILED;
    }
    uint32_t condShapeVec[20] {};
    uint32_t x1ShapeVec[20] {};
    uint32_t x2ShapeVec[20] {};
    uint32_t yShapeVec[20] {};
    for (int32_t i = 0; i < yDimNum; i++) {
        yShapeVec[i] = yShape.GetDim(yDimNum - 1 - i);
        condShapeVec[i] = condDimNum - 1 - i >= 0 ? condShape.GetDim(condDimNum - 1 - i) : 1;
        x1ShapeVec[i] = x1DimNum - 1 - i >= 0 ? x1Shape.GetDim(x1DimNum - 1 - i) : 1;
        x2ShapeVec[i] = x2DimNum - 1 - i >= 0 ? x2Shape.GetDim(x2DimNum - 1 - i) : 1;
    }
    
    // 2. 获取输入输出strides
    uint32_t condStrides[20] {};
    uint32_t x1Strides[20] {};
    uint32_t x2Strides[20] {};
    uint32_t yStrides[20] {};
    
    uint32_t y_stride = 1, cond_stride = 1, x1_stride = 1, x2_stride = 1;
    for (size_t i = 0; i < yDimNum; i++) {
        if (yShapeVec[i] != 1) {
            yStrides[i] = y_stride;
            y_stride *= yShapeVec[i];
        }
        if (condShapeVec[i] != 1) {
            condStrides[i] = cond_stride;
            cond_stride *= condShapeVec[i];
        }
        if (x1ShapeVec[i] != 1) {
            x1Strides[i] = x1_stride;
            x1_stride *= x1ShapeVec[i];
        }
        if (x2ShapeVec[i] != 1) {
            x2Strides[i] = x2_stride;
            x2_stride *= x2ShapeVec[i];
        }
    }
    
    // 判断是否需要广播
    uint32_t needBroadcast = 0;
    for (int i = 0; i < yDimNum; i++) {
        if (condShapeVec[i] != yShapeVec[i] || x1ShapeVec[i] != yShapeVec[i] || x2ShapeVec[i] != yShapeVec[i]) {
            needBroadcast = 1;
            break;
        }
    }

    // 每个核一次计算最多能处理的字节数，从接口获取
    uint64_t ubSize; 	
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 获取AICore数量
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = needBroadcast ? 1 : aivNum * 4;
    context->SetBlockDim(coreNum);
    
    // 获取输入数据数量, totalDataNum表示几个元素
    uint32_t totalDataNum = context->GetOutputShape(0)->GetOriginShape().GetShapeSize();
    
    // typeLength表示输入的数据类型占几个字节，inputLength表示输入数据的总字节
    uint32_t condTypeLength = 0, x1TypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), condTypeLength);
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), x1TypeLength);
    uint32_t r = x1TypeLength / condTypeLength; // 一定能整除，因为condTypeLength=1B
            
    // 总共有几个 cond 块
    uint32_t condBlockNum = (totalDataNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /// 计算每个tile内的参数
    // 1. tileCondBlockNum 一个tile里可以存几个 condBlock
    // uint32_t rate = 3 * r + 1 + 2 * r + 2; // 3r: x1, x2, y; 1: condition; 2r: oneBuf, condBuf; 2: castBuf
    uint32_t rate;
    auto x1DataType = context->GetInputDesc(1)->GetDataType();
    if (x1DataType == ge::DataType::DT_FLOAT16) {
        rate = 11;
    } else if (x1DataType == ge::DataType::DT_FLOAT) {
        rate = 23;
    } else if (x1DataType == ge::DataType::DT_INT8) {
        rate = 14;
    } else if (x1DataType == ge::DataType::DT_INT32) {
        rate = 23;
    } else {
        return ge::GRAPH_FAILED;
    }
    
    uint32_t tileCondBlockNum = ubSize / BUFFER_NUM / BLOCK_SIZE / rate;
    // 2. 一个tile里的总block数量
    uint32_t tileBlockNum = rate * tileCondBlockNum;
    // 3. 一个tile里的数据数量
    uint32_t tileDataNum = BLOCK_SIZE * tileCondBlockNum / condTypeLength;
    
    /// 计算核数上下界
    coreNum = (coreNum < condBlockNum) ? coreNum : condBlockNum;
    coreNum = (coreNum >= 1) ? coreNum : 1;
        
    // 每个核平均计算几个cond block
    uint32_t everyCoreInputCondBlock = condBlockNum / coreNum;
    // 余数，即需要多少个大核
    uint32_t tailBlockNum = condBlockNum % coreNum;
    
    /// 计算小核其他参数
    // 1. 小核处理的总数据数量（个）
    uint32_t smallDataNum = everyCoreInputCondBlock * BLOCK_SIZE / condTypeLength;
    // 2. 小核处理的tile数量, finalSmallTileNum
    uint32_t smallTileNum = everyCoreInputCondBlock / tileCondBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputCondBlock % tileCondBlockNum == 0) ? smallTileNum : smallTileNum + 1;
    // 3. 最后一次搬运的数据数量
    uint32_t smallTailDataNum = smallDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0? tileDataNum : smallTailDataNum;
         
    /// 计算大核其他参数
    everyCoreInputCondBlock += 1;
    // 1. 大核处理的总数据数量（个）
    uint32_t bigDataNum = everyCoreInputCondBlock * BLOCK_SIZE / condTypeLength;
    // 2. 大核处理的tile数量, finalBigTileNum
    uint32_t bigTileNum = everyCoreInputCondBlock / tileCondBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputCondBlock % tileCondBlockNum == 0) ? bigTileNum : bigTileNum + 1;
    // 3. 最后一次搬运的数据数量
    uint32_t bigTailDataNum = bigDataNum - (tileDataNum * bigTileNum);
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    /// 塞进tiling结构体
    tiling.set_bigDataNum(bigDataNum);
    tiling.set_smallDataNum(smallDataNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_tailBlockNum(tailBlockNum);

    
    // 3. 塞进tiling结构体
    tiling.set_yDimNum(yDimNum);
    tiling.set_condShape(condShapeVec);
    tiling.set_x1Shape(x1ShapeVec);
    tiling.set_x2Shape(x2ShapeVec);
    tiling.set_yShape(yShapeVec);
    tiling.set_condStrides(condStrides);
    tiling.set_x1Strides(x1Strides);
    tiling.set_x2Strides(x2Strides);
    tiling.set_yStrides(yStrides);
    
    /// workspace
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SelectV2 : public OpDef {
public:
    explicit SelectV2(const char* name) : OpDef(name)
    {
        this->Input("condition")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore()
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");

    }
};

OP_ADD(SelectV2);
}
