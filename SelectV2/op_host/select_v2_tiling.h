#include "register/tilingdata_base.h"
#include "graph/utils/type_utils.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, smallDataNum); 	    // 小核处理的总数据数量（个）
    TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);	// 小核上数据搬运的次数
    TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);		    // 单核单次搬运可处理的数据数量
    TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);	// 小核最后一次搬运可处理的数据数量
    
    // 使用单个标志位表示是否需要广播
    TILING_DATA_FIELD_DEF(uint8_t, needBroadcast);
    
    // 只有当needBroadcast为1时，以下字段才会被使用
    // 广播相关的 - 减小数组大小和数据类型
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, yShape);      // y的shape 
    TILING_DATA_FIELD_DEF(uint8_t, yDimNum);             // y的维度数量
    
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, condStrides); // cond的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x1Strides);   // x1的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x2Strides);   // x2的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, yStrides);    // y的strides 
    
    // 使用位域减少内存使用
    TILING_DATA_FIELD_DEF(uint8_t, condNeedBroadcast);
    TILING_DATA_FIELD_DEF(uint8_t, x1NeedBroadcast);
    TILING_DATA_FIELD_DEF(uint8_t, x2NeedBroadcast);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectV2, SelectV2TilingData)
}
