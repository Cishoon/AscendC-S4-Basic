
#include "register/tilingdata_base.h"
#include "graph/utils/type_utils.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, smallDataNum); 	    // 小核处理的总数据数量（个）
    TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);	// 小核上数据搬运的次数
    TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);		    // 单核单次搬运可处理的数据数量
    TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);	// 小核最后一次搬运可处理的数据数量
    
    // 广播相关的
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, condShape); // cond的shape
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x1Shape);   // x1的shape
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x2Shape);   // x2的shape
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, yShape);    // y的shape 
    TILING_DATA_FIELD_DEF(uint32_t, yDimNum);    // y的shape 
    
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, condStrides); // cond的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x1Strides);   // x1的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x2Strides);   // x2的strides
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, yStrides);    // y的strides 
    
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectV2, SelectV2TilingData)
}
