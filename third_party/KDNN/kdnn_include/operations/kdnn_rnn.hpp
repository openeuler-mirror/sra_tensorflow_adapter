/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: Rnn
 * Author: KPL
 * Create: 2025-01-08
 * Notes: NA
 */

#ifndef KDNN_RNN_HPP
#define KDNN_RNN_HPP

#include <memory>
#include <vector>
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_tensor_info.hpp"

namespace KDNN {

enum class RnnAlgorithm {
    UNIMPLEMENTED,
    RNN,
    LSTM,
    GRU,
    LBR_GRU,
    AUGRU,
    LBR_AUGRU
};

enum class ActivateFunctionRNN {
    UNIMPLEMENTED,
    RELU,
    TANH,
    LOGISTIC
};

namespace Detail {

enum class ExecutionDirectionT {
    L2R,
    R2L,
    BI_CONCAT,
    BI_SUM,
};

enum class CellPositionT {
    MIDDLE_CELL = 0x0,
    FIRST_LAYER = 0x1,
    FIRST_ITER = 0x2,
    LAST_LAYER = 0x4,
    LAST_ITER = 0x8,
    C_STATE_FIRST_ITER = 0x10,
    C_STATE_LAST_ITER = 0x20,
    MERGED_ITER = 0x40,
    MERGED_LAYER = 0x80
};

enum class WeightsTypeT {
    LAYER,
    ITER,
    PROJECTION,
    PEEPHOLE,
};

inline CellPositionT &operator|=(CellPositionT &lhs, CellPositionT rhs)
{
    lhs = static_cast<CellPositionT>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline CellPositionT operator|(CellPositionT lhs, CellPositionT rhs)
{
    return static_cast<CellPositionT>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline bool operator&(CellPositionT lhs, CellPositionT rhs)
{
    return static_cast<bool>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

enum class BrgemmRnnExecuteLoopOrderT {
    // default for kernels w/o loop order choice
    UNDEFINED = 0x0,
    // mBlocking loop is outermost
    M_BLK_N_BLK = 0x1,
    // nBlocking loop is outermost
    N_BLK_M_BLK = 0x2
};

struct DiffSrcBrgemmConfT {
    SizeType m = 0, n = 0, k = 0;

    SizeType nBlock = 0, nBlocks = 0, nTail = 0;
    SizeType mBlock = 0, mBlocks = 0;

    SizeType kBlocks = 0, kBlock = 0, kTail = 0;
    SizeType kpadded = 0;

    SizeType nIter = 0, nLayer = 0;
    SizeType nLayerBlocks = 0, nLayerTail = 0;
    SizeType nIterBlocks = 0, nIterTail = 0;
    SizeType lda = 0, ldb = 0, ldc = 0;

    BrgemmRnnExecuteLoopOrderT loopOrder
            = BrgemmRnnExecuteLoopOrderT::UNDEFINED;
    int gatesBlock;
};

struct DiffWeiBrgemmConfT {
    SizeType m = 0, mLayer = 0, mIter = 0, n = 0, k = 0;

    SizeType nBlock = 0, nBlocks = 0, nTail = 0;
    SizeType mBlock = 0, mBlocks = 0;
    SizeType kBlocks = 0, kBlock = 0, kTail = 0;
    SizeType kpadded = 0;
    SizeType ldaLayer = 0, ldaIter = 0, ldb = 0, ldcIter = 0, ldcLayer = 0;

    bool globalTranspose = false;

    BrgemmRnnExecuteLoopOrderT loopOrder
            = BrgemmRnnExecuteLoopOrderT::UNDEFINED;
};

constexpr int KDNN_RNN_MAX_N_PARTS = 4;

namespace Rnn {
class RnnFWDImpl;
class RnnBWDImpl;
} // Rnn

} // Detail

struct RnnCfg {
    using SizeType = KDNN::SizeType;

    KDNN::Detail::ExecutionDirectionT execDir;

    IntType nLayer = 0, nIter = 0, nDir = 0, nGates = 0;
    IntType mb = 0;
    IntType slc = 0, sic = 0, dhc = 0, dic = 0, dlc = 0;

    IntType nBias = 0;

    int weightsLayerLd = 0;
    int diffWeightsLayerLd = 0;
    int weightsIterLd = 0;
    int diffWeightsIterLd = 0;
    int weightsProjectionLd = 0;
    int diffWeightsProjectionLd = 0, diffWeightsProjectionNld = 0;

    int wsGatesLd = 0, wsGatesNld = 0;
    int wsHtLd = 0, wsHtNld = 0;
    int projHtLd = 0;
    int wsStatesLayerLd = 0, wsStatesLayerNld = 0;
    int wsStatesIterLd = 0;
    int wsDiffStatesLayerLd = 0;
    int wsDiffStatesIterLd = 0;
    int wsDiffStatesIterCLd = 0;

    int scratchGatesLd = 0, scratchGatesNld = 0;
    int scratchDiffHtLd = 0;

    int srcLayerLd = 0, srcLayerCellLd = 0;
    int srcIterLd = 0;
    int srcIterCLd = 0;
    int dstLayerLd = 0, dstLayerCellLd = 0;
    int dstIterLd = 0;
    int dstIterCLd = 0;

    bool isTraining = false, isLbr = false;

    SizeType wsGatesSize = 0;
    SizeType wsHtSize = 0;
    SizeType wsStatesLayerSize = 0;
    SizeType wsStatesIterCSize = 0;
    SizeType wsDiffStatesLayerSize = 0;
    SizeType wsDiffStatesIterSize = 0;
    SizeType wsDiffStatesIterCSize = 0;
    SizeType scratchGatesSize = 0;
    SizeType scratchCellSize = 0;
    SizeType wsGridCompSize = 0;
    SizeType wsPerCell = 0;

    bool srcLayerIsTrivialStride = false;
    bool dstLayerIsTrivialStride = false;

    bool diffWeightsOverwrite = false;

    SizeType wsGatesOffset = 0;
    SizeType wsHtOffset = 0;
    SizeType wsStatesLayerOffset = 0;
    SizeType wsStatesIterOffset = 0;
    SizeType wsStatesIterCOffset = 0;
    SizeType wsBiasOffset = 0;
    SizeType wsDiffStatesLayerOffset = 0;
    SizeType wsDiffStatesIterOffset = 0;
    SizeType wsDiffStatesIterCOffset = 0;
    SizeType wsGridCompOffset = 0;
    SizeType scratchGatesOffset = 0;
    SizeType scratchHtOffset = 0;
    SizeType scratchDiffHtOffset = 0;
    SizeType scratchCellOffset = 0;

    float alpha = 0;
    inline bool SkipDstLayerCopy() const
    {
        return (execDir == KDNN::Detail::ExecutionDirectionT::L2R || execDir == KDNN::Detail::ExecutionDirectionT::R2L);
    }
    inline bool SkipDstIterCopy() const
    {
        return (execDir == KDNN::Detail::ExecutionDirectionT::L2R ||
            execDir == KDNN::Detail::ExecutionDirectionT::R2L) && (dstIterLd > 0);
    }

    inline SizeType SrcLayerLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::FIRST_LAYER)
                ? srcLayerLd
                : (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER)
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType SrcIterLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER)
                ? srcIterLd
                : ((cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) && SkipDstLayerCopy()
                                        && !(cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER)
                                ? dstLayerLd
                                : wsStatesIterLd);
    }

    inline SizeType SrcIterCLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return isTraining ?
            ((cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER) ? srcIterCLd : wsDiffStatesIterCLd) :
            (cellPosition & KDNN::Detail::CellPositionT::FIRST_ITER) ? srcIterCLd : dstIterCLd;
    }
    inline SizeType DstIterCLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return isTraining ?
            ((cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) ? dstIterCLd : wsDiffStatesIterCLd) :
            dstIterCLd;
    }
    inline SizeType DstLayerLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) && SkipDstLayerCopy()
                ? dstLayerLd
                : (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) && SkipDstIterCopy()
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType DstIterLd(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER) && SkipDstLayerCopy()
                ? dstIterLd
                : wsStatesLayerLd;
    }

    inline SizeType DstIterPart2Ld(KDNN::Detail::CellPositionT cellPosition) const
    {
        return (cellPosition & KDNN::Detail::CellPositionT::LAST_LAYER) ? DstLayerLd(cellPosition)
                                            : DstIterLd(cellPosition);
    }

    // get DiffWeightsBeta based on cell position
    inline float DiffWeightsBeta(KDNN::Detail::CellPositionT cellPosition) const
    {
        if (diffWeightsOverwrite && (cellPosition & KDNN::Detail::CellPositionT::LAST_ITER)) {
            // Initialize diff weights if needed
            return 0.0f;
        }
        return 1.0f;
    }
};

class KDNN_API_PUBLIC RnnFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn,
    RnnAlgorithm alg, const ActivateFunctionRNN actf) noexcept;
    RnnFWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
        const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn, RnnAlgorithm alg,
        const ActivateFunctionRNN actf) noexcept(false);
    RnnFWD(const RnnFWD &other) noexcept(false);
    RnnFWD(RnnFWD &&other) noexcept;
    RnnFWD& operator=(const RnnFWD &other) noexcept(false);
    RnnFWD& operator=(RnnFWD &&other) noexcept;
    void Run(const void **src, const void **weight, void **dst, const void *bias, void *wsBase,
        const void *augruAttentions) const noexcept(false);
    ~RnnFWD() noexcept;
private:
    std::unique_ptr<Detail::Rnn::RnnFWDImpl> pImpl;
};

class KDNN_API_PUBLIC RnnBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
    const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
    const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnAlgorithm alg,
    const ActivateFunctionRNN actf) noexcept;
    RnnBWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
           const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
           const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
           const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnCfg &rnn, RnnAlgorithm alg,
           const ActivateFunctionRNN actf) noexcept(false);
    RnnBWD(const RnnBWD &other) noexcept(false);
    RnnBWD(RnnBWD &&other) noexcept;
    RnnBWD& operator=(const RnnBWD &other) noexcept(false);
    RnnBWD& operator=(RnnBWD &&other) noexcept;
    void Run(const void **src, const void **weight, const void **dst, const void *bias, void *wsBase,
        const void **diffDstVec, void **diffSrcVec, void **diffWeightVec, void *diffBias,
        const void *augruAttention, void *diffAugruAttention) const noexcept(false);
    ~RnnBWD() noexcept;
private:
    std::unique_ptr<Detail::Rnn::RnnBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_RNN_HPP
