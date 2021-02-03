// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoderSeqLen(
        const ngraph::Output<Node>& inputData,
        const ngraph::Output<Node>& seqLen,
        int blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrec) {
    const auto& inputDataShape = inputData.get_shape();
    const size_t B = inputDataShape[0];
    const size_t T = inputDataShape[1];

    std::vector<int> blankIdxData = {blankIndex};
    auto blankIndexNode = makeConstant(idxPrec, {1}, blankIdxData);

    return std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(inputData, seqLen, blankIndexNode,
                                                            mergeRepeated, idxPrec, idxPrec);
}
}  // namespace builder
}  // namespace ngraph
