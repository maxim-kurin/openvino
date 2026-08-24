// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_dynamic_pipeline.hpp"
#include "zero_infer_request.hpp"

namespace intel_npu {

class ZeroDynamicInferRequest final : public ZeroInferRequest {
public:
    explicit ZeroDynamicInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                     const std::shared_ptr<const ICompiledModel>& compiledModel,
                                     const Config& config);

    void infer_async() override;

protected:
    void create_pipeline_impl() override;

    std::shared_ptr<ZeroTensor> allocate_tensor(
        const size_t index,
        const bool isInput,
        const std::optional<std::size_t>& batchSize = std::nullopt) const override;

    void sync_zero_tensor_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                     const ov::SoPtr<ov::ITensor>& tensor) override;
    void sync_zero_tensors_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                      const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                      const std::optional<size_t>& batchSize = std::nullopt) override;

    void predict_shapes(std::vector<IDynamicGraph::MemRefType>& outputProps);
    void check_tensor_and_predicted_shapes(const std::vector<IDynamicGraph::MemRefType>& outputProps);

    void update_tensor(const std::vector<IDynamicGraph::MemRefType>& outputProps);

    bool _isTensorChanged = false;

    /**
     * @brief The output shapes the last prediction computed from the actual input shapes, one entry per
     * output, empty until the first prediction runs.
     * @details This is what an output whose shape the compiler could not bound gets allocated at: no upper
     * bound means there is no capacity in the metadata to fall back on, and unlike an input there is no
     * tensor the caller must have bound already. The prediction runs before the outputs are prepared, so
     * the size is known by the time the buffer is allocated.
     */
    std::vector<std::optional<ov::Shape>> _predictedOutputShapes;

private:
    std::shared_ptr<IDynamicGraph::GraphArguments> _binding;
};

}  //  namespace intel_npu
