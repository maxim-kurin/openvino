// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

// SDPA-unroll and transpose transformations
class OptimizeValueTensors : public ov::pass::ModelPass {
    bool m_is_prefill;
    bool m_is_whisper;
    bool m_decompose_sdpa;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::OptimizeValueTensors");
    // decompose_sdpa=false keeps SDPA ops and restores the V layout with an explicit Transpose
    explicit OptimizeValueTensors(bool is_prefill, bool is_whisper = false, bool decompose_sdpa = true)
        : m_is_prefill(is_prefill),
          m_is_whisper(is_whisper),
          m_decompose_sdpa(decompose_sdpa) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace util
}  // namespace npuw
}  // namespace ov
