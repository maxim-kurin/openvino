// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "optimize_value_tensors.hpp"

#include <algorithm>
#include <numeric>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>
#include <unordered_set>
#include <vector>

#include "../llm_compiled_model_utils.hpp"
#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/util/pp.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class TransposeValueTensors : public ov::pass::MatcherPass {
public:
    struct Context {
        using Ref = std::reference_wrapper<Context>;
        bool bTransposed = false;
        std::unordered_set<const ov::op::v0::Parameter*> transposed_params;
        ov::OutputVector changed_outputs;

        void track(const ov::Output<ov::Node>& output) {
            if (std::find(changed_outputs.begin(), changed_outputs.end(), output) == changed_outputs.end()) {
                changed_outputs.push_back(output);
            }
        }

        void restore_shape_of_consumers() const {
            for (const auto& output : changed_outputs) {
                const auto targets = output.get_target_inputs();
                for (const auto& input : targets) {
                    auto consumer = input.get_node()->shared_from_this();
                    if (!ov::is_type<ov::op::v0::ShapeOf>(consumer) && !ov::is_type<ov::op::v3::ShapeOf>(consumer)) {
                        continue;
                    }
                    const auto& rank = output.get_partial_shape().rank();
                    OPENVINO_ASSERT(rank.is_static() && rank.get_length() >= 2,
                                    "Cannot restore ShapeOf dimension order for a value tensor with dynamic rank");
                    std::vector<int64_t> order(static_cast<size_t>(rank.get_length()));
                    for (size_t i = 0; i < order.size(); ++i) {
                        order[i] = static_cast<int64_t>(i);
                    }
                    std::swap(order[order.size() - 2], order[order.size() - 1]);
                    auto order_constant =
                        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{order.size()}, order);
                    // Reorder only the shape vector, preserving runtime dimensions and all data edges.
                    auto shape_of = consumer->clone_with_new_inputs({output});
                    shape_of->set_friendly_name(consumer->get_friendly_name() + "/transposed_layout");
                    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                    auto restore_shape = std::make_shared<ov::op::v8::Gather>(shape_of, order_constant, axis);
                    restore_shape->set_friendly_name(consumer->get_friendly_name());
                    ov::copy_runtime_info(consumer, {shape_of, restore_shape});
                    ov::replace_node(consumer, restore_shape);
                }
            }
        }
    };

protected:
    // MQA: aten::transpose on a dim-1 axis (num_kv_heads==1) is optimized by the
    // frontend into a Reshape instead of a real Transpose. Replace such a Reshape with a
    // real Transpose so that transpose_v_cache can set the permutation order uniformly.
    // The placeholder order {0,2,1,3} will be overwritten to {0,2,3,1} by transpose_v_cache.
    // Returns the (possibly new) Transpose, or nullptr if the node is a Reshape that is
    // not a genuine MQA fake-transpose.
    static std::shared_ptr<ov::Node> to_real_transpose(const std::shared_ptr<ov::Node>& node) {
        if (!ov::is_type<ov::op::v1::Reshape>(node)) {
            return node;
        }
        // Only accept a Reshape that is semantically a genuine MQA fake-transpose:
        //   [B, Snew, 1, D] → [B, 1, Snew, D]
        // i.e. 4D→4D, axes 1 and 2 are swapped, and the swapped-out dim equals 1
        // (num_kv_heads==1).  Any other Reshape (e.g. 3D→4D in negative tests)
        // must be rejected so the matcher does not fire.
        const auto& in_shape = node->input_value(0).get_partial_shape();
        const auto& out_shape = node->get_output_partial_shape(0);
        const bool is_mqa_fake_transpose = in_shape.rank().is_static() && in_shape.rank().get_length() == 4 &&
                                           out_shape.rank().is_static() && out_shape.rank().get_length() == 4 &&
                                           in_shape[0].compatible(out_shape[0]) &&  // B unchanged
                                           in_shape[3].compatible(out_shape[3]) &&  // D unchanged
                                           in_shape[1].compatible(out_shape[2]) &&  // Snew swapped to dim 2
                                           in_shape[2].compatible(out_shape[1]) &&  // kv_heads swapped to dim 1
                                           in_shape[2].is_static() && in_shape[2].get_length() == 1;  // kv_heads==1
        if (!is_mqa_fake_transpose) {
            return nullptr;
        }
        LOG_DEBUG("MQA: matched 'transpose' is a Reshape (num_kv_heads==1); replacing with Transpose");
        auto data_input = node->input_value(0);
        auto placeholder_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 1, 3});
        auto new_transpose = std::make_shared<ov::op::v1::Transpose>(data_input, placeholder_order);
        new_transpose->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, new_transpose);
        ov::replace_node(node, new_transpose);
        return new_transpose;
    }

    // generic part of matchers, to transpose v-tensors, and concat
    void transpose_v_cache(Context::Ref ctx,
                           const std::shared_ptr<ov::op::v0::Parameter>& matched_param,
                           const std::shared_ptr<ov::op::v0::Concat>& matched_concat,
                           const std::shared_ptr<ov::op::v1::Transpose>& matched_transpose) {
        // NB: The same param->concat pair may be matched multiple times when the
        // V-concat output feeds more than one downstream branch (e.g. shared KV-cache
        // across attention layers in Gemma4).  Guard the shared-state mutations so they
        // are applied exactly once.
        if (matched_concat->get_axis() != 3u) {
            if (ctx.get().transposed_params.insert(matched_param.get()).second) {
                auto param_shape = matched_param->get_partial_shape();
                NPUW_ASSERT(param_shape.size() == 4u);
                // NB: Transpose Parameter that correspond to V-tensor it will
                // speed-up its multiplication with attention scores
                std::swap(param_shape[2], param_shape[3]);
                matched_param->set_partial_shape(param_shape);
            }

            auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
            matched_transpose->set_argument(1, order_cst);
            matched_concat->set_axis(3u);
            for (size_t i = 0; i < matched_concat->get_input_size(); ++i) {
                const auto input = matched_concat->input_value(i);
                if (input.get_node_shared_ptr() == matched_param) {
                    ctx.get().track(matched_param->output(0));
                } else if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(input.get_node_shared_ptr())) {
                    if (convert->input_value(0).get_node_shared_ptr() == matched_param) {
                        ctx.get().track(matched_param->output(0));
                        ctx.get().track(convert->output(0));
                    }
                }
            }
            ctx.get().track(matched_transpose->output(0));
        }
        ctx.get().track(matched_concat->output(0));
        ctx.get().bTransposed = true;
    }

    // to transpose v-tensors, and concat, and update matmul args; per-branch matmul transpose_b is always set
    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::op::v0::Parameter>& matched_param,
                            const std::shared_ptr<ov::op::v0::Concat>& matched_concat,
                            const std::shared_ptr<ov::op::v1::Transpose>& matched_transpose,
                            const std::shared_ptr<ov::op::v0::MatMul>& matched_matmul) {
        transpose_v_cache(ctx, matched_param, matched_concat, matched_transpose);
        matched_matmul->set_transpose_b(true);
    }

    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::Node>& node_param,
                            const std::shared_ptr<ov::Node>& node_concat,
                            const std::shared_ptr<ov::Node>& node_transpose,
                            const std::shared_ptr<ov::Node>& node_matmul) {
        auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(node_param);
        auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(node_concat);
        auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(node_transpose);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(node_matmul);

        transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
    }
};

class TransposeDirectValueTensorsPrefill : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeDirectValueTensorsPrefill");
    explicit TransposeDirectValueTensorsPrefill(TransposeValueTensors::Context::Ref ctx) {
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, transpose});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& node_to_output = m.get_pattern_value_map();
            const auto matched_transpose =
                ov::as_type_ptr<ov::op::v1::Transpose>(node_to_output.at(transpose).get_node_shared_ptr());
            const auto matched_matmul =
                ov::as_type_ptr<ov::op::v0::MatMul>(node_to_output.at(matmul).get_node_shared_ptr());

            if (matched_transpose == nullptr || matched_matmul == nullptr) {
                return false;
            }

            const auto shape = matched_transpose->get_output_partial_shape(0);
            if (shape.rank().is_dynamic() || shape.rank().get_length() != 4) {
                return false;
            }

            const auto order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
            matched_transpose->set_argument(1, order);
            matched_matmul->set_transpose_b(true);
            ctx.get().track(matched_transpose->output(0));
            ctx.get().bTransposed = true;
            LOG_DEBUG("vtensors transposed: Whisper cross-attention prefill pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeDirectValueTensorsPrefill"),
                         std::move(callback));
    }
};

class TransposeDirectValueTensorsGenerate : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeDirectValueTensorsGenerate");
    explicit TransposeDirectValueTensorsGenerate(TransposeValueTensors::Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, convert});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& node_to_output = m.get_pattern_value_map();
            const auto matched_param =
                ov::as_type_ptr<ov::op::v0::Parameter>(node_to_output.at(param).get_node_shared_ptr());
            const auto matched_matmul =
                ov::as_type_ptr<ov::op::v0::MatMul>(node_to_output.at(matmul).get_node_shared_ptr());

            if (matched_param == nullptr || matched_matmul == nullptr) {
                return false;
            }

            const auto shape = matched_param->get_partial_shape();
            if (shape.rank().is_dynamic() || shape.rank().get_length() != 4) {
                return false;
            }

            if (ctx.get().transposed_params.insert(matched_param.get()).second) {
                auto transposed_shape = shape;
                std::swap(transposed_shape[2], transposed_shape[3]);
                matched_param->set_partial_shape(transposed_shape);
                ctx.get().track(matched_param->output(0));
            }
            const auto matched_convert = node_to_output.find(convert);
            if (matched_convert != node_to_output.end()) {
                ctx.get().track(matched_convert->second);
            }
            matched_matmul->set_transpose_b(true);
            ctx.get().bTransposed = true;
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeDirectValueTensorsGenerate"),
                         std::move(callback));
    }
};

// MHA (Multi-Head Attention) pattern for value tensor concatenation
class TransposeValueTensors_MHA : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_MHA");
    TransposeValueTensors_MHA(Context::Ref ctx) {
        register_matcher_mha(ctx);
    }

private:
    void register_matcher_mha(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        // Softmax output maybe sliced when SDPA with sink input is decomposed
        auto maybe_slice = opp::optional<ov::op::v8::Slice>(
            {softmax, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({maybe_slice, concat});

        auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();

            transpose_matmul_b(ctx,
                               matched_node_param,
                               matched_node_concat,
                               matched_node_transpose,
                               matched_node_matmul);
            LOG_DEBUG("vtensors transposed: MHA pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_MHA"), std::move(callback));
    }
};

// GQA (Grouped Query Attention) pattern for value tensors with broadcasting
// Used by llama3, phi3, mistral, GPT-OSS, etc.
class TransposeValueTensors_GQA : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_GQA");
    TransposeValueTensors_GQA(Context::Ref ctx) {
        register_matcher_gqa(ctx);
    }

private:
    void register_matcher_gqa(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        // MQA: aten::transpose on a dim-1 axis (num_kv_heads==1) is optimized by the
        // frontend into a Reshape instead of a real Transpose, so match both.
        auto transpose =
            opp::wrap_type<ov::op::v1::Transpose, ov::op::v1::Reshape>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});

        // only difference is that broadcast wrapped into unsquese/reshape, while transposed tensor didn't change
        const auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({concat, unsqueeze_axes});
        auto broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({unsqueeze, opp::any_input()});
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({broadcast, opp::any_input()});

        // v8 softmax? what? can be other softmaxes
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        // Softmax output maybe sliced when SDPA with sink input is decomposed (e.g. GPT-OSS)
        auto maybe_slice = opp::optional<ov::op::v8::Slice>(
            {softmax, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({maybe_slice, reshape});

        auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            auto matched_node_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
            auto matched_node_unsqueeze_axes = node_to_output.at(unsqueeze_axes).get_node_shared_ptr();
            auto matched_node_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
            auto matched_node_reshape = node_to_output.at(reshape).get_node_shared_ptr();

            matched_node_transpose = to_real_transpose(matched_node_transpose);
            if (!matched_node_transpose) {
                return false;
            }

            auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_param);
            auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(matched_node_concat);
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(matched_node_transpose);
            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(matched_node_unsqueeze);
            auto matched_broadcast = std::static_pointer_cast<ov::op::v3::Broadcast>(matched_node_broadcast);
            auto matched_reshape = std::static_pointer_cast<ov::op::v1::Reshape>(matched_node_reshape);

            auto shape_broadcast = matched_broadcast->get_output_shape(0);
            NPUW_ASSERT(shape_broadcast.size() == 5u);
            std::swap(shape_broadcast[3], shape_broadcast[4]);

            LOG_DEBUG("shape_broadcast for: " << matched_broadcast->get_friendly_name()
                                              << ", shape=" << shape_broadcast);

            const auto broadcast_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, shape_broadcast);
            broadcast_axes_node->set_friendly_name(matched_broadcast->get_friendly_name() + "/new_broadcast_shape");
            matched_broadcast->input(1).replace_source_output(broadcast_axes_node);

            auto shape_reshape = matched_reshape->get_output_shape(0);
            NPUW_ASSERT(shape_reshape.size() == 4u);
            std::swap(shape_reshape[2], shape_reshape[3]);

            LOG_DEBUG("shape_reshape for: " << matched_reshape->get_friendly_name() << ", shape=" << shape_reshape);

            const auto reshape_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, shape_reshape);
            reshape_axes_node->set_friendly_name(matched_reshape->get_friendly_name() + "/new_reshape_shape");
            matched_reshape->input(1).replace_source_output(reshape_axes_node);

            transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
            ctx.get().track(matched_unsqueeze->output(0));
            ctx.get().track(matched_broadcast->output(0));
            ctx.get().track(matched_reshape->output(0));
            LOG_DEBUG("vtensors transposed: GQA pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_GQA"), std::move(callback));
    }
};

// Keeps SDPA intact: V-tensors are transposed the same way as in the MHA/GQA patterns, and an explicit
// Transpose right before SDPA.V restores the [B, H, S, D] layout that SDPA expects. The GQA broadcast
// chain works on the transposed layout, so the Transpose stays adjacent to SDPA (where the compiler can
// fuse it into the attention kernel, and the SDPAToPagedAttention pattern can match it):
//   param -> [convert] -> concat -> [unsqueeze -> broadcast -> reshape] -> Transpose{0,1,3,2} -> SDPA.V
class TransposeValueTensors_SDPA : public TransposeValueTensors {
    static std::shared_ptr<ov::Node> make_restore(const ov::Output<ov::Node>& input, const std::string& name) {
        auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 1, 3, 2});
        auto restore = std::make_shared<ov::op::v1::Transpose>(input, order_cst);
        restore->set_friendly_name(name + "/vt_restore");
        ov::copy_runtime_info(input.get_node_shared_ptr(), restore);
        return restore;
    }

    static bool has_single_consumer(const ov::Node* node) {
        return node->get_output_size() == 1u && node->output(0).get_target_inputs().size() == 1u;
    }

    // Target shape with the last two dims swapped, nullptr if it can't be done. A computed target gets a
    // Gather on top (a ShapeOf of the V-concat inside it is restored by restore_shape_of_consumers()).
    static std::shared_ptr<ov::Node> swap_last_dims(const ov::Output<ov::Node>& target, bool special_zero) {
        const auto& target_shape = target.get_partial_shape();
        if (target_shape.rank() != ov::Rank(1) || target_shape[0].is_dynamic() || target_shape[0].get_length() < 2) {
            return nullptr;
        }
        const auto n = static_cast<size_t>(target_shape[0].get_length());
        if (auto cst = ov::as_type_ptr<ov::op::v0::Constant>(target.get_node_shared_ptr())) {
            auto dims = cst->cast_vector<int64_t>();
            // NB: with special_zero 0 copies the input dim at the same index, so it can't be moved
            if (special_zero && (dims[n - 1] == 0 || dims[n - 2] == 0)) {
                return nullptr;
            }
            std::swap(dims[n - 1], dims[n - 2]);
            return ov::op::v0::Constant::create(cst->get_element_type(), cst->get_shape(), dims);
        }
        if (special_zero) {  // positions of the 0s are unknown
            return nullptr;
        }
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[n - 1], order[n - 2]);
        auto gather = std::make_shared<ov::op::v8::Gather>(
            target,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{n}, order),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        gather->set_friendly_name(target.get_node()->get_friendly_name() + "/transposed_layout");
        ov::copy_runtime_info(target.get_node_shared_ptr(), gather);
        return gather;
    }

    // GQA/MQA: concat -> unsqueeze -> broadcast -> reshape. Keeps the chain on the transposed [.., D, S]
    // layout (swaps the last two dims of the broadcast/reshape target shapes) and puts the restoring
    // Transpose after the reshape. Returns false (the graph is untouched) if the chain doesn't fit.
    static bool restore_after_broadcast(ov::Node* node) {
        auto unsqueeze = ov::as_type<ov::op::v0::Unsqueeze>(node);
        if (!unsqueeze || !has_single_consumer(unsqueeze) ||
            unsqueeze->get_output_partial_shape(0).rank() != ov::Rank(5)) {
            return false;
        }
        auto axes = ov::as_type<ov::op::v0::Constant>(unsqueeze->get_input_node_ptr(1));
        if (!axes || ov::shape_size(axes->get_shape()) != 1u) {
            return false;
        }
        auto axis = axes->cast_vector<int64_t>()[0];
        if (axis < 0) {
            axis += 5;
        }
        if (axis < 0 || axis > 2) {  // the last two dims (D, S) must stay the last two
            return false;
        }

        auto broadcast =
            ov::as_type<ov::op::util::BroadcastBase>(unsqueeze->output(0).get_target_inputs().begin()->get_node());
        if (!broadcast || broadcast->get_input_size() != 2u || !has_single_consumer(broadcast)) {
            return false;
        }
        const auto mode = broadcast->get_broadcast_spec().m_type;
        if (mode != ov::op::BroadcastType::NUMPY && mode != ov::op::BroadcastType::BIDIRECTIONAL) {
            return false;
        }
        auto new_broadcast_target = swap_last_dims(broadcast->input_value(1), false);

        auto reshape = ov::as_type<ov::op::v1::Reshape>(broadcast->output(0).get_target_inputs().begin()->get_node());
        if (!new_broadcast_target || !reshape || reshape->get_output_partial_shape(0).rank() != ov::Rank(4)) {
            return false;
        }
        auto new_reshape_target = swap_last_dims(reshape->input_value(1), reshape->get_special_zero());
        if (!new_reshape_target) {
            return false;
        }

        broadcast->set_argument(1, new_broadcast_target);
        reshape->set_argument(1, new_reshape_target);
        auto restore = make_restore(reshape->output(0), reshape->get_friendly_name());
        for (auto&& in : reshape->output(0).get_target_inputs()) {
            if (in.get_node() != restore.get()) {
                in.replace_source_output(restore);
            }
        }
        return true;
    }

public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_SDPA");
    TransposeValueTensors_SDPA(Context::Ref ctx) {
        auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto matched_sdpa = m.get_match_root();

            // Walk back from SDPA.V through the optional GQA broadcast chain to the V-concat
            auto v_node = matched_sdpa->input_value(2).get_node_shared_ptr();
            if (ov::is_type<ov::op::v1::Reshape>(v_node)) {
                v_node = v_node->input_value(0).get_node_shared_ptr();
                if (!ov::is_type<ov::op::v1::Broadcast>(v_node) && !ov::is_type<ov::op::v3::Broadcast>(v_node)) {
                    return false;
                }
                v_node = v_node->input_value(0).get_node_shared_ptr();
                if (!ov::is_type<ov::op::v0::Unsqueeze>(v_node)) {
                    return false;
                }
                v_node = v_node->input_value(0).get_node_shared_ptr();
            }
            auto matched_concat = ov::as_type_ptr<ov::op::v0::Concat>(v_node);
            if (!matched_concat || matched_concat->get_input_size() != 2u) {
                return false;
            }

            auto param_node = matched_concat->input_value(0).get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::Convert>(param_node)) {
                param_node = param_node->input_value(0).get_node_shared_ptr();
            }
            auto matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(param_node);
            if (!matched_param || matched_param->get_partial_shape().size() != 4u) {
                return false;
            }

            auto transpose_node = matched_concat->input_value(1).get_node_shared_ptr();
            if (!ov::is_type<ov::op::v1::Transpose>(transpose_node) &&
                !ov::is_type<ov::op::v1::Reshape>(transpose_node)) {
                return false;
            }
            transpose_node = to_real_transpose(transpose_node);
            if (!transpose_node) {
                return false;
            }
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(transpose_node);

            transpose_v_cache(ctx, matched_param, matched_concat, matched_transpose);

            // NB: Only Results (and Assigns) keep the transposed V-cache layout, all other consumers
            // (incl. SDPAs sharing this concat) are fed with the restored layout: GQA broadcast chains get
            // the Transpose at their end, anything else a shared one right after the concat.
            // ShapeOf consumers are fixed by restore_shape_of_consumers() (the concat output is tracked).
            std::shared_ptr<ov::Node> concat_restore;
            for (auto&& in : matched_concat->output(0).get_target_inputs()) {
                auto consumer = in.get_node();
                if (ov::is_type<ov::op::v0::Result>(consumer) || ov::is_type<ov::op::v6::Assign>(consumer) ||
                    ov::is_type<ov::op::v0::ShapeOf>(consumer) || ov::is_type<ov::op::v3::ShapeOf>(consumer)) {
                    continue;
                }
                if (restore_after_broadcast(consumer)) {
                    LOG_DEBUG("vtensors transposed: SDPA pattern, explicit transpose after the broadcast chain of "
                              << consumer->get_friendly_name());
                    continue;
                }
                if (!concat_restore) {
                    concat_restore = make_restore(matched_concat->output(0), matched_concat->get_friendly_name());
                    LOG_DEBUG("vtensors transposed: SDPA pattern, explicit transpose "
                              << concat_restore->get_friendly_name());
                }
                in.replace_source_output(concat_restore);
            }
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(sdpa, "TransposeValueTensors_SDPA"), std::move(callback));
    }
};

class ScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::ScaledDotProductAttentionDecomposition");
    explicit ScaledDotProductAttentionDecomposition(bool use_high_precision_on_add) {
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            auto new_output_node = decompose(node, use_high_precision_on_add);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, "ScaledDotProductAttentionDecomposition");
        register_matcher(m, std::move(callback));
    }
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node,
                                        bool use_high_precision_on_add) {
        using namespace ov::op;
        using namespace ov;
        auto query = node->input_value(0);
        auto key = node->input_value(1);
        auto value = node->input_value(2);
        auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
        auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
        auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
        auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
        auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
        auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

        Output<Node> scale;
        if (node->get_input_size() < 5) {
            scale = register_new_node<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
            scale = register_new_node<v1::ConvertLike>(scale, query);
            auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
            scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
        } else {
            scale = node->input_value(4);
        }

        auto q_scaled = register_new_node<v1::Multiply>(query, scale);
        auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
        auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
        auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
        k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
        auto minus_inf =
            register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
                ->output(0);
        auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
        auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

        auto scaled_atten = register_new_node<v0::MatMul>(q_scaled, key, false, true)->output(0);
        minus_inf = register_new_node<v1::ConvertLike>(minus_inf, scaled_atten);

        if (node->get_causal() || node->get_input_size() > 3) {
            Output<Node> mask;
            Output<Node> atten_mask;
            if (!node->get_causal()) {
                mask = node->input_value(3);

                // two types of masks are supported. A boolean mask where a value of True indicates that the element
                // should take part in attention. A float mask of the same type as query, key, value that is added to
                // the attention score.
                if (mask.get_element_type() == element::boolean) {
                    atten_mask = register_new_node<v1::Select>(mask, zero_f, minus_inf);
                } else {
                    atten_mask = mask;
                }
            } else {
                auto target_s_len = register_new_node<v8::Gather>(q_shape, minus_two, zero_i);
                auto source_s_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
                auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
                auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
                auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
                mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
                auto horizontal_range =
                    register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
                horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
                auto stop = register_new_node<v1::Add>(target_s_len, one_i);
                auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
                vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
                auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
                atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
            }
            if (use_high_precision_on_add) {
                npuw::util::HighPrecisionAttr attr_hp;
                attr_hp.compute_precision_type = ov::element::f32;
                atten_mask.get_rt_info()[npuw::util::HighPrecisionAttr::get_type_info_static()] = attr_hp;
            }

            scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
        }

        scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
        auto result = register_new_node<v0::MatMul>(scaled_atten, value);
        result->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, get_new_nodes());
        return result;
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

bool ov::npuw::util::OptimizeValueTensors::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;

    // Check if any SDPA has sink input
    bool has_sdpa_with_sink = false;
    for (const auto& op : model->get_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
            if (sdpa->get_input_size() == 6) {
                has_sdpa_with_sink = true;
                break;
            }
        }
    }

    TransposeValueTensors::Context ctx;
    const bool keep_sdpa = !m_decompose_sdpa && !m_is_whisper && !has_sdpa_with_sink;
    if (!m_decompose_sdpa && !keep_sdpa) {
        LOG_WARN("NPUW_LLM_DECOMPOSE_SDPA=NO is not supported for " << (m_is_whisper ? "Whisper" : "SDPA with sink")
                                                                    << ", SDPA will be decomposed");
    }

    if (keep_sdpa) {
        rewr.add_matcher<TransposeValueTensors_SDPA>(std::ref(ctx));
    } else if (has_sdpa_with_sink) {
        // TODO:  evaluate performance by older npu-drivers
        rewr.add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    } else {
        rewr.add_matcher<ScaledDotProductAttentionDecomposition>(m_is_prefill);
    }

    rewr.add_matcher<TransposeValueTensors_MHA>(std::ref(ctx));
    rewr.add_matcher<TransposeValueTensors_GQA>(std::ref(ctx));
    if (m_is_whisper) {
        if (m_is_prefill) {
            rewr.add_matcher<TransposeDirectValueTensorsPrefill>(std::ref(ctx));
        } else {
            rewr.add_matcher<TransposeDirectValueTensorsGenerate>(std::ref(ctx));
        }
    }

    rewr.run_on_model(model);

    ctx.restore_shape_of_consumers();

    ov::pass::Validate().run_on_model(model);

    // NB: matmul parameters gets transposed, if pass applied
    return ctx.bTransposed;
}
