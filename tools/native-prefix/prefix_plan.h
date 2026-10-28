// SPDX-License-Identifier: MIT
// Serial, exact-token prefix reuse. No logits or answers are cached.
#ifndef JEV_NATIVE_PREFIX_PLAN_H
#define JEV_NATIVE_PREFIX_PLAN_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace s1_prefix {

struct geometry {
    size_t chunk = 0;
    size_t batch = 0;
    size_t microbatch = 0;
};

struct request_view {
    const std::vector<int32_t> & tokens;
    const std::vector<int32_t> & slots;
    const std::vector<int32_t> & counts;
    bool single_text;
};

struct plan {
    bool enabled = false;
    bool eligible = false;
    size_t previous_input_tokens = 0;
    size_t exact_common_prefix_tokens = 0;
    size_t unaligned_retained_token_cap = 0;
    size_t retained = 0;
    const char * reason = "disabled";
};

// This transaction owns only token identity/geometry. Memory is owned by the
// runtime. The small interface is also used by the offline failure tests.
class cache {
    std::vector<int32_t> previous_;
    geometry previous_geometry_;

public:
    bool valid() const { return !previous_.empty(); }

    void invalidate() noexcept {
        previous_.clear();
        previous_geometry_ = {};
    }

    template<class Memory>
    void reset(Memory & memory) noexcept {
        invalidate();
        memory.clear();
    }

    template<class Memory>
    plan begin(const request_view & request, bool enabled, geometry native, Memory & memory) {
        plan out;
        out.enabled = enabled;
        out.previous_input_tokens = previous_.size();
        out.eligible = request.single_text && !request.tokens.empty() &&
            request.slots.size() == 1 && request.counts.size() == 1 && request.counts[0] == 2 &&
            request.slots[0] >= 0 && size_t(request.slots[0]) == request.tokens.size() - 1 &&
            native.chunk > 0 && native.batch == native.chunk && native.microbatch == native.chunk;
        if (!enabled) {
            out.reason = "disabled";
        } else if (!out.eligible) {
            out.reason = "unsupported_input_or_batch_geometry";
        } else if (previous_.empty()) {
            out.reason = "cold_or_invalidated";
        } else if (previous_geometry_.chunk != native.chunk || previous_geometry_.batch != native.batch ||
                   previous_geometry_.microbatch != native.microbatch) {
            out.reason = "batch_geometry_changed";
        } else {
            const size_t bound = std::min(previous_.size(), request.tokens.size());
            while (out.exact_common_prefix_tokens < bound &&
                   previous_[out.exact_common_prefix_tokens] == request.tokens[out.exact_common_prefix_tokens]) {
                ++out.exact_common_prefix_tokens;
            }
            // The previous last chunk requested logits too. If an input grows
            // across a batch boundary, replay that chunk instead of retaining
            // KV produced with a different requested-output mask/graph shape.
            out.unaligned_retained_token_cap = std::min(
                {out.exact_common_prefix_tokens, size_t(request.slots[0]), previous_.size() - 1});
            // Preserve the original per-part decode chunks. In particular an
            // identical input must replay the original last chunk, not one token.
            out.retained = out.unaligned_retained_token_cap / native.chunk * native.chunk;
            out.reason = out.retained ? "exact_prefix_batch_aligned" : "no_complete_batch_prefix";
        }

        // A failed/incomplete request can never commit the previous identity.
        invalidate();
        if (out.retained) {
            if (memory.min_position() != 0 || memory.max_position() != int64_t(out.previous_input_tokens) - 1) {
                out.reason = "incomplete_previous_kv";
                out.retained = 0;
            } else if (!memory.remove_from(out.retained)) {
                out.reason = "kv_suffix_removal_failed";
                out.retained = 0;
            } else if (memory.min_position() != 0 || memory.max_position() != int64_t(out.retained) - 1) {
                out.reason = "incomplete_retained_kv";
                out.retained = 0;
            }
        }
        if (!out.retained) memory.clear();
        return out;
    }

    template<class Memory>
    void commit(const request_view & request, const plan & prepared, geometry native, Memory & memory) {
        invalidate();
        if (prepared.enabled && prepared.eligible && memory.min_position() == 0 &&
            memory.max_position() == int64_t(request.tokens.size()) - 1) {
            previous_ = request.tokens;
            previous_geometry_ = native;
        }
    }
};

} // namespace s1_prefix
#endif
