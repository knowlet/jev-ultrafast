// SPDX-License-Identifier: MIT
// Offline memory/geometry contracts. These do not run llama.cpp or any model.
#include "prefix_plan.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

static size_t checks = 0;
static void check(bool value, const char * message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

struct memory {
    int64_t first = -1, last = -1;
    size_t clears = 0, removals = 0;
    bool removal_ok = true, corrupt_removal = false;
    void clear() noexcept { ++clears; first = last = -1; }
    bool remove_from(size_t position) {
        ++removals;
        last = int64_t(position) - 1;
        if (corrupt_removal) first = 1;
        return removal_ok;
    }
    int64_t min_position() const { return first; }
    int64_t max_position() const { return last; }
    void decoded(size_t count) { first = 0; last = int64_t(count) - 1; }
};

using chunk = std::pair<size_t, size_t>;
static std::vector<chunk> chunks(size_t length, size_t batch, size_t start = 0) {
    std::vector<chunk> out;
    for (size_t position = start; position < length; position += batch) {
        out.emplace_back(position, std::min(batch, length - position));
    }
    return out;
}

static s1_prefix::request_view view(const std::vector<int32_t> & tokens,
                                   const std::vector<int32_t> & slots,
                                   const std::vector<int32_t> & counts, bool single_text = true) {
    return {tokens, slots, counts, single_text};
}

static void geometry_regressions() {
    const std::vector<int32_t> counts{2};
    // Exhaustive modest input lengths, every possible exact LCP, variable chunk
    // sizes and actual final slots. Check actual decode chunk sequences against
    // the original full-decode schedule, not merely the alignment formula.
    for (size_t batch : {size_t(1), size_t(2), size_t(3), size_t(8), size_t(16)}) {
        for (size_t length = 1; length <= 40; ++length) {
            for (size_t common = 0; common <= length; ++common) {
                s1_prefix::cache cache;
                memory mem;
                const s1_prefix::geometry shape{batch, batch, batch};
                std::vector<int32_t> original(length, 7), next = original;
                const std::vector<int32_t> slots{int32_t(length - 1)};
                auto initial = cache.begin(view(original, slots, counts), true, shape, mem);
                mem.decoded(length);
                cache.commit(view(original, slots, counts), initial, shape, mem);
                if (common < length) next[common] = 8;
                auto plan = cache.begin(view(next, slots, counts), true, shape, mem);
                auto full = chunks(length, batch);
                auto suffix = chunks(length, batch, plan.retained);
                std::vector<chunk> expected;
                for (const auto & item : full) if (item.first >= plan.retained) expected.push_back(item);
                check(suffix == expected, "suffix changed original decode chunk geometry");
                check(plan.retained <= common, "retained a changed token");
                check(plan.retained <= size_t(slots[0]), "retained a requested logit");
                check(!suffix.empty(), "exact repeat skipped requested logits");
                check(suffix.back() == full.back(), "exact final chunk changed");
                check(!cache.valid(), "incomplete decode committed cache identity");
            }
        }
    }
    for (size_t length : {size_t(511), size_t(512), size_t(513), size_t(580), size_t(590),
                         size_t(591), size_t(601), size_t(1024), size_t(1025)}) {
        s1_prefix::cache cache;
        memory mem;
        const s1_prefix::geometry shape{512, 512, 512};
        std::vector<int32_t> tokens(length, 9), slots{int32_t(length - 1)};
        auto first = cache.begin(view(tokens, slots, counts), true, shape, mem);
        mem.decoded(length);
        cache.commit(view(tokens, slots, counts), first, shape, mem);
        auto repeated = cache.begin(view(tokens, slots, counts), true, shape, mem);
        check(chunks(length, 512, repeated.retained).back() == chunks(length, 512).back(),
              "512-token boundary regression");
        check(repeated.retained < length, "repeat reused output instead of decoding");
    }
    for (const auto & extension : std::vector<std::pair<size_t, size_t>>{{512,601}, {1024,1100}}) {
        s1_prefix::cache cache;
        memory mem;
        const s1_prefix::geometry shape{512, 512, 512};
        std::vector<int32_t> before(extension.first, 7), after(extension.second, 7);
        std::vector<int32_t> old_slots{int32_t(before.size() - 1)}, new_slots{int32_t(after.size() - 1)};
        auto first = cache.begin(view(before, old_slots, counts), true, shape, mem);
        mem.decoded(before.size());
        cache.commit(view(before, old_slots, counts), first, shape, mem);
        auto plan = cache.begin(view(after, new_slots, counts), true, shape, mem);
        check(plan.retained == before.size() - 512, "retained previous requested-logit chunk after input grew");
        check(plan.retained <= size_t(old_slots[0]), "previous output slot entered retained prefix");
    }
}

static void failure_regressions() {
    const s1_prefix::geometry shape{512, 512, 512};
    std::vector<int32_t> tokens(601, 7), slots{600}, counts{2};
    for (int scenario = 0; scenario < 15; ++scenario) {
        s1_prefix::cache cache;
        memory mem;
        auto first = cache.begin(view(tokens, slots, counts), true, shape, mem);
        mem.decoded(tokens.size());
        cache.commit(view(tokens, slots, counts), first, shape, mem);
        check(cache.valid(), "completed request was not committed");
        auto test_slots = slots, test_counts = counts, test_tokens = tokens;
        auto test_shape = shape;
        bool enabled = true, single_text = true;
        switch (scenario) {
            case 0: enabled = false; break;
            case 1: single_text = false; break;
            case 2: test_slots = {}; break;
            case 3: test_slots = {-1}; break;
            case 4: test_slots = {599}; break;
            case 5: test_slots = {600, 600}; test_counts = {2, 2}; break;
            case 6: test_counts = {3}; break;
            case 7: test_shape = {0, 0, 0}; break;
            case 8: test_shape.microbatch = 256; break;
            case 9: test_shape = {256, 256, 256}; break;
            case 10: mem.removal_ok = false; break;
            case 11: mem.corrupt_removal = true; break;
            case 12: mem.first = 1; break;
            case 13: mem.last = 500; break;
            case 14: test_tokens.clear(); break;
        }
        const size_t before = mem.clears;
        auto plan = cache.begin(view(test_tokens, test_slots, test_counts, single_text), enabled, test_shape, mem);
        check(plan.retained == 0, "unsupported/broken cache did not fall back");
        check(mem.clears == before + 1 && mem.first == -1 && mem.last == -1,
              "fallback failed to clear real memory");
        check(!cache.valid(), "fallback kept stale cache metadata");
        // Simulate a partial decode. A failure must reset KV and metadata, and
        // the next valid request must decode from zero.
        mem.decoded(512);
        cache.reset(mem);
        mem.removal_ok = true;
        mem.corrupt_removal = false;
        auto recovered = cache.begin(view(tokens, slots, counts), true, shape, mem);
        check(recovered.retained == 0, "partial-decode failure poisoned next request");
        mem.decoded(tokens.size());
        cache.commit(view(tokens, slots, counts), recovered, shape, mem);
        auto warm = cache.begin(view(tokens, slots, counts), true, shape, mem);
        check(warm.retained == 512, "valid request after failure could not warm cache");
    }

    s1_prefix::cache cache;
    memory mem;
    auto first = cache.begin(view(tokens, slots, counts), true, shape, mem);
    mem.decoded(512);
    cache.commit(view(tokens, slots, counts), first, shape, mem);
    check(!cache.valid(), "incomplete memory was committed");
}

static void replay() {
    s1_prefix::cache cache;
    memory mem;
    size_t length;
    while (std::cin >> length) {
        int32_t slot;
        size_t batch;
        std::cin >> slot >> batch;
        if (!std::cin || batch == 0 || length > 100000) throw std::runtime_error("invalid replay header");
        std::vector<int32_t> tokens(length), slots{slot}, counts{2};
        for (auto & token : tokens) std::cin >> token;
        if (!std::cin) throw std::runtime_error("incomplete replay tokens");
        const s1_prefix::geometry shape{batch, batch, batch};
        auto plan = cache.begin(view(tokens, slots, counts), true, shape, mem);
        std::cout << plan.exact_common_prefix_tokens << ' ' << plan.retained << ' ' << length - plan.retained;
        for (const auto & item : chunks(length, batch, plan.retained)) {
            std::cout << ' ' << item.first << ':' << item.second;
        }
        std::cout << '\n';
        mem.decoded(length);
        cache.commit(view(tokens, slots, counts), plan, shape, mem);
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--replay") { replay(); return 0; }
        geometry_regressions();
        failure_regressions();
        std::cout << "offline prefix memory/geometry checks: " << checks << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
