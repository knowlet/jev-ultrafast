// SPDX-License-Identifier: MIT
// Execute the actual scorer/JSON loop with a fake C API. No inference library,
// model, GPU, network, tokenizer, or real numerical result is used by this test.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

struct llama_model {};
struct llama_context {};
struct llama_vocab {};
struct mtmd_context {};
struct mtmd_input_chunks {};
struct mtmd_input_chunk {};
struct mtmd_bitmap {};

namespace fake {
llama_model model;
llama_context context;
llama_vocab vocab;
mtmd_context media_context;
mtmd_input_chunks input_chunks;
mtmd_input_chunk input_chunk;
mtmd_bitmap bitmap;
llama_context_params params{};
int64_t first = -1, last = -1;
int clear_count = 0, decode_count = 0, fail_decode = 0;
bool remove_ok = true, missing_logits = false, nonfinite_logits = false;
std::vector<std::pair<int, int>> chunks;
std::vector<int> requested_positions;
std::vector<int32_t> decoded_tokens;
std::array<float, 256> logits{};

void reset() {
    first = last = -1;
    clear_count = decode_count = fail_decode = 0;
    remove_ok = true;
    missing_logits = nonfinite_logits = false;
    chunks.clear();
    requested_positions.clear();
    decoded_tokens.clear();
}
void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
} // namespace fake

extern "C" {
void ggml_backend_load_all() {}
void llama_backend_init() {}
llama_model_params llama_model_default_params() { return {}; }
llama_context_params llama_context_default_params() { return {}; }
llama_model * llama_model_load_from_file(const char *, llama_model_params) { return &fake::model; }
void llama_model_free(llama_model *) {}
const llama_vocab * llama_model_get_vocab(const llama_model *) { return &fake::vocab; }
llama_context * llama_init_from_model(llama_model *, llama_context_params values) {
    fake::params = values;
    return &fake::context;
}
void llama_free(llama_context *) {}
uint32_t llama_n_ctx(const llama_context *) { return fake::params.n_ctx; }
uint32_t llama_n_batch(const llama_context *) { return fake::params.n_batch; }
uint32_t llama_n_ubatch(const llama_context *) { return fake::params.n_ubatch; }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return 256; }
int32_t llama_tokenize(const llama_vocab *, const char * text, int32_t length,
                       llama_token * tokens, int32_t capacity, bool, bool) {
    if (length > capacity) return -length;
    for (int32_t i = 0; i < length; ++i) tokens[i] = static_cast<unsigned char>(text[i]);
    return length;
}
llama_memory_t llama_get_memory(const llama_context *) { return nullptr; }
void llama_memory_clear(llama_memory_t, bool) {
    ++fake::clear_count;
    fake::first = fake::last = -1;
}
bool llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos start, llama_pos end) {
    fake::check(end == -1, "wrong half-open suffix range");
    fake::last = start - 1;
    return fake::remove_ok;
}
llama_pos llama_memory_seq_pos_min(llama_memory_t, llama_seq_id) { return fake::first; }
llama_pos llama_memory_seq_pos_max(llama_memory_t, llama_seq_id) { return fake::last; }
void llama_set_causal_attn(llama_context *, bool value) { fake::check(value, "causal attention changed"); }
llama_batch llama_batch_init(int32_t count, int32_t embd, int32_t max_seq) {
    fake::check(embd == 0 && max_seq == 1, "unexpected batch contract");
    llama_batch out{};
    out.n_tokens = count;
    out.token = new llama_token[count];
    out.pos = new llama_pos[count];
    out.n_seq_id = new int32_t[count];
    out.seq_id = new llama_seq_id * [count];
    out.logits = new int8_t[count];
    for (int32_t i = 0; i < count; ++i) out.seq_id[i] = new llama_seq_id[1];
    return out;
}
void llama_batch_free(llama_batch batch) {
    for (int32_t i = 0; i < batch.n_tokens; ++i) delete[] batch.seq_id[i];
    delete[] batch.token;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    delete[] batch.seq_id;
    delete[] batch.logits;
}
int32_t llama_decode(llama_context *, llama_batch batch) {
    ++fake::decode_count;
    fake::chunks.emplace_back(batch.pos[0], batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        fake::check(batch.pos[i] == fake::last + 1, "noncontiguous decode or uncleared stale KV");
        fake::check(batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == 0, "sequence contract changed");
        if (fake::first < 0) fake::first = batch.pos[i];
        fake::last = batch.pos[i];
        fake::decoded_tokens.push_back(batch.token[i]);
        if (batch.logits[i]) fake::requested_positions.push_back(batch.pos[i]);
    }
    // Fail after mutating memory to emulate committed earlier microbatches.
    if (fake::fail_decode == fake::decode_count) return -2;
    fake::logits.fill(float(fake::decode_count));
    return 0;
}
float * llama_get_logits_ith(llama_context *, int32_t index) {
    fake::check(index >= 0 && index < fake::chunks.back().second, "logit index not relative to current batch");
    if (fake::missing_logits) return nullptr;
    if (fake::nonfinite_logits) fake::logits[65] = std::numeric_limits<float>::infinity();
    return fake::logits.data();
}
mtmd_context_params mtmd_context_params_default() { return {}; }
mtmd_context * mtmd_init_from_file(const char *, const llama_model *, mtmd_context_params) {
    return &fake::media_context;
}
void mtmd_free(mtmd_context *) {}
bool mtmd_decode_use_mrope(const mtmd_context *) { return false; }
bool mtmd_support_vision(const mtmd_context *) { return true; }
bool mtmd_support_audio(const mtmd_context *) { return true; }
int mtmd_get_audio_sample_rate(const mtmd_context *) { return 16000; }
const char * mtmd_get_marker(const mtmd_context *) { return "media"; }
mtmd_bitmap * mtmd_bitmap_init(uint32_t, uint32_t, const unsigned char *) { return &fake::bitmap; }
mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t, const float *) { return &fake::bitmap; }
void mtmd_bitmap_free(mtmd_bitmap *) {}
mtmd_input_chunks * mtmd_input_chunks_init() { return &fake::input_chunks; }
size_t mtmd_input_chunks_size(const mtmd_input_chunks *) { return 1; }
const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks *, size_t) { return &fake::input_chunk; }
void mtmd_input_chunks_free(mtmd_input_chunks *) {}
mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk *) { return MTMD_INPUT_CHUNK_TYPE_IMAGE; }
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk *) { return 1; }
llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk *) { return 1; }
int32_t mtmd_tokenize(const mtmd_context *, mtmd_input_chunks *, const mtmd_input_text *,
                      const mtmd_bitmap * const *, size_t) { return 0; }
int32_t mtmd_helper_eval_chunk_single(mtmd_context *, llama_context *, const mtmd_input_chunk *,
                                      llama_pos position, llama_seq_id, int32_t, bool, llama_pos * after) {
    if (fake::first < 0) fake::first = position;
    fake::last = position;
    *after = position + 1;
    return 0;
}
} // extern C

#define main embedded_native_main
#include "s1_prefix.cpp"
#undef main

static json request(size_t length = 601) {
    std::vector<int32_t> letters;
    for (const char * c = LETTERS; *c; ++c) letters.push_back(*c);
    return {{"case_id", "offline-fake"}, {"question_ids", {"candidate"}},
        {"slots", {length - 1}}, {"nopts", {2}}, {"letters", letters},
        {"source_input_token_count", length},
        {"parts", {{{"type", "text"}, {"tokens", std::vector<int32_t>(length, 7)}}}}};
}

static options settings(bool reuse = true) {
    options out;
    out.model = "fake-no-model-access";
    out.mmproj = "fake-no-projector-access";
    out.batch = 512;
    out.ctx = 4096;
    out.reuse_prefix = reuse;
    return out;
}

static void warm_and_fallback() {
    fake::reset();
    runtime engine(settings());
    auto input = request();
    auto first = engine.score(input);
    auto second = engine.score(input);
    fake::check(fake::chunks == std::vector<std::pair<int,int>>{{0,512},{512,89},{512,89}},
                "runtime did not replay original final chunk");
    fake::check(first["raw_logits"] != second["raw_logits"], "runtime reused an answer instead of fresh logits");
    fake::check(fake::requested_positions == std::vector<int>{600,600}, "requested slot was skipped or shifted");
    fake::check(second["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 512, "wrong retention telemetry");
    fake::check(fake::decoded_tokens == std::vector<int32_t>(690, 7), "runtime changed input tokens");

    input["parts"][0]["tokens"][526] = 8;
    auto changed = engine.score(input);
    fake::check(changed["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 512, "changed suffix not recomputed");
    fake::check(fake::decoded_tokens[690 + 14] == 8, "changed token not sent to decode");

    input["parts"] = {{{"type", "text"}, {"tokens", std::vector<int32_t>(300, 7)}},
                      {{"type", "text"}, {"tokens", std::vector<int32_t>(301, 7)}}};
    auto multipart = engine.score(input);
    fake::check(multipart["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 0, "multipart reused KV");
    fake::check(fake::chunks[fake::chunks.size()-2] == std::make_pair(0,300) &&
                fake::chunks.back() == std::make_pair(300,301), "multipart chunk origins changed");
    auto after = engine.score(request());
    fake::check(after["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 0, "unsupported request committed cache");

    fake::remove_ok = false;
    auto failed_removal = engine.score(request());
    fake::check(failed_removal["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 0,
                "failed removal did not fully recompute");
    fake::remove_ok = true;
}

static void failure_recovery() {
    for (int scenario = 0; scenario < 5; ++scenario) {
        fake::reset();
        runtime engine(settings());
        engine.score(request());
        auto bad = request();
        if (scenario == 0) fake::fail_decode = 3;
        if (scenario == 1) fake::missing_logits = true;
        if (scenario == 2) fake::nonfinite_logits = true;
        if (scenario == 3) bad["slots"] = {99999};
        if (scenario == 4) bad.erase("case_id"); // fails after decode during response construction
        bool caught = false;
        try { engine.score(bad); } catch (const std::exception &) { caught = true; }
        fake::check(caught && fake::first == -1 && fake::last == -1, "failure left real KV live");
        fake::fail_decode = 0;
        fake::missing_logits = fake::nonfinite_logits = false;
        auto recovered = engine.score(request());
        fake::check(recovered["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == 0,
                    "failure left cache metadata live");
    }
}

static void input_extension() {
    for (const auto & lengths : std::vector<std::pair<size_t, size_t>>{{512,601}, {1024,1100}}) {
        fake::reset();
        runtime engine(settings());
        engine.score(request(lengths.first));
        auto grown = engine.score(request(lengths.second));
        fake::check(grown["preprocessing"]["prefix_cache"]["retained_kv_tokens"] == lengths.first - 512,
                    "runtime retained old requested-logit chunk after input extension");
        fake::check(fake::requested_positions ==
                    std::vector<int>{int(lengths.first - 1), int(lengths.second - 1)},
                    "extended input changed output slot positions");
    }
}

static void disabled_and_json_error() {
    fake::reset();
    runtime engine(settings(false));
    engine.score(request());
    engine.score(request());
    fake::check(fake::chunks == std::vector<std::pair<int,int>>{{0,512},{512,89},{0,512},{512,89}},
                "default-off path changed full decoding");

    fake::reset();
    std::istringstream input(request().dump() + "\n{broken-json\n" + request().dump() + "\n");
    std::ostringstream output;
    auto * original_input = std::cin.rdbuf(input.rdbuf());
    auto * original_output = std::cout.rdbuf(output.rdbuf());
    const char * args[] = {"test", "--model", "fake", "--mmproj", "fake", "--batch-size", "512", "--reuse-prefix"};
    int result = embedded_native_main(8, const_cast<char **>(args));
    std::cin.rdbuf(original_input);
    std::cout.rdbuf(original_output);
    std::cin.clear();
    fake::check(result == 0, "JSON loop failed to recover");
    fake::check(fake::chunks == std::vector<std::pair<int,int>>{{0,512},{512,89},{0,512},{512,89}},
                "malformed JSON preserved cache");
    std::istringstream responses(output.str());
    std::string line;
    std::getline(responses, line);
    fake::check(json::parse(line)["status"] == "ok", "first JSON response failed");
    std::getline(responses, line);
    fake::check(json::parse(line)["status"] == "error", "malformed JSON was not rejected");
    std::getline(responses, line);
    fake::check(json::parse(line)["status"] == "ok", "post-error JSON response failed");
}

int main() {
    try {
        warm_and_fallback();
        failure_recovery();
        input_extension();
        disabled_and_json_error();
        std::cout << "offline native control-flow checks passed (fake C API; no model)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
