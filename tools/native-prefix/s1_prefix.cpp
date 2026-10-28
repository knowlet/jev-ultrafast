// SPDX-License-Identifier: MIT
// Adapted from System One: Copyright (c) 2026 Rami Mithalouni. See LICENSE and origin.json.
// Reconstructed batch-aligned candidate; not the unavailable October 14 candidate source.
// Persistent, non-generative S1 slot scoring against the pinned llama.cpp C APIs.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include "prefix_plan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using chunks_ptr = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
static constexpr const char * LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static constexpr uint32_t MAX_ANSWER_SLOTS = 64;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<int32_t> integers(const json & value, const std::string & name) {
    require(value.is_array(), name + " must be an array");
    std::vector<int32_t> out;
    for (const auto & v : value) {
        require(v.is_number_integer() && v.get<int64_t>() >= 0 && v.get<int64_t>() <= INT32_MAX,
                name + " must contain nonnegative int32 values");
        out.push_back(v.get<int32_t>());
    }
    return out;
}

struct options {
    std::string model, mmproj;
    bool reuse_prefix = false;
    int ctx = 16384, batch = 8192, gpu_layers = 99, threads = 8, image_max_tokens = 280;
};

static options parse_options(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--reuse-prefix") { out.reuse_prefix = true; continue; }
        if (key == "--help") {
            std::cout << "s1-gguf --model FILE --mmproj FILE [--ctx-size 16384 --batch-size 8192 "
                         "--gpu-layers 99 --threads 8 --image-max-tokens 280 --reuse-prefix]\n";
            std::exit(0);
        }
        require(i + 1 < argc, "missing value for " + key);
        std::string value = argv[++i];
        if (key == "--model") out.model = value;
        else if (key == "--mmproj") out.mmproj = value;
        else if (key == "--ctx-size") out.ctx = std::stoi(value);
        else if (key == "--batch-size") out.batch = std::stoi(value);
        else if (key == "--gpu-layers") out.gpu_layers = std::stoi(value);
        else if (key == "--threads") out.threads = std::stoi(value);
        else if (key == "--image-max-tokens") out.image_max_tokens = std::stoi(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    require(!out.model.empty() && !out.mmproj.empty(), "model and mmproj are required");
    require(out.ctx > 0 && out.ctx <= 16384 && out.batch >= 512 && out.batch <= 16384,
            "context must be 1..16384; batch must be 512..16384 (default 8192; "
            "values below 8192 are research-only for the batch-size sweep)");
    require(out.threads > 0 && out.image_max_tokens > 0, "invalid thread/image token limit");
    return out;
}

struct prepared_part {
    std::vector<llama_token> tokens;
    chunks_ptr chunks{nullptr, mtmd_input_chunks_free};
    const mtmd_input_chunk * media = nullptr;
    size_t count = 0;
};

class runtime {
    options opts;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mm{nullptr, mtmd_free};
    const llama_vocab * vocab = nullptr;
    s1_prefix::cache prefix;

    struct native_memory {
        llama_memory_t memory;
        void clear() noexcept { llama_memory_clear(memory, true); }
        bool remove_from(size_t position) {
            return llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(position), -1);
        }
        int64_t min_position() const { return llama_memory_seq_pos_min(memory, 0); }
        int64_t max_position() const { return llama_memory_seq_pos_max(memory, 0); }
    };

    std::vector<llama_token> tokenize(const std::string & text) const {
        std::vector<llama_token> tokens(text.size() + 8);
        int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
        if (n < 0) {
            tokens.resize(-n);
            n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
        }
        require(n >= 0, "native tokenizer failed");
        tokens.resize(n);
        return tokens;
    }

    void validate_candidates(const std::vector<int32_t> & letters) const {
        require(letters.size() == 52 && std::set<int32_t>(letters.begin(), letters.end()).size() == 52,
                "exactly 52 distinct candidate token IDs are required");
        const std::string boundary = "\nAnswer: (";
        auto before = tokenize(boundary);
        for (size_t i = 0; i < letters.size(); ++i) {
            auto one = tokenize(std::string(1, LETTERS[i]));
            auto expected = before;
            expected.push_back(letters[i]);
            require(one == std::vector<llama_token>{letters[i]} && tokenize(boundary + LETTERS[i]) == expected,
                    "native/HF candidate token or answer boundary mismatch");
        }
    }

    prepared_part prepare_part(const json & part) {
        prepared_part out;
        const auto type = part.at("type").get<std::string>();
        if (type == "text") {
            out.tokens = integers(part.at("tokens"), "text tokens");
            for (auto token : out.tokens) require(token < llama_vocab_n_tokens(vocab), "token outside vocabulary");
            out.count = out.tokens.size();
            return out;
        }
        bitmap_ptr bitmap(nullptr, mtmd_bitmap_free);
        if (type == "image") {
            require(mtmd_support_vision(mm.get()), "mmproj does not support images");
            const int width = part.at("width").get<int>();
            const int height = part.at("height").get<int>();
            require(width > 0 && height > 0 && int64_t(width) * height <= 16000000, "invalid image size");
            const size_t count = size_t(width) * height * 3;
            std::ifstream file(part.at("rgb_path").get<std::string>(), std::ios::binary);
            require(bool(file), "cannot open processed RGB file");
            std::vector<unsigned char> rgb(count);
            file.read(reinterpret_cast<char *>(rgb.data()), count);
            require(size_t(file.gcount()) == count && file.peek() == EOF, "processed RGB file length mismatch");
            bitmap.reset(mtmd_bitmap_init(width, height, rgb.data()));
        } else if (type == "audio") {
            require(mtmd_support_audio(mm.get()) && mtmd_get_audio_sample_rate(mm.get()) == 16000,
                    "mmproj does not support 16 kHz audio");
            const auto samples = part.at("samples").get<std::vector<float>>();
            require(!samples.empty() && samples.size() <= 480000, "audio sample count outside contract");
            for (float value : samples) require(std::isfinite(value) && std::abs(value) <= 1, "invalid PCM sample");
            bitmap.reset(mtmd_bitmap_init_from_audio(samples.size(), samples.data()));
        } else {
            throw std::runtime_error("unsupported input part " + type);
        }
        require(bool(bitmap), "cannot initialize media bitmap");
        out.chunks.reset(mtmd_input_chunks_init());
        const std::string marker = mtmd_get_marker(mm.get());
        const mtmd_input_text text{marker.data(), marker.size(), false, true};
        const mtmd_bitmap * bitmap_raw = bitmap.get();
        require(mtmd_tokenize(mm.get(), out.chunks.get(), &text, &bitmap_raw, 1) == 0,
                "native media preprocessing failed");
        for (size_t i = 0; i < mtmd_input_chunks_size(out.chunks.get()); ++i) {
            auto chunk = mtmd_input_chunks_get(out.chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                require(out.media == nullptr, "one media item unexpectedly produced multiple embedding chunks");
                out.media = chunk;
            }
        }
        require(out.media != nullptr, "native media embedding chunk missing");
        out.count = mtmd_input_chunk_get_n_tokens(out.media);
        require(out.count == part.at("expected_tokens").get<size_t>(), "HF/native media token count mismatch");
        require(mtmd_input_chunk_get_n_pos(out.media) == int64_t(out.count), "unexpected media position semantics");
        if (type == "image") {
            require(out.count <= llama_n_batch(ctx.get()) && out.count <= llama_n_ubatch(ctx.get()),
                    "complete image chunk must fit both batch and microbatch for bidirectional SWA attention");
        }
        return out;
    }

public:
    explicit runtime(options values) : opts(std::move(values)) {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = opts.gpu_layers;
        model.reset(llama_model_load_from_file(opts.model.c_str(), mp));
        require(bool(model), "failed to load GGUF language model");
        vocab = llama_model_get_vocab(model.get());
        auto cp = llama_context_default_params();
        cp.n_ctx = opts.ctx;
        cp.n_batch = opts.batch;
        cp.n_ubatch = opts.batch;
        // Reserve logits for the S1 answer-slot bound, rather than every input token.
        cp.n_outputs_max = MAX_ANSWER_SLOTS;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = opts.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.swa_full = true;
        ctx.reset(llama_init_from_model(model.get(), cp));
        require(bool(ctx), "failed to initialize llama context");
        auto mm_params = mtmd_context_params_default();
        mm_params.use_gpu = opts.gpu_layers != 0;
        mm_params.n_threads = opts.threads;
        mm_params.warmup = false;
        mm_params.print_timings = false;
        // Images already have the authoritative HF dimensions; disable native upscaling.
        mm_params.image_min_tokens = 1;
        mm_params.image_max_tokens = opts.image_max_tokens;
        mm.reset(mtmd_init_from_file(opts.mmproj.c_str(), model.get(), mm_params));
        require(bool(mm), "failed to load multimodal projector");
        require(!mtmd_decode_use_mrope(mm.get()), "this runner requires ordinary Gemma 4 positions");
    }

    void invalidate_prefix() noexcept {
        native_memory memory{llama_get_memory(ctx.get())};
        prefix.reset(memory);
    }

    json score(const json & request) {
        try {
            return score_impl(request);
        } catch (...) {
            // llama_decode may leave earlier microbatches committed on failure.
            invalidate_prefix();
            throw;
        }
    }

private:
    json score_impl(const json & request) {
        auto started = std::chrono::steady_clock::now();
        llama_set_causal_attn(ctx.get(), true);
        auto slots = integers(request.at("slots"), "slots");
        auto counts = integers(request.at("nopts"), "nopts");
        auto letters = integers(request.at("letters"), "letters");
        const auto & question_ids = request.at("question_ids");
        require(!slots.empty() && slots.size() <= MAX_ANSWER_SLOTS && counts.size() == slots.size() &&
                question_ids.is_array() && question_ids.size() == slots.size(), "inconsistent question metadata");
        require(std::is_sorted(slots.begin(), slots.end()) &&
                std::adjacent_find(slots.begin(), slots.end()) == slots.end(), "slots must be strictly increasing");
        for (auto count : counts) require(count >= 2 && count <= 52, "invalid option count");
        validate_candidates(letters);
        require(request.at("parts").is_array(), "parts must be an array");
        std::vector<prepared_part> parts;
        size_t token_count = 0;
        for (const auto & part : request.at("parts")) {
            parts.push_back(prepare_part(part));
            token_count += parts.back().count;
        }
        require(token_count <= llama_n_ctx(ctx.get()) && slots.back() < int64_t(token_count),
                "context overflow or answer slot outside input; nothing was truncated");
        require(token_count == request.at("source_input_token_count").get<size_t>(), "source token count mismatch");
        const std::vector<int32_t> no_tokens;
        const bool single_text = parts.size() == 1 && !parts[0].media;
        const s1_prefix::request_view view{
            single_text ? parts[0].tokens : no_tokens, slots, counts, single_text};
        const s1_prefix::geometry geometry{
            size_t(opts.batch), llama_n_batch(ctx.get()), llama_n_ubatch(ctx.get())};
        native_memory memory{llama_get_memory(ctx.get())};
        const auto prefix_plan = prefix.begin(view, opts.reuse_prefix, geometry, memory);
        json rows = json::array();
        for (size_t i = 0; i < slots.size(); ++i) rows.push_back(nullptr);
        llama_pos position = static_cast<llama_pos>(prefix_plan.retained);
        int decode_calls = 0, media_chunks = 0;
        for (auto & part : parts) {
            if (part.media) {
                for (auto slot : slots) require(slot < position || slot >= position + int64_t(part.count),
                                                "answer slot falls inside media embeddings");
                llama_pos after = position;
                require(mtmd_helper_eval_chunk_single(mm.get(), ctx.get(), part.media, position, 0,
                                                      opts.batch, false, &after) == 0,
                        "native media encoding/decoding failed");
                require(after == position + int64_t(part.count), "native media position mismatch");
                position = after;
                decode_calls += int((part.count + opts.batch - 1) / opts.batch);
                ++media_chunks;
                continue;
            }
            // A retained prefix exists only for a single text part. Otherwise
            // preserve each original part's independent chunk origin.
            const size_t first_offset = single_text ? prefix_plan.retained : 0;
            for (size_t offset = first_offset; offset < part.tokens.size(); offset += opts.batch) {
                int n = int(std::min(size_t(opts.batch), part.tokens.size() - offset));
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int i = 0; i < n; ++i) {
                    batch.token[i] = part.tokens[offset + i];
                    batch.pos[i] = position + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = std::binary_search(slots.begin(), slots.end(), position + i);
                }
                int status = llama_decode(ctx.get(), batch);
                llama_batch_free(batch);
                require(status == 0, "native text decode failed: " + std::to_string(status));
                ++decode_calls;
                for (size_t qi = 0; qi < slots.size(); ++qi) {
                    if (slots[qi] < position || slots[qi] >= position + n) continue;
                    const float * logits = llama_get_logits_ith(ctx.get(), slots[qi] - position);
                    require(logits != nullptr, "requested answer-slot logits are missing");
                    json row = json::array();
                    for (int j = 0; j < counts[qi]; ++j) {
                        float value = logits[letters[j]];
                        require(std::isfinite(value), "nonfinite native candidate logit");
                        row.push_back(value);
                    }
                    // Copy now: llama's output storage is replaced by the next decode.
                    rows[qi] = std::move(row);
                }
                position += n;
            }
        }
        for (const auto & row : rows) require(!row.is_null(), "missing question output");
        double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        json response = {{"status", "ok"}, {"case_id", request.at("case_id")},
                {"question_ids", question_ids}, {"slots", slots}, {"runtime_slots", slots},
                {"native_token_count", token_count}, {"raw_logits", rows}, {"latency_ms", elapsed},
                {"preprocessing", {{"decode_calls", decode_calls}, {"media_chunks", media_chunks}}},
                {"runtime", {{"engine", "llama.cpp"}, {"llama_cpp_revision", S1_LLAMA_CPP_REVISION},
                    {"logits_semantics", "post_native_softcap_pre_temperature"},
                    {"projection", "full_vocabulary_then_legal_candidate_selection"},
                    {"decision_passes", 1}, {"autoregressive_tokens", 0},
                    {"cache_scope", opts.reuse_prefix ? "exact_token_serial_batch_aligned_prefix; unsupported_inputs_full_clear"
                                                       : "one_request_cleared_between_requests"},
                    {"media_projection", "native_mtmd"}, {"image_preprocessing", "hf_processed_rgb_roundtrip"},
                    {"context_size", llama_n_ctx(ctx.get())}, {"batch_size", llama_n_batch(ctx.get())},
                    {"max_output_slots", MAX_ANSWER_SLOTS},
                    {"microbatch_size", llama_n_ubatch(ctx.get())}, {"gpu_layers_requested", opts.gpu_layers}}}};
        // Commit only after all requested logits and response fields validate.
        // Constructing the telemetry can still throw; score() then resets both
        // metadata and real KV before the request-level error is returned.
        prefix.commit(view, prefix_plan, geometry, memory);
        response["preprocessing"]["full_input_token_positions"] = token_count;
        response["preprocessing"]["prefill_token_positions"] = token_count - prefix_plan.retained;
        response["preprocessing"]["prefix_cache"] = {
            {"enabled", opts.reuse_prefix}, {"eligible", prefix_plan.eligible},
            {"cache_hit", prefix_plan.retained > 0}, {"cache_valid_for_next_request", prefix.valid()},
            {"previous_input_tokens", prefix_plan.previous_input_tokens},
            {"exact_common_prefix_tokens", prefix_plan.exact_common_prefix_tokens},
            {"unaligned_retained_token_cap", prefix_plan.unaligned_retained_token_cap},
            {"retained_kv_tokens", prefix_plan.retained},
            {"recomputed_token_positions", token_count - prefix_plan.retained},
            {"requested_slots_recomputed", true}, {"native_batch_chunk_size", opts.batch},
            {"alignment", "original_batch_boundary"}, {"answer_cache", false}, {"reason", prefix_plan.reason}};
        return response;
    }
};

int main(int argc, char ** argv) {
    try {
        runtime engine(parse_options(argc, argv));
        std::string line;
        while (std::getline(std::cin, line)) {
            json request;
            json response;
            try {
                request = json::parse(line);
                response = engine.score(request);
            } catch (const std::exception & error) {
                // JSON parsing fails before score() can install its failure guard.
                engine.invalidate_prefix();
                response = {{"status", "error"}, {"case_id", request.is_object() ? request.value("case_id", json(nullptr)) : json(nullptr)},
                            {"error", error.what()}};
            }
            std::cout << response.dump() << '\n' << std::flush;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "s1-gguf: " << error.what() << '\n';
        return 1;
    }
}
