// Standalone Llama 3.2 GGUF inference. CUDA is the only runtime dependency.
// The GGUF reader, byte BPE tokenizer, transformer, and quantized GEMV are here.
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

static std::string current_power_mode() {
    FILE * pipe = popen("nvpmodel -q 2>/dev/null", "r");
    if (!pipe) return "unavailable";
    char name[128] = {}, id[32] = {};
    bool read = fgets(name, sizeof name, pipe) && fgets(id, sizeof id, pipe);
    int status = pclose(pipe);
    if (!read || status != 0) return "unavailable";
    std::string mode(name), mode_id(id);
    const std::string prefix = "NV Power Mode: ";
    if (mode.rfind(prefix, 0) != 0) return "unavailable";
    mode.erase(0, prefix.size());
    while (!mode.empty() && (mode.back() == '\n' || mode.back() == '\r')) mode.pop_back();
    while (!mode_id.empty() && (mode_id.back() == '\n' || mode_id.back() == '\r')) mode_id.pop_back();
    return mode + " (ID " + mode_id + ")";
}

// The INA3221 VDD_IN rail measures input power for the whole Jetson board.
// Samples are held until the next read and integrated using a monotonic clock.
class PowerMeter {
    std::string voltage_path, current_path;
    std::atomic<bool> running{false};
    std::thread sampler;
    mutable std::mutex mutex;
    bool valid = false;
    double watts = 0, energy_j = 0;
    Clock::time_point last_sample;

    static double read_number(const std::string & path) {
        std::ifstream file(path);
        double value;
        if (!(file >> value)) return std::numeric_limits<double>::quiet_NaN();
        return value;
    }
    double read_watts() const {
        // hwmon voltage is mV and current is mA; their product / 1e6 is W.
        return read_number(voltage_path) * read_number(current_path) / 1e6;
    }
    void sample() {
        double power = read_watts();
        auto now = Clock::now();
        std::lock_guard<std::mutex> lock(mutex);
        if (!std::isfinite(power) || power < 0) { valid = false; return; }
        energy_j += watts * seconds(last_sample, now);
        last_sample = now;
        watts = power;
    }
public:
    PowerMeter() {
        for (int h = 0; h < 32 && voltage_path.empty(); ++h) {
            for (int c = 1; c <= 3; ++c) {
                std::string base = "/sys/class/hwmon/hwmon" + std::to_string(h) + "/";
                std::ifstream label(base + "in" + std::to_string(c) + "_label");
                std::string name;
                if (label >> name && name == "VDD_IN") {
                    voltage_path = base + "in" + std::to_string(c) + "_input";
                    current_path = base + "curr" + std::to_string(c) + "_input";
                    break;
                }
            }
        }
        if (voltage_path.empty()) return;
        watts = read_watts();
        if (!std::isfinite(watts) || watts < 0) return;
        last_sample = Clock::now();
        valid = true;
        running = true;
        sampler = std::thread([this] {
            while (running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (running) sample();
            }
        });
    }
    ~PowerMeter() {
        running = false;
        if (sampler.joinable()) sampler.join();
    }
    double joules() const {
        std::lock_guard<std::mutex> lock(mutex);
        if (!valid) return std::numeric_limits<double>::quiet_NaN();
        return energy_j + watts * seconds(last_sample, Clock::now());
    }
};

static void cuda_check(cudaError_t err, const char * where) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
}

struct Reader {
    std::ifstream in;
    explicit Reader(const std::string & path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("Cannot open GGUF: " + path);
    }
    template<class T> T number() {
        T x{};
        in.read(reinterpret_cast<char *>(&x), sizeof x);
        if (!in) throw std::runtime_error("Truncated GGUF metadata");
        return x;
    }
    std::string str() {
        u64 n = number<u64>();
        if (n > 100000000) throw std::runtime_error("Invalid GGUF string length");
        std::string s(n, '\0');
        in.read(s.data(), n);
        if (!in) throw std::runtime_error("Truncated GGUF string");
        return s;
    }
    double scalar(u32 type) {
        switch (type) {
            case 0: return number<u8>();
            case 1: return number<int8_t>();
            case 2: return number<uint16_t>();
            case 3: return number<int16_t>();
            case 4: return number<u32>();
            case 5: return number<int32_t>();
            case 6: return number<float>();
            case 7: return number<u8>();
            case 10: return static_cast<double>(number<u64>());
            case 11: return static_cast<double>(number<int64_t>());
            case 12: return number<double>();
            default: throw std::runtime_error("Unsupported GGUF metadata type");
        }
    }
    void skip(u32 type) {
        if (type == 8) { str(); return; }
        if (type == 9) {
            u32 element = number<u32>();
            u64 count = number<u64>();
            if (count > 10000000) throw std::runtime_error("Invalid GGUF array length");
            for (u64 i = 0; i < count; ++i) skip(element);
            return;
        }
        scalar(type);
    }
};

struct Tensor {
    std::vector<u64> dims;
    u32 type;
    u64 offset;
    const u8 * host = nullptr;
    const u8 * gpu = nullptr;
    int cols() const { return static_cast<int>(dims.at(0)); }
    int rows() const { return static_cast<int>(dims.at(1)); }
};

struct StorageInfo {
    u64 tensors = 0;
    u64 parameters = 0;
    u64 bytes = 0;
};

static const char * storage_name(u32 type) {
    switch (type) {
        case 0: return "FP32";
        case 1: return "FP16";
        case 12: return "Q4_K";
        case 14: return "Q6_K";
        default: return "unknown";
    }
}

struct GGUF {
    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_map<std::string, double> params;
    std::vector<std::string> tokens, merges;
    std::map<u32, StorageInfo> storage;
    u64 parameter_count = 0, tensor_bytes = 0;
    void * mapping = MAP_FAILED;
    u64 file_size = 0, data_start = 0;
    int fd = -1;
    u8 * gpu_data = nullptr;
    std::string gpu_name;
    u64 uploaded_bytes = 0;

    explicit GGUF(const std::string & path) {
        Reader r(path);
        char magic[4];
        r.in.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4)) throw std::runtime_error("File is not GGUF");
        if (r.number<u32>() != 3) throw std::runtime_error("Only GGUF v3 is supported");
        u64 n_tensors = r.number<u64>(), n_meta = r.number<u64>();
        for (u64 i = 0; i < n_meta; ++i) {
            std::string key = r.str();
            u32 type = r.number<u32>();
            if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") {
                if (type != 9 || r.number<u32>() != 8) throw std::runtime_error("Expected string array: " + key);
                u64 count = r.number<u64>();
                auto & target = key == "tokenizer.ggml.tokens" ? tokens : merges;
                target.reserve(count);
                for (u64 j = 0; j < count; ++j) target.push_back(r.str());
            } else if (type == 8) {
                r.str();
            } else if (type == 9) {
                r.skip(type);
            } else {
                params[key] = r.scalar(type);
            }
        }
        for (u64 i = 0; i < n_tensors; ++i) {
            std::string name = r.str();
            Tensor t;
            u32 nd = r.number<u32>();
            if (nd == 0 || nd > 4) throw std::runtime_error("Invalid tensor dimensions");
            for (u32 j = 0; j < nd; ++j) t.dims.push_back(r.number<u64>());
            t.type = r.number<u32>();
            t.offset = r.number<u64>();
            u64 elements = 1;
            for (u64 dim : t.dims) elements *= dim;
            u64 bytes = 0;
            switch (t.type) {
                case 0: bytes = elements * 4; break;
                case 1: bytes = elements * 2; break;
                case 12:
                case 14:
                    if (elements % 256) throw std::runtime_error("Invalid quantized tensor shape: " + name);
                    bytes = (elements / 256) * (t.type == 12 ? 144 : 210);
                    break;
                default: throw std::runtime_error("Unsupported tensor storage type: " + name);
            }
            auto & info = storage[t.type];
            ++info.tensors;
            info.parameters += elements;
            info.bytes += bytes;
            parameter_count += elements;
            tensor_bytes += bytes;
            tensors.emplace(std::move(name), std::move(t));
        }
        u64 alignment = params.count("general.alignment") ? static_cast<u64>(params["general.alignment"]) : 32;
        if (!alignment || (alignment & (alignment - 1))) throw std::runtime_error("Invalid GGUF alignment");
        data_start = (static_cast<u64>(r.in.tellg()) + alignment - 1) & ~(alignment - 1);
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot map GGUF file");
        struct stat st{};
        if (fstat(fd, &st) || st.st_size <= static_cast<off_t>(data_start)) throw std::runtime_error("Invalid GGUF file size");
        file_size = st.st_size;
        mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("GGUF mmap failed");
        const u8 * data = static_cast<const u8 *>(mapping) + data_start;
        for (auto & [name, t] : tensors) {
            if (t.offset >= file_size - data_start) throw std::runtime_error("Invalid tensor offset: " + name);
            t.host = data + t.offset;
        }
    }
    ~GGUF() {
        if (gpu_data) cudaFree(gpu_data);
        if (mapping != MAP_FAILED) munmap(mapping, file_size);
        if (fd >= 0) close(fd);
    }
    int param(const std::string & key) const {
        auto it = params.find(key);
        if (it == params.end()) throw std::runtime_error("Missing GGUF parameter: " + key);
        return static_cast<int>(it->second);
    }
    void print_storage(bool use_dp4a, bool use_q6_dp4a) const {
        std::cerr << std::fixed << std::setprecision(3)
                  << "Model: " << double(parameter_count) / 1e9 << "B parameters; storage ";
        bool first = true;
        for (const auto & [type, info] : storage) {
            if (!first) std::cerr << ", ";
            first = false;
            double percent = 100.0 * double(info.parameters) / double(parameter_count);
            std::cerr << storage_name(type) << " ";
            if (percent < 0.05) std::cerr << "<0.1%";
            else std::cerr << std::setprecision(1) << percent << "%";
        }
        std::cerr << std::setprecision(2) << "; average "
                  << 8.0 * double(tensor_bytes) / double(parameter_count)
                  << " bits/parameter; " << double(tensor_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB tensor data; Q4_K compute "
                  << (use_dp4a ? "INT8 DP4A" : "FP32") << "; Q6_K compute "
                  << (use_q6_dp4a ? "INT8 DP4A" : "FP32")
                  << "; FP32 accumulation\n";
    }
    const Tensor & tensor(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("Missing tensor: " + name);
        return it->second;
    }
    void upload() {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "CUDA device detection");
        if (!count) throw std::runtime_error("No CUDA GPU is available");
        cuda_check(cudaSetDevice(0), "CUDA device selection");
        cudaDeviceProp device{};
        cuda_check(cudaGetDeviceProperties(&device, 0), "CUDA device properties");
        gpu_name = device.name;
        size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "CUDA memory query");
        u64 bytes = file_size - data_start;
        if (free_bytes < bytes + 64 * 1024 * 1024)
            throw std::runtime_error("Not enough free GPU memory for the quantized weights");
        cuda_check(cudaMalloc(&gpu_data, bytes), "GPU weight allocation");
        cuda_check(cudaMemcpy(gpu_data, static_cast<u8 *>(mapping) + data_start, bytes, cudaMemcpyHostToDevice),
                   "GPU weight upload");
        uploaded_bytes = bytes;
        for (auto & [name, t] : tensors) t.gpu = gpu_data + t.offset;
    }
};

static std::string utf8(u32 cp) {
    std::string s;
    if (cp < 0x80) s.push_back(char(cp));
    else if (cp < 0x800) {
        s.push_back(char(0xc0 | (cp >> 6)));
        s.push_back(char(0x80 | (cp & 63)));
    } else if (cp < 0x10000) {
        s.push_back(char(0xe0 | (cp >> 12)));
        s.push_back(char(0x80 | ((cp >> 6) & 63)));
        s.push_back(char(0x80 | (cp & 63)));
    } else {
        s.push_back(char(0xf0 | (cp >> 18)));
        s.push_back(char(0x80 | ((cp >> 12) & 63)));
        s.push_back(char(0x80 | ((cp >> 6) & 63)));
        s.push_back(char(0x80 | (cp & 63)));
    }
    return s;
}

struct Tokenizer {
    const std::vector<std::string> & tokens;
    std::unordered_map<std::string, int> ids, rank;
    std::string byte_symbol[256];
    std::unordered_map<std::string, u8> symbol_byte;
    pcre2_code * regex = nullptr;

    explicit Tokenizer(const GGUF & model) : tokens(model.tokens) {
        if (tokens.size() != 128256) throw std::runtime_error("Expected Llama 3 tokenizer");
        ids.reserve(tokens.size());
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) ids.emplace(tokens[i], i);
        rank.reserve(model.merges.size());
        for (int i = 0; i < static_cast<int>(model.merges.size()); ++i) rank.emplace(model.merges[i], i);
        u32 cp = 256;
        for (int b = 0; b < 256; ++b) {
            bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            byte_symbol[b] = utf8(plain ? b : cp++);
            symbol_byte.emplace(byte_symbol[b], static_cast<u8>(b));
        }
        // Llama 3's Unicode pre-tokenizer; byte BPE is applied to each match.
        const char * pattern =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
            " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        int error = 0;
        PCRE2_SIZE offset = 0;
        regex = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                              PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
        if (!regex) throw std::runtime_error("Could not compile tokenizer regex");
    }
    ~Tokenizer() { if (regex) pcre2_code_free(regex); }

    void bpe_piece(const std::string & piece, std::vector<int> & out) const {
        std::vector<std::string> parts;
        parts.reserve(piece.size());
        for (u8 b : piece) parts.push_back(byte_symbol[b]);
        while (parts.size() > 1) {
            int best = std::numeric_limits<int>::max();
            size_t at = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto it = rank.find(parts[i] + " " + parts[i + 1]);
                if (it != rank.end() && it->second < best) { best = it->second; at = i; }
            }
            if (best == std::numeric_limits<int>::max()) break;
            parts[at] += parts[at + 1];
            parts.erase(parts.begin() + at + 1);
        }
        for (const auto & s : parts) {
            auto it = ids.find(s);
            if (it == ids.end()) throw std::runtime_error("Tokenizer produced an unknown token");
            out.push_back(it->second);
        }
    }
    std::vector<int> encode(const std::string & text) const {
        std::vector<int> out;
        pcre2_match_data * md = pcre2_match_data_create_from_pattern(regex, nullptr);
        PCRE2_SIZE pos = 0;
        while (pos < text.size()) {
            int status = pcre2_match(regex, reinterpret_cast<PCRE2_SPTR>(text.data()),
                                     text.size(), pos, 0, md, nullptr);
            if (status < 0) { bpe_piece(text.substr(pos), out); break; }
            PCRE2_SIZE * span = pcre2_get_ovector_pointer(md);
            if (span[0] > pos) bpe_piece(text.substr(pos, span[0] - pos), out);
            if (span[1] == span[0]) throw std::runtime_error("Tokenizer regex made no progress");
            bpe_piece(text.substr(span[0], span[1] - span[0]), out);
            pos = span[1];
        }
        pcre2_match_data_free(md);
        return out;
    }
    std::string decode(int id) const {
        if (id < 0 || id >= static_cast<int>(tokens.size())) throw std::runtime_error("Invalid token id");
        const auto & s = tokens[id];
        std::string bytes;
        for (size_t i = 0; i < s.size();) {
            u8 first = s[i];
            size_t n = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
            if (i + n > s.size()) throw std::runtime_error("Malformed token UTF-8");
            auto it = symbol_byte.find(s.substr(i, n));
            if (it == symbol_byte.end()) throw std::runtime_error("Cannot decode token");
            bytes.push_back(char(it->second));
            i += n;
        }
        return bytes;
    }
    int special(const std::string & name) const {
        auto it = ids.find(name);
        if (it == ids.end()) throw std::runtime_error("Missing special token " + name);
        return it->second;
    }
};

// GGML Q4_K and Q6_K superblocks. Each block encodes 256 weights.
__device__ float q4_value(const u8 * block, int i) {
    float d = __half2float(*reinterpret_cast<const __half *>(block));
    float dmin = __half2float(*reinterpret_cast<const __half *>(block + 2));
    const u8 * scales = block + 4;
    const u8 * qs = block + 16;
    int sub = i / 32;
    int q_index = (i / 64) * 32 + (i % 32);
    int q = (i % 64) < 32 ? (qs[q_index] & 15) : (qs[q_index] >> 4);
    int sc, m;
    if (sub < 4) {
        sc = scales[sub] & 63;
        m = scales[sub + 4] & 63;
    } else {
        sc = (scales[sub + 4] & 15) | ((scales[sub - 4] >> 6) << 4);
        m = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
    }
    return d * sc * q - dmin * m;
}

__device__ float q6_value(const u8 * block, int i) {
    const u8 * ql = block;
    const u8 * qh = block + 128;
    const int8_t * scales = reinterpret_cast<const int8_t *>(block + 192);
    float d = __half2float(*reinterpret_cast<const __half *>(block + 208));
    int group = i / 128;
    int p = i % 128;
    int lane = p % 32;
    int part = p / 32;
    int low = ql[group * 64 + (part % 2) * 32 + lane];
    low = part < 2 ? low & 15 : low >> 4;
    int high = (qh[group * 32 + lane] >> (part * 2)) & 3;
    int q = (low | (high << 4)) - 32;
    return d * scales[group * 8 + part * 2 + lane / 16] * q;
}

__device__ __forceinline__ void q4_scale_min(const u8 * scales, int sub, int & scale, int & minimum) {
    if (sub < 4) {
        scale = scales[sub] & 63;
        minimum = scales[sub + 4] & 63;
    } else {
        scale = (scales[sub + 4] & 15) | ((scales[sub - 4] >> 6) << 4);
        minimum = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
    }
}

// Each lane consumes the two 4-bit weights packed in one byte.
__device__ __forceinline__ float dot_q4_block(const u8 * block, const float * x, int lane) {
    float d = __half2float(*reinterpret_cast<const __half *>(block));
    float dmin = __half2float(*reinterpret_cast<const __half *>(block + 2));
    const u8 * scales = block + 4;
    const u8 * qs = block + 16;
    float sum = 0;
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        int scale0, min0, scale1, min1;
        q4_scale_min(scales, group * 2, scale0, min0);
        q4_scale_min(scales, group * 2 + 1, scale1, min1);
        int quant = qs[group * 32 + lane];
        sum += (d * scale0 * (quant & 15) - dmin * min0) * x[group * 64 + lane];
        sum += (d * scale1 * (quant >> 4) - dmin * min1) * x[group * 64 + 32 + lane];
    }
    return sum;
}

// Q4_K weights and the current activation are multiplied as packed INT8
// values. The activation scale is per 32 elements, matching Q4_K's subblocks.
__device__ __forceinline__ float dot_q4_block_dp4a(const u8 * block, const int8_t * xq,
                                                    const float * xscale, int block_index, int lane) {
    int sub = lane / 4;
    int chunk = lane % 4;
    int scale, minimum;
    q4_scale_min(block + 4, sub, scale, minimum);
    const u8 * qs = block + 16 + (sub / 2) * 32 + chunk * 8;
    const int8_t * x = xq + block_index * 256 + sub * 32 + chunk * 8;
    uint32_t packed0 = *reinterpret_cast<const uint32_t *>(qs);
    uint32_t packed1 = *reinterpret_cast<const uint32_t *>(qs + 4);
    int x0 = *reinterpret_cast<const int *>(x);
    int x1 = *reinterpret_cast<const int *>(x + 4);
    bool high = sub & 1;
    int q0 = high ? int((packed0 >> 4) & 0x0f0f0f0f) : int(packed0 & 0x0f0f0f0f);
    int q1 = high ? int((packed1 >> 4) & 0x0f0f0f0f) : int(packed1 & 0x0f0f0f0f);
    int dot = __dp4a(q0, x0, 0);
    dot = __dp4a(q1, x1, dot);
    int sum = __dp4a(0x01010101, x0, 0);
    sum = __dp4a(0x01010101, x1, sum);
    float d = __half2float(*reinterpret_cast<const __half *>(block));
    float dmin = __half2float(*reinterpret_cast<const __half *>(block + 2));
    return (d * scale * dot - dmin * minimum * sum) * xscale[block_index * 8 + sub];
}

__global__ void quantize_x32_cuda(const float * x, int8_t * quantized, float * scales, int n) {
    int block = blockIdx.x;
    int lane = threadIdx.x;
    int index = block * 32 + lane;
    float value = index < n ? x[index] : 0.f;
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float scale = maximum > 0 ? maximum / 127.f : 1.f;
    if (lane == 0) scales[block] = scale;
    if (index < n) quantized[index] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value / scale))));
}

__global__ void gemv_q4_dp4a(const u8 * weights, const int8_t * xq, const float * xscale,
                              float * y, int cols, int rows) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const u8 * row_data = weights + static_cast<size_t>(row) * (cols / 256) * 144;
    float sum = 0;
    for (int block = 0; block < cols / 256; ++block)
        sum += dot_q4_block_dp4a(row_data + block * 144, xq, xscale, block, lane);
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) y[row] = sum;
}

// Four 6-bit weights share two low-nibble bytes and one high-bit byte.
__device__ __forceinline__ float dot_q6_block(const u8 * block, const float * x, int lane) {
    const u8 * low = block;
    const u8 * high = block + 128;
    const int8_t * scales = reinterpret_cast<const int8_t *>(block + 192);
    float d = __half2float(*reinterpret_cast<const __half *>(block + 208));
    float sum = 0;
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        int a = low[group * 64 + lane];
        int b = low[group * 64 + 32 + lane];
        int h = high[group * 32 + lane];
        int q0 = ((a & 15) | ((h & 3) << 4)) - 32;
        int q1 = ((b & 15) | (((h >> 2) & 3) << 4)) - 32;
        int q2 = ((a >> 4) | (((h >> 4) & 3) << 4)) - 32;
        int q3 = ((b >> 4) | (((h >> 6) & 3) << 4)) - 32;
        int offset = group * 128 + lane;
        int s = group * 8 + lane / 16;
        sum += scales[s]     * q0 * x[offset];
        sum += scales[s + 2] * q1 * x[offset + 32];
        sum += scales[s + 4] * q2 * x[offset + 64];
        sum += scales[s + 6] * q3 * x[offset + 96];
    }
    return d * sum;
}

// One lane handles eight adjacent weights from one Q6_K 16-weight scale group.
// Both the weight scale and the activation's 32-value scale remain constant
// across two signed INT8 DP4A instructions.
__device__ __forceinline__ uint32_t load_u32_q6(const u8 * bytes) {
    // GGUF tensor starts are aligned, but successive 210-byte Q6_K blocks
    // alternate between 4-byte and 2-byte alignment. Read aligned words and
    // assemble the requested four bytes without an unaligned uint32_t load.
    uintptr_t address = reinterpret_cast<uintptr_t>(bytes);
    const uint32_t * aligned = reinterpret_cast<const uint32_t *>(address & ~uintptr_t(3));
    uint32_t first = aligned[0];
    unsigned shift = (address & 3) * 8;
    if (!shift) return first;
    uint32_t second = aligned[1];
    return (first >> shift) | (second << (32 - shift));
}

__device__ __forceinline__ float dot_q6_block_dp4a(const u8 * block, const int8_t * xq,
                                                    const float * xscale, int block_index, int lane) {
    int sub = lane / 2;
    int offset = (sub & 1) * 16 + (lane & 1) * 8;
    int group = sub / 8;
    int part = (sub / 2) & 3;
    const u8 * low = block + group * 64 + (part & 1) * 32 + offset;
    const u8 * high = block + 128 + group * 32 + offset;
    // Q6_K blocks are 210 bytes, so alternating blocks are only 2-byte aligned.
    uint32_t low0 = load_u32_q6(low);
    uint32_t low1 = load_u32_q6(low + 4);
    uint32_t high0 = load_u32_q6(high);
    uint32_t high1 = load_u32_q6(high + 4);
    int low_shift = part < 2 ? 0 : 4;
    int high_shift = part * 2;
    uint32_t q0 = ((low0 >> low_shift) & 0x0f0f0f0f)
                | (((high0 >> high_shift) & 0x03030303) << 4);
    uint32_t q1 = ((low1 >> low_shift) & 0x0f0f0f0f)
                | (((high1 >> high_shift) & 0x03030303) << 4);
    // Q6_K stores unsigned codes 0..63 with an implicit zero point of 32.
    q0 = __vsub4(q0, 0x20202020);
    q1 = __vsub4(q1, 0x20202020);
    const int8_t * x = xq + block_index * 256 + sub * 16 + (lane & 1) * 8;
    int x0 = *reinterpret_cast<const int *>(x);
    int x1 = *reinterpret_cast<const int *>(x + 4);
    int dot = __dp4a(static_cast<int>(q0), x0, 0);
    dot = __dp4a(static_cast<int>(q1), x1, dot);
    int scale = reinterpret_cast<const int8_t *>(block + 192)[sub];
    float d = __half2float(*reinterpret_cast<const __half *>(block + 208));
    return d * scale * dot * xscale[block_index * 8 + sub / 2];
}

template<int TYPE>
__global__ void gemv_quant(const u8 * weights, const float * x, float * y, int cols, int rows) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    constexpr int bytes_per_block = TYPE == 12 ? 144 : 210;
    const u8 * row_data = weights + static_cast<size_t>(row) * (cols / 256) * bytes_per_block;
    float sum = 0;
    for (int block = 0; block < cols / 256; ++block) {
        const u8 * q = row_data + block * bytes_per_block;
        if constexpr (TYPE == 12) sum += dot_q4_block(q, x + block * 256, lane);
        else sum += dot_q6_block(q, x + block * 256, lane);
    }
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) y[row] = sum;
}

__global__ void gemv_q6_dp4a(const u8 * weights, const int8_t * xq, const float * xscale,
                              float * y, int cols, int rows) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const u8 * row_data = weights + static_cast<size_t>(row) * (cols / 256) * 210;
    float sum = 0;
    for (int block = 0; block < cols / 256; ++block)
        sum += dot_q6_block_dp4a(row_data + block * 210, xq, xscale, block, lane);
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) y[row] = sum;
}

// In prefill, neighboring CTAs process different tokens of the same rows so
// their weight reads can be reused from cache.
template<int TYPE>
__global__ void gemv_batch_dp4a(const u8 * weights, const int8_t * xq, const float * xscale,
                                float * y, int cols, int rows) {
    int token = blockIdx.x;
    int lane = threadIdx.x % 32;
    int row = blockIdx.y * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    constexpr int bytes_per_block = TYPE == 12 ? 144 : 210;
    const u8 * row_data = weights + static_cast<size_t>(row) * (cols / 256) * bytes_per_block;
    const int8_t * token_xq = xq + token * 8192;
    const float * token_scale = xscale + token * 256;
    float sum = 0;
    for (int block = 0; block < cols / 256; ++block) {
        const u8 * quant = row_data + block * bytes_per_block;
        if constexpr (TYPE == 12)
            sum += dot_q4_block_dp4a(quant, token_xq, token_scale, block, lane);
        else
            sum += dot_q6_block_dp4a(quant, token_xq, token_scale, block, lane);
    }
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) y[static_cast<size_t>(token) * rows + row] = sum;
}

__global__ void qkv_batch_cuda(const u8 * wq, const u8 * wk, const u8 * wv,
                               const int8_t * xq, const float * xscale,
                               float * q, float * k, float * v, int dim, int kv_dim, int v_type) {
    int token = blockIdx.x;
    int lane = threadIdx.x % 32;
    int row = blockIdx.y * 4 + threadIdx.x / 32;
    if (row >= dim + 2 * kv_dim) return;
    const u8 * weights;
    float * result;
    int local_row, rows, bytes_per_block;
    if (row < dim) {
        weights = wq; result = q; local_row = row; rows = dim; bytes_per_block = 144;
    } else if (row < dim + kv_dim) {
        weights = wk; result = k; local_row = row - dim; rows = kv_dim; bytes_per_block = 144;
    } else {
        weights = wv; result = v; local_row = row - dim - kv_dim;
        rows = kv_dim; bytes_per_block = v_type == 12 ? 144 : 210;
    }
    const u8 * row_data = weights + static_cast<size_t>(local_row) * (dim / 256) * bytes_per_block;
    const int8_t * token_xq = xq + token * 8192;
    const float * token_scale = xscale + token * 256;
    float sum = 0;
    for (int block = 0; block < dim / 256; ++block) {
        const u8 * quant = row_data + block * bytes_per_block;
        sum += bytes_per_block == 144
            ? dot_q4_block_dp4a(quant, token_xq, token_scale, block, lane)
            : dot_q6_block_dp4a(quant, token_xq, token_scale, block, lane);
    }
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) result[static_cast<size_t>(token) * rows + local_row] = sum;
}

__global__ void gate_up_batch_cuda(const u8 * wgate, const u8 * wup,
                                   const int8_t * xq, const float * xscale,
                                   float * gate, float * up, int dim, int hidden) {
    int token = blockIdx.x;
    int lane = threadIdx.x % 32;
    int row = blockIdx.y * 4 + threadIdx.x / 32;
    if (row >= 2 * hidden) return;
    bool is_gate = row < hidden;
    int local_row = is_gate ? row : row - hidden;
    const u8 * weights = is_gate ? wgate : wup;
    const u8 * row_data = weights + static_cast<size_t>(local_row) * (dim / 256) * 144;
    const int8_t * token_xq = xq + token * 8192;
    const float * token_scale = xscale + token * 256;
    float sum = 0;
    for (int block = 0; block < dim / 256; ++block)
        sum += dot_q4_block_dp4a(row_data + block * 144, token_xq, token_scale, block, lane);
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) (is_gate ? gate : up)[static_cast<size_t>(token) * hidden + local_row] = sum;
}

__global__ void qkv_cuda(const u8 * wq, const u8 * wk, const u8 * wv,
                         const float * x, const int8_t * xq, const float * xscale,
                         float * q, float * k, float * v, int dim, int kv_dim, int v_type,
                         bool dp4a, bool q6_dp4a) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= dim + 2 * kv_dim) return;
    const u8 * weights;
    float * result;
    int local_row;
    int bytes_per_block;
    if (row < dim) {
        weights = wq; result = q; local_row = row; bytes_per_block = 144;
    } else if (row < dim + kv_dim) {
        weights = wk; result = k; local_row = row - dim; bytes_per_block = 144;
    } else {
        weights = wv; result = v; local_row = row - dim - kv_dim;
        bytes_per_block = v_type == 12 ? 144 : 210;
    }
    const u8 * row_data = weights + static_cast<size_t>(local_row) * (dim / 256) * bytes_per_block;
    float sum = 0;
    for (int block = 0; block < dim / 256; ++block) {
        const u8 * quant = row_data + block * bytes_per_block;
        if (bytes_per_block == 144)
            sum += dp4a ? dot_q4_block_dp4a(quant, xq, xscale, block, lane)
                        : dot_q4_block(quant, x + block * 256, lane);
        else sum += q6_dp4a ? dot_q6_block_dp4a(quant, xq, xscale, block, lane)
                            : dot_q6_block(quant, x + block * 256, lane);
    }
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) result[local_row] = sum;
}

__global__ void gate_up_cuda(const u8 * wgate, const u8 * wup,
                             const float * x, const int8_t * xq, const float * xscale,
                             float * gate, float * up, int dim, int hidden, bool dp4a) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= 2 * hidden) return;
    const u8 * weights = row < hidden ? wgate : wup;
    float * result = row < hidden ? gate : up;
    int local_row = row < hidden ? row : row - hidden;
    const u8 * row_data = weights + static_cast<size_t>(local_row) * (dim / 256) * 144;
    float sum = 0;
    for (int block = 0; block < dim / 256; ++block) {
        const u8 * quant = row_data + block * 144;
        sum += dp4a ? dot_q4_block_dp4a(quant, xq, xscale, block, lane)
                    : dot_q4_block(quant, x + block * 256, lane);
    }
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) result[local_row] = sum;
}

__global__ void embedding_cuda(const u8 * weights, float * out, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const u8 * row = weights + static_cast<size_t>(token) * (dim / 256) * 210;
    out[i] = q6_value(row + (i / 256) * 210, i % 256);
}

__global__ void embedding_batch_cuda(const u8 * weights, const int * tokens, float * out, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int token = blockIdx.y;
    const u8 * row = weights + static_cast<size_t>(tokens[token]) * (dim / 256) * 210;
    out[static_cast<size_t>(token) * dim + i] = q6_value(row + (i / 256) * 210, i % 256);
}

__global__ void rmsnorm_cuda(const float * x, const float * weight, float * out, int n, float eps) {
    __shared__ float partial[256];
    int lane = threadIdx.x;
    float sum = 0;
    for (int i = lane; i < n; i += 256) sum += x[i] * x[i];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / n + eps);
    for (int i = lane; i < n; i += 256) out[i] = x[i] * scale * weight[i];
}

// Each 32-value group computes the same RMS scale, then writes its normalized
// values and INT8 representation. Repeating the small reduction across blocks
// avoids a second launch and a global synchronization within the kernel.
__global__ void rmsnorm_quantize_cuda(const float * x, const float * weight, float * out,
                                      int8_t * xq, float * xscale, int n, float eps) {
    __shared__ float partial[256];
    int thread = threadIdx.x;
    float sum = 0;
    for (int i = thread; i < n; i += 256) sum += x[i] * x[i];
    partial[thread] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }
    if (thread >= 32) return;
    int index = blockIdx.x * 32 + thread;
    float value = x[index] * rsqrtf(partial[0] / n + eps) * weight[index];
    out[index] = value;
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float scale = maximum > 0 ? maximum / 127.f : 1.f;
    if (thread == 0) xscale[blockIdx.x] = scale;
    xq[index] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value / scale))));
}

__global__ void rmsnorm_quantize_batch_cuda(const float * x, const float * weight, float * out,
                                            int8_t * xq, float * xscale, int n, float eps) {
    __shared__ float partial[256];
    int token = blockIdx.y;
    int thread = threadIdx.x;
    const float * token_x = x + static_cast<size_t>(token) * n;
    float sum = 0;
    for (int i = thread; i < n; i += 256) sum += token_x[i] * token_x[i];
    partial[thread] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }
    if (thread >= 32) return;
    int index = blockIdx.x * 32 + thread;
    float value = token_x[index] * rsqrtf(partial[0] / n + eps) * weight[index];
    out[static_cast<size_t>(token) * n + index] = value;
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float scale = maximum > 0 ? maximum / 127.f : 1.f;
    if (thread == 0) xscale[token * 256 + blockIdx.x] = scale;
    xq[token * 8192 + index] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value / scale))));
}

__global__ void quantize_batch_cuda(const float * x, int8_t * xq, float * xscale, int cols) {
    int token = blockIdx.x;
    int group = blockIdx.y;
    int lane = threadIdx.x;
    int index = group * 32 + lane;
    float value = x[static_cast<size_t>(token) * cols + index];
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float scale = maximum > 0 ? maximum / 127.f : 1.f;
    if (lane == 0) xscale[token * 256 + group] = scale;
    xq[token * 8192 + index] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value / scale))));
}

__global__ void rope_cuda(float * vector, const float * cosine, const float * sine,
                          int pairs, int pos, int half_head) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    int j = pair * 2;
    int frequency = pair % half_head;
    float c = cosine[pos * half_head + frequency];
    float s = sine[pos * half_head + frequency];
    float a = vector[j], b = vector[j + 1];
    vector[j] = a * c - b * s;
    vector[j + 1] = a * s + b * c;
}

__global__ void cache_cuda(const float * key, const float * value, float * key_cache, float * value_cache,
                           int layer, int pos, int context, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    size_t offset = (static_cast<size_t>(layer) * context + pos) * kv_dim + i;
    key_cache[offset] = key[i];
    value_cache[offset] = value[i];
}

// The key is only consumed from the cache, so rotate it directly into its
// cache slot while rotating the query and copying the value in the same launch.
__global__ void rope_cache_cuda(float * query, const float * key, const float * value,
                                const float * cosine, const float * sine,
                                float * key_cache, float * value_cache,
                                int layer, int pos, int context, int dim, int kv_dim) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= dim / 2) return;
    int i = pair * 2;
    int frequency = pair % 64;
    float c = cosine[pos * 64 + frequency];
    float s = sine[pos * 64 + frequency];
    float q0 = query[i], q1 = query[i + 1];
    query[i] = q0 * c - q1 * s;
    query[i + 1] = q0 * s + q1 * c;
    if (pair < kv_dim / 2) {
        float k0 = key[i], k1 = key[i + 1];
        size_t offset = (static_cast<size_t>(layer) * context + pos) * kv_dim + i;
        key_cache[offset] = k0 * c - k1 * s;
        key_cache[offset + 1] = k0 * s + k1 * c;
        value_cache[offset] = value[i];
        value_cache[offset + 1] = value[i + 1];
    }
}

__global__ void rope_cache_batch_cuda(float * query, const float * key, const float * value,
                                      const float * cosine, const float * sine,
                                      float * key_cache, float * value_cache,
                                      int layer, int start_pos, int context, int dim, int kv_dim) {
    int token = blockIdx.y;
    int pos = start_pos + token;
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= dim / 2) return;
    int i = pair * 2;
    int frequency = pair % 64;
    float c = cosine[pos * 64 + frequency];
    float s = sine[pos * 64 + frequency];
    float * token_q = query + static_cast<size_t>(token) * dim;
    const float * token_k = key + static_cast<size_t>(token) * kv_dim;
    const float * token_v = value + static_cast<size_t>(token) * kv_dim;
    float q0 = token_q[i], q1 = token_q[i + 1];
    token_q[i] = q0 * c - q1 * s;
    token_q[i + 1] = q0 * s + q1 * c;
    if (pair < kv_dim / 2) {
        float k0 = token_k[i], k1 = token_k[i + 1];
        size_t offset = (static_cast<size_t>(layer) * context + pos) * kv_dim + i;
        key_cache[offset] = k0 * c - k1 * s;
        key_cache[offset + 1] = k0 * s + k1 * c;
        value_cache[offset] = token_v[i];
        value_cache[offset + 1] = token_v[i + 1];
    }
}

// One block computes one query head with an online softmax over its visible KV cache.
__global__ void attention_cuda(const float * query, const float * key_cache, const float * value_cache,
                               float * out, int layer, int pos, int context, int kv_dim, int group_size) {
    __shared__ float warp_sums[4];
    int head = blockIdx.x;
    int i = threadIdx.x;
    int warp = i / 32, lane = i % 32;
    int kv_head = head / group_size;
    float max_score = -INFINITY, denominator = 0, weighted = 0;
    const float q = query[head * 128 + i];
    for (int t = 0; t <= pos; ++t) {
        size_t index = (static_cast<size_t>(layer) * context + t) * kv_dim + kv_head * 128 + i;
        float dot = q * key_cache[index];
        for (int delta = 16; delta > 0; delta /= 2) dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) warp_sums[warp] = dot;
        __syncthreads();
        if (i == 0) warp_sums[0] = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * 0.0883883476483f;
        __syncthreads();
        float score = warp_sums[0];
        float new_max = fmaxf(max_score, score);
        float old_factor = __expf(max_score - new_max);
        float new_factor = __expf(score - new_max);
        weighted = weighted * old_factor + new_factor * value_cache[index];
        denominator = denominator * old_factor + new_factor;
        max_score = new_max;
        __syncthreads();
    }
    out[head * 128 + i] = weighted / denominator;
}

__global__ void attention_batch_cuda(const float * query, const float * key_cache, const float * value_cache,
                                     float * out, int layer, int start_pos, int context, int kv_dim,
                                     int group_size, int dim) {
    __shared__ float warp_sums[4];
    int token = blockIdx.y;
    int pos = start_pos + token;
    int head = blockIdx.x;
    int i = threadIdx.x;
    int warp = i / 32, lane = i % 32;
    int kv_head = head / group_size;
    float max_score = -INFINITY, denominator = 0, weighted = 0;
    const float q = query[static_cast<size_t>(token) * dim + head * 128 + i];
    for (int t = 0; t <= pos; ++t) {
        size_t index = (static_cast<size_t>(layer) * context + t) * kv_dim + kv_head * 128 + i;
        float dot = q * key_cache[index];
        for (int delta = 16; delta > 0; delta /= 2) dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) warp_sums[warp] = dot;
        __syncthreads();
        if (i == 0) warp_sums[0] = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * 0.0883883476483f;
        __syncthreads();
        float score = warp_sums[0];
        float new_max = fmaxf(max_score, score);
        float old_factor = __expf(max_score - new_max);
        float new_factor = __expf(score - new_max);
        weighted = weighted * old_factor + new_factor * value_cache[index];
        denominator = denominator * old_factor + new_factor;
        max_score = new_max;
        __syncthreads();
    }
    out[static_cast<size_t>(token) * dim + head * 128 + i] = weighted / denominator;
}

__global__ void add_cuda(float * x, const float * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void add_batch_cuda(float * x, const float * y, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i];
}

__global__ void silu_cuda(float * gate, const float * up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = gate[i] / (1.f + __expf(-gate[i])) * up[i];
}

__global__ void silu_batch_cuda(float * gate, const float * up, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) gate[i] = gate[i] / (1.f + __expf(-gate[i])) * up[i];
}

__global__ void argmax_blocks_cuda(const float * values, float * block_values, int * block_indices, int n) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    int local = threadIdx.x;
    int index = blockIdx.x * 256 + local;
    best_values[local] = index < n ? values[index] : -INFINITY;
    best_indices[local] = index;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (local < stride && best_values[local + stride] > best_values[local]) {
            best_values[local] = best_values[local + stride];
            best_indices[local] = best_indices[local + stride];
        }
        __syncthreads();
    }
    if (local == 0) {
        block_values[blockIdx.x] = best_values[0];
        block_indices[blockIdx.x] = best_indices[0];
    }
}

__global__ void argmax_final_cuda(const float * block_values, const int * block_indices, int * out, int n) {
    __shared__ float best_values[512];
    __shared__ int best_indices[512];
    int i = threadIdx.x;
    best_values[i] = i < n ? block_values[i] : -INFINITY;
    best_indices[i] = i < n ? block_indices[i] : INT_MAX;
    __syncthreads();
    for (int stride = 256; stride > 0; stride /= 2) {
        if (i < stride && best_values[i + stride] > best_values[i]) {
            best_values[i] = best_values[i + stride];
            best_indices[i] = best_indices[i + stride];
        }
        __syncthreads();
    }
    if (i == 0) *out = best_indices[0];
}

struct FastGPU {
    float * x = nullptr, * xn = nullptr, * q = nullptr, * k = nullptr, * v = nullptr;
    float * attn = nullptr, * tmp = nullptr, * gate = nullptr, * up = nullptr;
    float * logits = nullptr, * key_cache = nullptr, * value_cache = nullptr;
    float * cosine = nullptr, * sine = nullptr, * block_values = nullptr;
    int8_t * xq = nullptr;
    float * xscale = nullptr;
    int * block_indices = nullptr, * next_token = nullptr;
    cudaEvent_t event_start = nullptr, event_end = nullptr;
    u64 calls = 0;
    double gpu_seconds = 0;
    int context, q6_warps;
    bool use_dp4a, use_q6_dp4a;

    FastGPU(int layers, int ctx, int kv_dim, bool dp4a, bool q6_dp4a, int q6_launch_warps)
        : context(ctx), q6_warps(q6_launch_warps), use_dp4a(dp4a), use_q6_dp4a(q6_dp4a) {
        auto alloc = [](auto ** ptr, size_t bytes, const char * label) {
            cuda_check(cudaMalloc(ptr, bytes), label);
        };
        alloc(&x, 3072 * sizeof(float), "GPU residual");
        alloc(&xn, 3072 * sizeof(float), "GPU normalized state");
        alloc(&q, 3072 * sizeof(float), "GPU query");
        alloc(&k, 1024 * sizeof(float), "GPU key");
        alloc(&v, 1024 * sizeof(float), "GPU value");
        alloc(&attn, 3072 * sizeof(float), "GPU attention");
        alloc(&tmp, 3072 * sizeof(float), "GPU residual update");
        alloc(&gate, 8192 * sizeof(float), "GPU gate");
        alloc(&up, 8192 * sizeof(float), "GPU up projection");
        alloc(&xq, 8192 * sizeof(int8_t), "GPU quantized activation");
        alloc(&xscale, 256 * sizeof(float), "GPU activation scales");
        alloc(&logits, 128256 * sizeof(float), "GPU logits");
        size_t cache_bytes = static_cast<size_t>(layers) * ctx * kv_dim * sizeof(float);
        alloc(&key_cache, cache_bytes, "GPU key cache");
        alloc(&value_cache, cache_bytes, "GPU value cache");
        alloc(&cosine, static_cast<size_t>(ctx) * 64 * sizeof(float), "GPU RoPE cosine");
        alloc(&sine, static_cast<size_t>(ctx) * 64 * sizeof(float), "GPU RoPE sine");
        alloc(&block_values, 512 * sizeof(float), "GPU argmax values");
        alloc(&block_indices, 512 * sizeof(int), "GPU argmax indices");
        alloc(&next_token, sizeof(int), "GPU next token");
        cuda_check(cudaEventCreate(&event_start), "GPU timer start");
        cuda_check(cudaEventCreate(&event_end), "GPU timer end");
    }
    ~FastGPU() {
        cudaEventDestroy(event_start); cudaEventDestroy(event_end);
        cudaFree(x); cudaFree(xn); cudaFree(q); cudaFree(k); cudaFree(v);
        cudaFree(attn); cudaFree(tmp); cudaFree(gate); cudaFree(up); cudaFree(logits);
        cudaFree(xq); cudaFree(xscale);
        cudaFree(key_cache); cudaFree(value_cache); cudaFree(cosine); cudaFree(sine);
        cudaFree(block_values); cudaFree(block_indices); cudaFree(next_token);
    }
    void quantize(const float * input, int cols) {
        if (!use_dp4a && !use_q6_dp4a) return;
        quantize_x32_cuda<<<(cols + 31) / 32, 32>>>(input, xq, xscale, cols);
        cuda_check(cudaGetLastError(), "activation quantization launch");
    }
    void matvec(const Tensor & t, const float * input, float * output, int expected_cols, int expected_rows) {
        if (t.cols() != expected_cols || t.rows() != expected_rows || (t.type != 12 && t.type != 14))
            throw std::runtime_error("Unexpected quantized matrix shape");
        if (t.type == 12 && use_dp4a) {
            quantize(input, expected_cols);
            gemv_q4_dp4a<<<(expected_rows + 3) / 4, 128>>>(t.gpu, xq, xscale, output,
                                                             expected_cols, expected_rows);
        } else if (t.type == 14 && use_q6_dp4a) {
            quantize(input, expected_cols);
            gemv_q6_dp4a<<<(expected_rows + q6_warps - 1) / q6_warps, q6_warps * 32>>>(
                t.gpu, xq, xscale, output, expected_cols, expected_rows);
        } else if (t.type == 12) gemv_quant<12><<<(expected_rows + 3) / 4, 128>>>(t.gpu, input, output, expected_cols, expected_rows);
        else gemv_quant<14><<<(expected_rows + 3) / 4, 128>>>(t.gpu, input, output, expected_cols, expected_rows);
        cuda_check(cudaGetLastError(), "quantized CUDA matrix launch");
        ++calls;
    }
    void start_timer() { cuda_check(cudaEventRecord(event_start), "GPU timer start"); }
    void stop_timer() {
        cuda_check(cudaEventRecord(event_end), "GPU timer end");
        cuda_check(cudaEventSynchronize(event_end), "GPU timer synchronize");
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, event_start, event_end), "GPU elapsed time");
        gpu_seconds += ms / 1000.0;
    }
};

struct FastRunner {
    GGUF & model;
    FastGPU gpu;
    int dim, layers, heads, kv_heads, head_dim, kv_dim, hidden, vocab, context, gate_warps;
    float eps;
    bool fuse_norm_quant, fuse_rope_cache;
    int batch_capacity;
    int * batch_tokens = nullptr;
    int8_t * batch_xq = nullptr;
    float * batch_xscale = nullptr;
    float * batch_x = nullptr, * batch_xn = nullptr, * batch_q = nullptr;
    float * batch_k = nullptr, * batch_v = nullptr, * batch_attn = nullptr;
    float * batch_tmp = nullptr, * batch_gate = nullptr, * batch_up = nullptr;

    FastRunner(GGUF & m, int max_context, bool use_dp4a, bool use_q6_dp4a,
               bool norm_quant, bool rope_cache, int gate_launch_warps, int q6_launch_warps,
               int prefill_capacity)
        : model(m), gpu(m.param("llama.block_count"), max_context,
                        m.param("llama.embedding_length") / m.param("llama.attention.head_count")
                        * m.param("llama.attention.head_count_kv"), use_dp4a, use_q6_dp4a,
                        q6_launch_warps),
          context(max_context), gate_warps(gate_launch_warps),
          fuse_norm_quant(norm_quant), fuse_rope_cache(rope_cache), batch_capacity(prefill_capacity) {
        dim = m.param("llama.embedding_length");
        layers = m.param("llama.block_count");
        heads = m.param("llama.attention.head_count");
        kv_heads = m.param("llama.attention.head_count_kv");
        hidden = m.param("llama.feed_forward_length");
        vocab = m.param("llama.vocab_size");
        head_dim = dim / heads;
        kv_dim = head_dim * kv_heads;
        eps = m.params.at("llama.attention.layer_norm_rms_epsilon");
        float rope_base = m.params.at("llama.rope.freq_base");
        if (dim != 3072 || layers != 28 || heads != 24 || kv_heads != 8 ||
            head_dim != 128 || hidden != 8192 || vocab != 128256)
            throw std::runtime_error("Fast runner supports only Llama 3.2 3B");
        std::vector<float> cosines(static_cast<size_t>(context) * 64), sines(cosines.size());
        float frequency[64];
        for (int i = 0; i < 64; ++i) frequency[i] = std::pow(rope_base, -2.f * i / head_dim);
        for (int pos = 0; pos < context; ++pos)
            for (int i = 0; i < 64; ++i) {
                float angle = pos * frequency[i];
                cosines[pos * 64 + i] = std::cos(angle);
                sines[pos * 64 + i] = std::sin(angle);
            }
        cuda_check(cudaMemcpy(gpu.cosine, cosines.data(), cosines.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "RoPE cosine upload");
        cuda_check(cudaMemcpy(gpu.sine, sines.data(), sines.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "RoPE sine upload");
        if (batch_capacity > 1) {
            auto alloc = [](auto ** ptr, size_t bytes, const char * label) {
                cuda_check(cudaMalloc(ptr, bytes), label);
            };
            size_t b = batch_capacity;
            alloc(&batch_tokens, b * sizeof(int), "batch input tokens");
            alloc(&batch_xq, b * 8192 * sizeof(int8_t), "batch quantized activations");
            alloc(&batch_xscale, b * 256 * sizeof(float), "batch activation scales");
            alloc(&batch_x, b * dim * sizeof(float), "batch residual");
            alloc(&batch_xn, b * dim * sizeof(float), "batch normalized state");
            alloc(&batch_q, b * dim * sizeof(float), "batch query");
            alloc(&batch_k, b * kv_dim * sizeof(float), "batch key");
            alloc(&batch_v, b * kv_dim * sizeof(float), "batch value");
            alloc(&batch_attn, b * dim * sizeof(float), "batch attention");
            alloc(&batch_tmp, b * dim * sizeof(float), "batch residual update");
            alloc(&batch_gate, b * hidden * sizeof(float), "batch gate");
            alloc(&batch_up, b * hidden * sizeof(float), "batch up projection");
        }
    }
    ~FastRunner() {
        if (batch_tokens) cudaFree(batch_tokens);
        if (batch_xq) cudaFree(batch_xq);
        if (batch_xscale) cudaFree(batch_xscale);
        if (batch_x) cudaFree(batch_x);
        if (batch_xn) cudaFree(batch_xn);
        if (batch_q) cudaFree(batch_q);
        if (batch_k) cudaFree(batch_k);
        if (batch_v) cudaFree(batch_v);
        if (batch_attn) cudaFree(batch_attn);
        if (batch_tmp) cudaFree(batch_tmp);
        if (batch_gate) cudaFree(batch_gate);
        if (batch_up) cudaFree(batch_up);
    }
    void reset() {} // Every cache position used by the next prompt is overwritten before it is read.
    void batch_matvec(const Tensor & t, const float * input, float * output,
                      int cols, int rows, int count) {
        if (t.cols() != cols || t.rows() != rows || (t.type != 12 && t.type != 14))
            throw std::runtime_error("Unexpected batched matrix shape");
        quantize_batch_cuda<<<dim3(count, cols / 32), 32>>>(input, batch_xq, batch_xscale, cols);
        if (t.type == 12)
            gemv_batch_dp4a<12><<<dim3(count, (rows + 3) / 4), 128>>>(
                t.gpu, batch_xq, batch_xscale, output, cols, rows);
        else
            gemv_batch_dp4a<14><<<dim3(count, (rows + 3) / 4), 128>>>(
                t.gpu, batch_xq, batch_xscale, output, cols, rows);
        cuda_check(cudaGetLastError(), "batched CUDA matrix launch");
        gpu.calls += count;
    }
    int prefill(const std::vector<int> & input,
                std::vector<std::vector<float>> * layer_trace = nullptr,
                std::vector<std::vector<int8_t>> * quant_trace = nullptr,
                std::vector<std::vector<float>> * all_layer_trace = nullptr,
                std::vector<std::vector<int8_t>> * first_quant_trace = nullptr) {
        if (batch_capacity <= 1) throw std::runtime_error("Batched prefill was not configured");
        if (all_layer_trace) all_layer_trace->resize(input.size() * layers);
        const Tensor & embedding = model.tensor("token_embd.weight");
        if (embedding.type != 14) throw std::runtime_error("Expected Q6_K token embedding");
        for (size_t start = 0; start < input.size(); start += batch_capacity) {
            int count = std::min<size_t>(batch_capacity, input.size() - start);
            cuda_check(cudaMemcpy(batch_tokens, input.data() + start, count * sizeof(int),
                                  cudaMemcpyHostToDevice), "batch input upload");
            embedding_batch_cuda<<<dim3((dim + 255) / 256, count), 256>>>(
                embedding.gpu, batch_tokens, batch_x, dim);
            for (int layer = 0; layer < layers; ++layer) {
                std::string prefix = "blk." + std::to_string(layer) + ".";
                const Tensor & attn_norm = model.tensor(prefix + "attn_norm.weight");
                rmsnorm_quantize_batch_cuda<<<dim3(dim / 32, count), 256>>>(
                    batch_x, reinterpret_cast<const float *>(attn_norm.gpu), batch_xn,
                    batch_xq, batch_xscale, dim, eps);
                if (quant_trace && start + count == input.size()) {
                    quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(quant_trace->back().data(),
                                          batch_xq + static_cast<size_t>(count - 1) * 8192,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched attention quantization trace download");
                }
                if (first_quant_trace && start == 0) {
                    first_quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(first_quant_trace->back().data(), batch_xq,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched first-token QKV quantization trace download");
                }
                const Tensor & wq = model.tensor(prefix + "attn_q.weight");
                const Tensor & wk = model.tensor(prefix + "attn_k.weight");
                const Tensor & wv = model.tensor(prefix + "attn_v.weight");
                if (wq.type != 12 || wk.type != 12 || (wv.type != 12 && wv.type != 14))
                    throw std::runtime_error("Unexpected QKV quantization");
                qkv_batch_cuda<<<dim3(count, (dim + 2 * kv_dim + 3) / 4), 128>>>(
                    wq.gpu, wk.gpu, wv.gpu, batch_xq, batch_xscale,
                    batch_q, batch_k, batch_v, dim, kv_dim, wv.type);
                gpu.calls += 3 * count;
                rope_cache_batch_cuda<<<dim3((dim / 2 + 255) / 256, count), 256>>>(
                    batch_q, batch_k, batch_v, gpu.cosine, gpu.sine,
                    gpu.key_cache, gpu.value_cache, layer, static_cast<int>(start), context, dim, kv_dim);
                attention_batch_cuda<<<dim3(heads, count), 128>>>(
                    batch_q, gpu.key_cache, gpu.value_cache, batch_attn,
                    layer, static_cast<int>(start), context, kv_dim, heads / kv_heads, dim);
                batch_matvec(model.tensor(prefix + "attn_output.weight"),
                             batch_attn, batch_tmp, dim, dim, count);
                if (quant_trace && start + count == input.size()) {
                    quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(quant_trace->back().data(),
                                          batch_xq + static_cast<size_t>(count - 1) * 8192,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched attention output quantization trace download");
                }
                if (first_quant_trace && start == 0) {
                    first_quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(first_quant_trace->back().data(), batch_xq,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched first-token attention output quantization trace download");
                }
                add_batch_cuda<<<(count * dim + 255) / 256, 256>>>(batch_x, batch_tmp, count * dim);
                const Tensor & ffn_norm = model.tensor(prefix + "ffn_norm.weight");
                rmsnorm_quantize_batch_cuda<<<dim3(dim / 32, count), 256>>>(
                    batch_x, reinterpret_cast<const float *>(ffn_norm.gpu), batch_xn,
                    batch_xq, batch_xscale, dim, eps);
                if (quant_trace && start + count == input.size()) {
                    quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(quant_trace->back().data(),
                                          batch_xq + static_cast<size_t>(count - 1) * 8192,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched feedforward quantization trace download");
                }
                if (first_quant_trace && start == 0) {
                    first_quant_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(first_quant_trace->back().data(), batch_xq,
                                          dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched first-token gate/up quantization trace download");
                }
                const Tensor & wgate = model.tensor(prefix + "ffn_gate.weight");
                const Tensor & wup = model.tensor(prefix + "ffn_up.weight");
                if (wgate.type != 12 || wup.type != 12)
                    throw std::runtime_error("Unexpected feedforward quantization");
                gate_up_batch_cuda<<<dim3(count, (2 * hidden + 3) / 4), 128>>>(
                    wgate.gpu, wup.gpu, batch_xq, batch_xscale,
                    batch_gate, batch_up, dim, hidden);
                gpu.calls += 2 * count;
                silu_batch_cuda<<<(count * hidden + 255) / 256, 256>>>(
                    batch_gate, batch_up, count * hidden);
                batch_matvec(model.tensor(prefix + "ffn_down.weight"),
                             batch_gate, batch_tmp, hidden, dim, count);
                if (quant_trace && start + count == input.size()) {
                    quant_trace->emplace_back(hidden);
                    cuda_check(cudaMemcpy(quant_trace->back().data(),
                                          batch_xq + static_cast<size_t>(count - 1) * 8192,
                                          hidden * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched feedforward down quantization trace download");
                }
                if (first_quant_trace && start == 0) {
                    first_quant_trace->emplace_back(hidden);
                    cuda_check(cudaMemcpy(first_quant_trace->back().data(), batch_xq,
                                          hidden * sizeof(int8_t), cudaMemcpyDeviceToHost),
                               "batched first-token down quantization trace download");
                }
                add_batch_cuda<<<(count * dim + 255) / 256, 256>>>(batch_x, batch_tmp, count * dim);
                if (all_layer_trace) {
                    std::vector<float> chunk(static_cast<size_t>(count) * dim);
                    cuda_check(cudaMemcpy(chunk.data(), batch_x, chunk.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost), "batched all-token layer trace download");
                    for (int token = 0; token < count; ++token)
                        (*all_layer_trace)[(start + token) * layers + layer] =
                            std::vector<float>(chunk.begin() + static_cast<size_t>(token) * dim,
                                               chunk.begin() + static_cast<size_t>(token + 1) * dim);
                }
                if (layer_trace && start + count == input.size()) {
                    layer_trace->emplace_back(dim);
                    cuda_check(cudaMemcpy(layer_trace->back().data(),
                                          batch_x + static_cast<size_t>(count - 1) * dim,
                                          dim * sizeof(float), cudaMemcpyDeviceToHost),
                               "batched layer trace download");
                }
            }
            if (start + count == input.size())
                cuda_check(cudaMemcpy(gpu.x, batch_x + static_cast<size_t>(count - 1) * dim,
                                      dim * sizeof(float), cudaMemcpyDeviceToDevice),
                           "last prefill residual copy");
        }
        cuda_check(cudaGetLastError(), "batched prefill CUDA launch");
        const Tensor & output_norm = model.tensor("output_norm.weight");
        rmsnorm_cuda<<<1, 256>>>(gpu.x, reinterpret_cast<const float *>(output_norm.gpu), gpu.xn, dim, eps);
        const Tensor & output = model.tensors.count("output.weight")
            ? model.tensor("output.weight") : model.tensor("token_embd.weight");
        gpu.matvec(output, gpu.xn, gpu.logits, dim, vocab);
        constexpr int blocks = (128256 + 255) / 256;
        argmax_blocks_cuda<<<blocks, 256>>>(gpu.logits, gpu.block_values, gpu.block_indices, vocab);
        argmax_final_cuda<<<1, 512>>>(gpu.block_values, gpu.block_indices, gpu.next_token, blocks);
        cuda_check(cudaGetLastError(), "batched prefill argmax launch");
        int result = -1;
        cuda_check(cudaMemcpy(&result, gpu.next_token, sizeof(int), cudaMemcpyDeviceToHost),
                   "batched prefill next token download");
        return result;
    }
    int forward(int token, int pos, bool need_logits,
                std::vector<std::vector<float>> * layer_trace = nullptr,
                std::vector<std::vector<int8_t>> * quant_trace = nullptr) {
        if (pos < 0 || pos >= context || token < 0 || token >= vocab)
            throw std::runtime_error("Token or position outside model context");
        const Tensor & embedding = model.tensor("token_embd.weight");
        if (embedding.type != 14) throw std::runtime_error("Expected Q6_K token embedding");
        embedding_cuda<<<(dim + 255) / 256, 256>>>(embedding.gpu, gpu.x, token, dim);
        for (int layer = 0; layer < layers; ++layer) {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            const Tensor & attn_norm = model.tensor(prefix + "attn_norm.weight");
            if (fuse_norm_quant && (gpu.use_dp4a || gpu.use_q6_dp4a))
                rmsnorm_quantize_cuda<<<dim / 32, 256>>>(gpu.x,
                    reinterpret_cast<const float *>(attn_norm.gpu), gpu.xn,
                    gpu.xq, gpu.xscale, dim, eps);
            else
                rmsnorm_cuda<<<1, 256>>>(gpu.x, reinterpret_cast<const float *>(attn_norm.gpu), gpu.xn, dim, eps);
            const Tensor & wq = model.tensor(prefix + "attn_q.weight");
            const Tensor & wk = model.tensor(prefix + "attn_k.weight");
            const Tensor & wv = model.tensor(prefix + "attn_v.weight");
            if (wq.type != 12 || wk.type != 12 || (wv.type != 12 && wv.type != 14))
                throw std::runtime_error("Unexpected QKV quantization");
            if (!fuse_norm_quant) gpu.quantize(gpu.xn, dim);
            if (quant_trace) {
                quant_trace->emplace_back(dim);
                cuda_check(cudaMemcpy(quant_trace->back().data(), gpu.xq,
                                      dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                           "sequential attention quantization trace download");
            }
            qkv_cuda<<<(dim + 2 * kv_dim + 3) / 4, 128>>>(wq.gpu, wk.gpu, wv.gpu,
                                                            gpu.xn, gpu.xq, gpu.xscale,
                                                            gpu.q, gpu.k, gpu.v, dim, kv_dim, wv.type,
                                                            gpu.use_dp4a, gpu.use_q6_dp4a);
            gpu.calls += 3;
            if (fuse_rope_cache)
                rope_cache_cuda<<<(dim / 2 + 255) / 256, 256>>>(gpu.q, gpu.k, gpu.v,
                    gpu.cosine, gpu.sine, gpu.key_cache, gpu.value_cache,
                    layer, pos, context, dim, kv_dim);
            else {
                rope_cuda<<<(dim / 2 + 255) / 256, 256>>>(gpu.q, gpu.cosine, gpu.sine, dim / 2, pos, head_dim / 2);
                rope_cuda<<<(kv_dim / 2 + 255) / 256, 256>>>(gpu.k, gpu.cosine, gpu.sine, kv_dim / 2, pos, head_dim / 2);
                cache_cuda<<<(kv_dim + 255) / 256, 256>>>(gpu.k, gpu.v, gpu.key_cache, gpu.value_cache,
                                                         layer, pos, context, kv_dim);
            }
            attention_cuda<<<heads, 128>>>(gpu.q, gpu.key_cache, gpu.value_cache, gpu.attn,
                                           layer, pos, context, kv_dim, heads / kv_heads);
            gpu.matvec(model.tensor(prefix + "attn_output.weight"), gpu.attn, gpu.tmp, dim, dim);
            if (quant_trace) {
                quant_trace->emplace_back(dim);
                cuda_check(cudaMemcpy(quant_trace->back().data(), gpu.xq,
                                      dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                           "sequential attention output quantization trace download");
            }
            add_cuda<<<(dim + 255) / 256, 256>>>(gpu.x, gpu.tmp, dim);
            const Tensor & ffn_norm = model.tensor(prefix + "ffn_norm.weight");
            if (fuse_norm_quant && (gpu.use_dp4a || gpu.use_q6_dp4a))
                rmsnorm_quantize_cuda<<<dim / 32, 256>>>(gpu.x,
                    reinterpret_cast<const float *>(ffn_norm.gpu), gpu.xn,
                    gpu.xq, gpu.xscale, dim, eps);
            else
                rmsnorm_cuda<<<1, 256>>>(gpu.x, reinterpret_cast<const float *>(ffn_norm.gpu), gpu.xn, dim, eps);
            const Tensor & wgate = model.tensor(prefix + "ffn_gate.weight");
            const Tensor & wup = model.tensor(prefix + "ffn_up.weight");
            if (wgate.type != 12 || wup.type != 12)
                throw std::runtime_error("Unexpected feedforward quantization");
            if (!fuse_norm_quant) gpu.quantize(gpu.xn, dim);
            if (quant_trace) {
                quant_trace->emplace_back(dim);
                cuda_check(cudaMemcpy(quant_trace->back().data(), gpu.xq,
                                      dim * sizeof(int8_t), cudaMemcpyDeviceToHost),
                           "sequential feedforward quantization trace download");
            }
            gate_up_cuda<<<(2 * hidden + gate_warps - 1) / gate_warps, gate_warps * 32>>>(
                                                        wgate.gpu, wup.gpu,
                                                        gpu.xn, gpu.xq, gpu.xscale,
                                                        gpu.gate, gpu.up, dim, hidden, gpu.use_dp4a);
            gpu.calls += 2;
            silu_cuda<<<(hidden + 255) / 256, 256>>>(gpu.gate, gpu.up, hidden);
            gpu.matvec(model.tensor(prefix + "ffn_down.weight"), gpu.gate, gpu.tmp, hidden, dim);
            if (quant_trace) {
                quant_trace->emplace_back(hidden);
                cuda_check(cudaMemcpy(quant_trace->back().data(), gpu.xq,
                                      hidden * sizeof(int8_t), cudaMemcpyDeviceToHost),
                           "sequential feedforward down quantization trace download");
            }
            add_cuda<<<(dim + 255) / 256, 256>>>(gpu.x, gpu.tmp, dim);
            if (layer_trace) {
                layer_trace->emplace_back(dim);
                cuda_check(cudaMemcpy(layer_trace->back().data(), gpu.x,
                                      dim * sizeof(float), cudaMemcpyDeviceToHost),
                           "sequential layer trace download");
            }
        }
        cuda_check(cudaGetLastError(), "transformer CUDA launch");
        if (!need_logits) return -1;
        const Tensor & output_norm = model.tensor("output_norm.weight");
        rmsnorm_cuda<<<1, 256>>>(gpu.x, reinterpret_cast<const float *>(output_norm.gpu), gpu.xn, dim, eps);
        const Tensor & output = model.tensors.count("output.weight")
            ? model.tensor("output.weight") : model.tensor("token_embd.weight");
        gpu.matvec(output, gpu.xn, gpu.logits, dim, vocab);
        constexpr int blocks = (128256 + 255) / 256;
        argmax_blocks_cuda<<<blocks, 256>>>(gpu.logits, gpu.block_values, gpu.block_indices, vocab);
        argmax_final_cuda<<<1, 512>>>(gpu.block_values, gpu.block_indices, gpu.next_token, blocks);
        cuda_check(cudaGetLastError(), "argmax CUDA launch");
        int result = -1;
        cuda_check(cudaMemcpy(&result, gpu.next_token, sizeof(int), cudaMemcpyDeviceToHost), "next token download");
        return result;
    }
};

static std::vector<int> chat_prompt(const Tokenizer & tok, const std::string & prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto part = tok.encode(text);
        ids.insert(ids.end(), part.begin(), part.end());
    };
    ids.push_back(tok.special("<|begin_of_text|>"));
    ids.push_back(tok.special("<|start_header_id|>"));
    append("user");
    ids.push_back(tok.special("<|end_header_id|>"));
    append("\n\n" + prompt);
    ids.push_back(tok.special("<|eot_id|>"));
    ids.push_back(tok.special("<|start_header_id|>"));
    append("assistant");
    ids.push_back(tok.special("<|end_header_id|>"));
    append("\n\n");
    return ids;
}

static void print_optimization_summary(bool tokenize_only, bool use_dp4a,
                                       bool use_q6_dp4a, bool fuse_norm_quant,
                                       bool fuse_rope_cache, int prefill_batch,
                                       int gate_warps, int q6_warps) {
    std::cerr << "Optimization strategies used:\n"
              << "  - GGUF memory mapping for model loading\n";
    if (tokenize_only) return;
    std::cerr << "  - One-time GPU weight upload; weights and KV cache remain on GPU\n"
              << "  - Q4_K matrix path: "
              << (use_dp4a ? "INT8 DP4A with per-32-value activation scales"
                            : "FP32 computation from quantized weights") << '\n'
              << "  - Q6_K matrix path: "
              << (use_q6_dp4a ? "INT8 DP4A with per-32-value activation scales"
                               : "FP32 computation from quantized weights") << '\n';
    if (prefill_batch > 1)
        std::cerr << "  - Batched prefill: " << prefill_batch
                  << " tokens per chunk for weight-cache reuse; fused RMSNorm/quantization"
                     " and RoPE/KV writes\n";
    else
        std::cerr << "  - Sequential prefill using the decode kernels\n";
    std::cerr << "  - Decode RMSNorm/quantization fusion: "
              << (fuse_norm_quant && (use_dp4a || use_q6_dp4a) ? "on" : "off") << '\n'
              << "  - Decode RoPE/KV-cache fusion: " << (fuse_rope_cache ? "on" : "off") << '\n'
              << "  - Decode matrix launch: gate/up " << gate_warps
              << " warps/block; Q6_K " << (use_q6_dp4a ? q6_warps : 4)
              << " warps/block\n"
              << "  - Online attention softmax and GPU argmax reduction\n";
}

static void print_supported_options(std::ostream & out) {
    out << "Supported options:\n"
        << "  --model path.gguf          GGUF file (default: llama3.2-3b/model.gguf)\n"
        << "  --max-tokens N             Maximum response tokens (default: 128)\n"
        << "  --context N                Context capacity, at least 128 (default: 1024)\n"
        << "  --prompts-file path        Read one prompt per nonempty line\n"
        << "  --stats                    Print per-prompt and average timing/energy statistics\n"
        << "  --q6-dp4a                 Use INT8 DP4A for Q6_K (default)\n"
        << "  --float-q6                 Use FP32 computation for Q6_K\n"
        << "  --float-gemv               Use FP32 computation for Q4_K and Q6_K\n"
        << "  --fused-norm-quant         Fuse decode RMSNorm and activation quantization\n"
        << "  --fused-rope-cache         Fuse decode RoPE and KV-cache writes\n"
        << "  --fused-small-kernels      Enable both decode fusions\n"
        << "  --gate-warps N             Decode gate/up warps per block: 2, 4, 8, or 16 (default: 4)\n"
        << "  --q6-warps N               Decode Q6_K DP4A warps per block: 2, 4, 8, or 16 (default: 4)\n"
        << "  --prefill-batch N          Prompt tokens per batch: 1..32 (default: 1; N>1 requires DP4A)\n"
        << "  --tokenize-only            Print token IDs without running GPU inference\n"
        << "  --compare-prefill          Compare sequential and batched prefill (requires N>1)\n"
        << "  --help                     Show usage and supported options\n"
        << "  prompt ...                 Positional prompts; stdin is used if none are given\n";
}

int main(int argc, char ** argv) {
    try {
        std::string model_path = "llama3.2-3b/model.gguf";
        std::vector<std::string> prompts;
        int max_tokens = 128, context = 1024;
        bool tokenize_only = false, stats = false, prompts_file_provided = false;
        bool compare_prefill = false;
        bool use_dp4a = true, use_q6_dp4a = true;
        bool fuse_norm_quant = false, fuse_rope_cache = false;
        int gate_warps = 4, q6_warps = 4, prefill_batch = 1;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
            else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
            else if (arg == "--context" && i + 1 < argc) context = std::stoi(argv[++i]);
            else if (arg == "--gate-warps" && i + 1 < argc) gate_warps = std::stoi(argv[++i]);
            else if (arg == "--q6-warps" && i + 1 < argc) q6_warps = std::stoi(argv[++i]);
            else if (arg == "--prefill-batch" && i + 1 < argc) prefill_batch = std::stoi(argv[++i]);
            else if (arg == "--prompts-file" && i + 1 < argc) {
                prompts_file_provided = true;
                std::ifstream file(argv[++i]);
                if (!file) throw std::runtime_error("Cannot open prompts file");
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (!line.empty()) prompts.push_back(line);
                }
                if (file.bad()) throw std::runtime_error("Could not read prompts file");
            } else if (arg == "--tokenize-only") tokenize_only = true;
            else if (arg == "--compare-prefill") compare_prefill = true;
            else if (arg == "--stats") stats = true;
            else if (arg == "--q6-dp4a") use_q6_dp4a = true;
            else if (arg == "--float-q6") use_q6_dp4a = false;
            else if (arg == "--float-gemv") use_dp4a = use_q6_dp4a = false;
            else if (arg == "--fused-norm-quant") fuse_norm_quant = true;
            else if (arg == "--fused-rope-cache") fuse_rope_cache = true;
            else if (arg == "--fused-small-kernels") fuse_norm_quant = fuse_rope_cache = true;
            else if (arg == "--help") {
                std::cout << "Usage: ./inferencer [options] [prompt ...]\n";
                print_supported_options(std::cout);
                return 0;
            } else if (arg[0] == '-') throw std::runtime_error("Unknown option: " + arg);
            else if (!arg.empty()) prompts.push_back(arg);
        }
        if (prompts.empty() && !prompts_file_provided) {
            std::string prompt;
            std::getline(std::cin, prompt, '\0');
            if (!prompt.empty()) prompts.push_back(prompt);
        }
        if (prompts.empty() || max_tokens < 1 || context < 128)
            throw std::runtime_error("Provide prompts, positive --max-tokens, and --context >= 128");
        auto valid_warps = [](int count) {
            return count == 2 || count == 4 || count == 8 || count == 16;
        };
        if (!valid_warps(gate_warps) || !valid_warps(q6_warps))
            throw std::runtime_error("--gate-warps and --q6-warps must be 2, 4, 8, or 16");
        if (prefill_batch < 1 || prefill_batch > 32)
            throw std::runtime_error("--prefill-batch must be in 1..32");
        if (prefill_batch > 1 && (!use_dp4a || !use_q6_dp4a))
            throw std::runtime_error("Batched prefill currently requires both DP4A matrix paths");
        if (compare_prefill && (tokenize_only || prefill_batch == 1))
            throw std::runtime_error("--compare-prefill requires --prefill-batch 2..32 and GPU inference");
        if (prompts.size() > 1) stats = true;

        std::string power_mode = current_power_mode();
        std::unique_ptr<PowerMeter> power;
        if (!tokenize_only) power = std::make_unique<PowerMeter>();
        double run_energy_start = power ? power->joules() : 0;
        auto program_started = Clock::now();
        GGUF model(model_path);
        auto model_loaded = Clock::now();
        Tokenizer tokenizer(model);
        auto tokenizer_ready = Clock::now();
        model.print_storage(use_dp4a, use_q6_dp4a);
        auto upload_started = tokenizer_ready;
        auto upload_finished = tokenizer_ready;
        auto runner_ready = tokenizer_ready;
        if (!tokenize_only) {
            upload_started = Clock::now();
            model.upload();
            upload_finished = Clock::now();
            std::cerr << "CUDA device 0: " << model.gpu_name << "; uploaded "
                      << std::fixed << std::setprecision(2)
                      << double(model.uploaded_bytes) / (1024.0 * 1024.0 * 1024.0)
                      << " GiB of model weights to GPU memory\n";
        }
        std::unique_ptr<FastRunner> runner;
        if (!tokenize_only) runner = std::make_unique<FastRunner>(model, context, use_dp4a,
                                                                  use_q6_dp4a, fuse_norm_quant, fuse_rope_cache,
                                                                  gate_warps, q6_warps, prefill_batch);
        runner_ready = Clock::now();
        if (!tokenize_only) std::cerr << "Q4_K matrix path: "
                                       << (use_dp4a ? "INT8 DP4A" : "FP32")
                                       << "; Q6_K matrix path: "
                                       << (use_q6_dp4a ? "INT8 DP4A" : "FP32") << '\n';
        if (!tokenize_only) std::cerr << "Decode small-kernel fusion: RMSNorm+quantize "
                                       << (fuse_norm_quant && (use_dp4a || use_q6_dp4a) ? "on" : "off")
                                       << "; RoPE+KV cache " << (fuse_rope_cache ? "on" : "off") << '\n';
        if (!tokenize_only) std::cerr << "Decode matrix launch: gate/up " << gate_warps
                                       << " warps/block; Q6_K GEMV "
                                       << (use_q6_dp4a ? q6_warps : 4) << " warps/block\n";
        if (!tokenize_only) std::cerr << "Prompt prefill batch: " << prefill_batch << " token(s)\n";

        if (compare_prefill) {
            std::cout << std::setprecision(9);
            struct Trace {
                std::vector<float> residual;
                std::vector<std::vector<float>> layers;
                std::vector<std::vector<float>> all_layers;
                std::vector<std::vector<int8_t>> quantized;
                std::vector<std::vector<int8_t>> first_quantized;
                std::vector<std::vector<float>> logits;
                std::vector<int> ids;
            };
            int steps = std::min(max_tokens, 8);
            for (size_t p = 0; p < prompts.size(); ++p) {
                auto input = chat_prompt(tokenizer, prompts[p]);
                if (input.size() + steps > static_cast<size_t>(context))
                    throw std::runtime_error("Comparison prompt exceeds --context");
                auto run = [&](bool batched) {
                    Trace trace;
                    runner->reset();
                    int next = 0;
                    trace.all_layers.resize(input.size() * runner->layers);
                    if (batched) next = runner->prefill(input, &trace.layers,
                                                         &trace.quantized, &trace.all_layers,
                                                         &trace.first_quantized);
                    else for (size_t i = 0; i < input.size(); ++i) {
                        std::vector<std::vector<float>> token_layers;
                        next = runner->forward(input[i], i, i + 1 == input.size(), &token_layers,
                                               i == 0 ? &trace.first_quantized :
                                               (i + 1 == input.size() ? &trace.quantized : nullptr));
                        for (int layer = 0; layer < runner->layers; ++layer)
                            trace.all_layers[i * runner->layers + layer] = std::move(token_layers[layer]);
                        if (i + 1 == input.size())
                            for (int layer = 0; layer < runner->layers; ++layer)
                                trace.layers.push_back(trace.all_layers[i * runner->layers + layer]);
                    }
                    if (!batched && input.size() == 1) trace.quantized = trace.first_quantized;
                    trace.residual.resize(runner->dim);
                    cuda_check(cudaMemcpy(trace.residual.data(), runner->gpu.x,
                                          trace.residual.size() * sizeof(float), cudaMemcpyDeviceToHost),
                               "comparison residual download");
                    for (int i = 0; i < steps; ++i) {
                        trace.ids.push_back(next);
                        trace.logits.emplace_back(runner->vocab);
                        cuda_check(cudaMemcpy(trace.logits.back().data(), runner->gpu.logits,
                                              runner->vocab * sizeof(float), cudaMemcpyDeviceToHost),
                                   "comparison logits download");
                        if (i + 1 < steps)
                            next = runner->forward(next, input.size() + i, true);
                    }
                    return trace;
                };
                Trace baseline = run(false), batched = run(true);
                auto difference = [](const std::vector<float> & a, const std::vector<float> & b) {
                    double max_abs = 0, square_sum = 0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        double delta = double(a[i]) - double(b[i]);
                        max_abs = std::max(max_abs, std::abs(delta));
                        square_sum += delta * delta;
                    }
                    return std::pair<double, double>{max_abs, std::sqrt(square_sum / a.size())};
                };
                auto [residual_max, residual_rms] = difference(baseline.residual, batched.residual);
                std::cout << "Prompt " << p + 1 << ": residual max abs difference "
                          << residual_max << ", RMS difference " << residual_rms << '\n';
                for (size_t token = 0; token < input.size(); ++token) {
                    for (int layer = 0; layer < runner->layers; ++layer) {
                        auto [token_max, token_rms] = difference(
                            baseline.all_layers[token * runner->layers + layer],
                            batched.all_layers[token * runner->layers + layer]);
                        if (token_max > 1e-6) {
                            std::cout << "  input token " << token << " (id " << input[token]
                                      << "): first residual difference >1e-6 at layer " << layer
                                      << ", max " << token_max << ", RMS " << token_rms << '\n';
                            break;
                        }
                    }
                }
                for (int layer = 0; layer < runner->layers; ++layer) {
                    int counts[4] = {};
                    for (int stage = 0; stage < 4; ++stage) {
                        const auto & a = baseline.first_quantized[4 * layer + stage];
                        const auto & b = batched.first_quantized[4 * layer + stage];
                        for (size_t j = 0; j < a.size(); ++j) counts[stage] += a[j] != b[j];
                    }
                    if (counts[0] || counts[1] || counts[2] || counts[3]) {
                        std::cout << "  input token 0 first INT8 differences at layer " << layer
                                  << " [QKV, attention output, gate/up, down] "
                                  << counts[0] << '/' << counts[1] << '/' << counts[2] << '/'
                                  << counts[3] << '\n';
                        break;
                    }
                }
                for (int layer = 0; layer < runner->layers; ++layer) {
                    auto [layer_max, layer_rms] = difference(baseline.layers[layer], batched.layers[layer]);
                    int quant_mismatches[4] = {};
                    for (int stage = 0; stage < 4; ++stage) {
                        const auto & a = baseline.quantized[4 * layer + stage];
                        const auto & b = batched.quantized[4 * layer + stage];
                        for (size_t j = 0; j < a.size(); ++j)
                            quant_mismatches[stage] += a[j] != b[j];
                    }
                    std::cout << "  layer " << layer << ": residual max abs difference "
                              << layer_max << ", RMS difference " << layer_rms
                              << ", INT8 differences [QKV, attention output, gate/up, down] "
                              << quant_mismatches[0] << '/' << quant_mismatches[1] << '/'
                              << quant_mismatches[2] << '/' << quant_mismatches[3] << '\n';
                }
                for (int i = 0; i < steps; ++i) {
                    auto [logit_max, logit_rms] = difference(baseline.logits[i], batched.logits[i]);
                    int id = baseline.ids[i];
                    float second = -INFINITY;
                    for (int j = 0; j < runner->vocab; ++j)
                        if (j != id) second = std::max(second, baseline.logits[i][j]);
                    std::cout << "  step " << i + 1 << ": baseline token " << id
                              << ", batched token " << batched.ids[i]
                              << ", baseline top-2 margin " << baseline.logits[i][id] - second
                              << ", logit max abs difference " << logit_max
                              << ", RMS difference " << logit_rms << '\n';
                }
            }
            print_optimization_summary(tokenize_only, use_dp4a, use_q6_dp4a,
                                       fuse_norm_quant, fuse_rope_cache, prefill_batch,
                                       gate_warps, q6_warps);
            print_supported_options(std::cerr);
            std::cerr << "Power mode: " << power_mode << '\n';
            return 0;
        }

        double sum_tokenize = 0, sum_prefill = 0, sum_decode = 0, sum_prompt = 0;
        double sum_gpu = 0, sum_decode_compute = 0, sum_prompt_joules = 0;
        u64 sum_input_tokens = 0, sum_output_tokens = 0, sum_decode_steps = 0;
        std::cerr << std::fixed << std::setprecision(3);
        for (size_t p = 0; p < prompts.size(); ++p) {
            double prompt_energy_start = power ? power->joules() : 0;
            auto prompt_started = Clock::now();
            auto input = chat_prompt(tokenizer, prompts[p]);
            auto tokenized = Clock::now();
            if (input.size() + max_tokens > static_cast<size_t>(context))
                throw std::runtime_error("Prompt " + std::to_string(p + 1) + " and response exceed --context");
            sum_tokenize += seconds(prompt_started, tokenized);
            sum_input_tokens += input.size();
            if (prompts.size() > 1) std::cout << "[" << p + 1 << "/" << prompts.size() << "] " << std::flush;
            if (tokenize_only) {
                for (int id : input) std::cout << id << ' ';
                std::cout << '\n';
                sum_prompt += seconds(prompt_started, Clock::now());
                continue;
            }

            runner->reset();
            double gpu_before = runner->gpu.gpu_seconds;
            u64 calls_before = runner->gpu.calls;
            auto prefill_started = Clock::now();
            runner->gpu.start_timer();
            int next = 0;
            if (prefill_batch > 1) next = runner->prefill(input);
            else for (size_t i = 0; i < input.size(); ++i)
                next = runner->forward(input[i], i, i + 1 == input.size());
            runner->gpu.stop_timer();
            auto prefill_finished = Clock::now();
            auto decode_started = Clock::now();
            int generated = 0, decode_steps = 0;
            double decode_compute = 0;
            for (int i = 0; i < max_tokens; ++i) {
                if (next == tokenizer.special("<|eot_id|>") ||
                    next == tokenizer.special("<|end_of_text|>") ||
                    next == tokenizer.special("<|eom_id|>")) break;
                std::cout << tokenizer.decode(next) << std::flush;
                ++generated;
                if (i + 1 == max_tokens || input.size() + i >= static_cast<size_t>(context)) break;
                auto step_started = Clock::now();
                runner->gpu.start_timer();
                next = runner->forward(next, input.size() + i, true);
                runner->gpu.stop_timer();
                decode_compute += seconds(step_started, Clock::now());
                ++decode_steps;
            }
            std::cout << '\n';
            auto finished = Clock::now();
            double prefill = seconds(prefill_started, prefill_finished);
            double decode = seconds(decode_started, finished);
            double prompt_time = seconds(prompt_started, finished);
            double gpu_time = runner->gpu.gpu_seconds - gpu_before;
            sum_prefill += prefill;
            sum_decode += decode;
            sum_prompt += prompt_time;
            sum_gpu += gpu_time;
            sum_output_tokens += generated;
            sum_decode_compute += decode_compute;
            sum_decode_steps += decode_steps;
            double prompt_joules = power ? power->joules() - prompt_energy_start : 0;
            if (std::isfinite(prompt_joules)) sum_prompt_joules += prompt_joules;
            if (stats) {
                std::cerr << "Prompt " << p + 1 << "/" << prompts.size()
                          << ": " << input.size() << " input tokens, " << generated << " output tokens\n"
                          << "  tokenization " << seconds(prompt_started, tokenized)
                          << " s; prompt computation " << prefill
                          << " s; response generation " << decode
                          << " s; total " << prompt_time << " s\n"
                          << "  output throughput " << generated / std::max(prompt_time, 1e-9)
                          << " tokens/s; steady decode "
                          << decode_steps / std::max(decode_compute, 1e-9) << " tokens/s\n"
                          << "  GPU pipeline " << gpu_time << " s across "
                          << runner->gpu.calls - calls_before << " quantized matrix calls\n";
                if (std::isfinite(prompt_joules) && prompt_joules > 0)
                    std::cerr << "  board energy " << prompt_joules << " J; output efficiency "
                              << generated / prompt_joules << " tokens/J\n";
            }
        }
        double run_joules = power ? power->joules() - run_energy_start : 0;
        if (stats) {
            auto finished = Clock::now();
            double n = static_cast<double>(prompts.size());
            std::cerr << "One-time GGUF metadata and mmap: " << seconds(program_started, model_loaded) << " s\n"
                      << "One-time tokenizer setup: " << seconds(model_loaded, tokenizer_ready) << " s\n";
            if (!tokenize_only) {
                std::cerr << "One-time GPU weight allocation and upload: "
                          << seconds(upload_started, upload_finished) << " s\n"
                          << "One-time runtime buffer setup: "
                          << seconds(upload_finished, runner_ready) << " s\n";
            }
            std::cerr << "Average over " << prompts.size() << " prompts:\n"
                      << "  tokenization: " << sum_tokenize / n << " s\n";
            if (!tokenize_only) {
                std::cerr << "  prompt computation: " << sum_prefill / n << " s\n"
                          << "  response generation: " << sum_decode / n << " s\n"
                          << "  inference per prompt (including tokenization): " << sum_prompt / n << " s\n"
                          << "  input processing: " << sum_input_tokens / std::max(sum_prefill, 1e-9)
                          << " tokens/s\n"
                          << "  output throughput: " << sum_output_tokens / std::max(sum_prompt, 1e-9)
                          << " tokens/s (all prompts combined)\n"
                          << "  steady decode: " << sum_decode_steps / std::max(sum_decode_compute, 1e-9)
                          << " tokens/s (all decode steps combined)\n"
                          << "  GPU pipeline: " << sum_gpu / n << " s\n";
                if (std::isfinite(run_joules) && run_joules > 0)
                    std::cerr << "  board energy per prompt (excluding loading): "
                              << sum_prompt_joules / n << " J\n"
                              << "  output efficiency (excluding loading): "
                              << sum_output_tokens / std::max(sum_prompt_joules, 1e-9)
                              << " tokens/J\n";
            }
            std::cerr << "  end-to-end per prompt (including share of one-time loading): "
                      << seconds(program_started, finished) / n << " s\n";
            if (!tokenize_only)
                std::cerr << "  end-to-end output throughput (including loading): "
                          << sum_output_tokens / std::max(seconds(program_started, finished), 1e-9)
                          << " tokens/s\n";
            if (!tokenize_only && std::isfinite(run_joules) && run_joules > 0)
                std::cerr << "  total board energy (including loading): " << run_joules << " J\n"
                          << "  average board input power: "
                          << run_joules / std::max(seconds(program_started, finished), 1e-9) << " W\n"
                          << "  end-to-end output efficiency (including loading): "
                          << sum_output_tokens / run_joules << " tokens/J\n";
            else if (!tokenize_only)
                std::cerr << "  board energy unavailable (VDD_IN sensor not readable)\n";
            std::cerr << "Total: " << sum_input_tokens << " input tokens, "
                      << sum_output_tokens << " output tokens, "
                      << seconds(program_started, finished) << " s elapsed\n";
        } else {
            if (!tokenize_only && std::isfinite(run_joules) && run_joules > 0)
                std::cerr << "Board energy: " << run_joules << " J; output efficiency: "
                          << sum_output_tokens / run_joules << " tokens/J\n";
        }
        print_optimization_summary(tokenize_only, use_dp4a, use_q6_dp4a,
                                   fuse_norm_quant, fuse_rope_cache, prefill_batch,
                                   gate_warps, q6_warps);
        print_supported_options(std::cerr);
        std::cerr << "Power mode: " << power_mode << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
