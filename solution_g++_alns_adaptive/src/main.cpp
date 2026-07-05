#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct ServerSpec {
    int server_id = 0;
    int gpu_count = 0;
    int gpu_memory = 0;
    int cpu_cores = 0;
    int memory = 0;
    double power_score = 0.0;
};

struct Job {
    int job_id = 0;
    long long release_time = 0;
    long long duration = 0;
    int min_gpu = 0;
    int gpu_memory = 0;
    int cpu_cores = 0;
    int memory = 0;
    int weight = 0;
    double demand_score = 0.0;
};

struct ScheduleRecord {
    int job_id = 0;
    int server_id = 0;
    long long start_time = 0;
    int gpu_used = 0;
    long long finish_time = 0;

    ScheduleRecord() = default;
    ScheduleRecord(int job_id, int server_id, long long start_time, int gpu_used, long long finish_time)
        : job_id(job_id),
          server_id(server_id),
          start_time(start_time),
          gpu_used(gpu_used),
          finish_time(finish_time) {}
};

struct Placement {
    int job_id = 0;
    long long start = 0;
    long long end = 0;
    int gpu_used = 0;
    int cpu_used = 0;
    int memory_used = 0;

    Placement() = default;
    Placement(int job_id, long long start, long long end, int gpu_used, int cpu_used, int memory_used)
        : job_id(job_id),
          start(start),
          end(end),
          gpu_used(gpu_used),
          cpu_used(cpu_used),
          memory_used(memory_used) {}
};

struct FeasibleEntry {
    int server_index = 0;
    int gpu_used = 0;
    int memory_waste = 0;

    FeasibleEntry() = default;
    FeasibleEntry(int server_index, int gpu_used, int memory_waste)
        : server_index(server_index),
          gpu_used(gpu_used),
          memory_waste(memory_waste) {}
};

struct Candidate {
    bool valid = false;
    int server_index = -1;
    int gpu_used = 0;
    long long start = 0;
    long long finish = 0;
    double cost = numeric_limits<double>::infinity();
    double tetris_score = 0.0;

    Candidate() = default;
    Candidate(
        bool valid,
        int server_index,
        int gpu_used,
        long long start,
        long long finish,
        double cost,
        double tetris_score = 0.0
    )
        : valid(valid),
          server_index(server_index),
          gpu_used(gpu_used),
          start(start),
          finish(finish),
          cost(cost),
          tetris_score(tetris_score) {}
};

struct StrategyConfig {
    string name;
    int order_type = 0;
    double wait_weight = 1.0;
    double finish_weight = 0.02;
    double memory_weight = 0.02;
    double fragmentation_weight = 0.01;
    double scarcity_weight = 0.01;
    double tail_weight = 0.05;

    StrategyConfig() = default;
    StrategyConfig(
        string name,
        int order_type,
        double wait_weight,
        double finish_weight,
        double memory_weight,
        double fragmentation_weight,
        double scarcity_weight,
        double tail_weight
    ) : name(move(name)),
        order_type(order_type),
        wait_weight(wait_weight),
        finish_weight(finish_weight),
        memory_weight(memory_weight),
        fragmentation_weight(fragmentation_weight),
        scarcity_weight(scarcity_weight),
        tail_weight(tail_weight) {}
};

struct Solution {
    vector<ScheduleRecord> records;
    double score = numeric_limits<double>::infinity();
    bool valid = false;
};

struct RawMetrics {
    long double weighted_wait = 0.0L;
    long double memory_waste = 0.0L;
    long long max_finish = 0;
};

struct ResourceFree {
    int gpu = 0;
    int cpu = 0;
    int memory = 0;
};

struct TuningParams {
    double wait_weight = 1.0;
    double memory_weight = 1.0;
    double finish_weight = 1.0;
    double fragment_weight = 1.0;
    double scarcity_weight = 1.0;
    double tail_weight = 1.0;
    double duration_bias = 0.0001;
    double random_repair_ratio = 0.25;
    int repair_rounds = 60;
    int candidate_limit = 0;
    double local_search_time_small = 1.2;
    double local_search_time_medium = 2.5;
    double local_search_time_large = 5.0;
    double local_search_time_huge = 8.0;
    double proxy_wait_weight = 1.0;
    double proxy_memory_weight = 1.0;
    double proxy_finish_weight = 1.0;
    double tetris_weight = 0.0;
    double tetris_tie_epsilon = 0.0;
    int late_acceptance_window = 0;
    double late_acceptance_margin = 0.0;
    int tabu_tenure = 0;
    int tabu_block_tenure = 0;
    int small_window_repair_rounds = 20;
    int small_window_repair_width = 0;
    int small_window_repair_cap = 64;
    int regret_repair_enabled = 1;
    int regret_repair_k = 3;
    double regret_repair_weight = 2.0;
    int regret_repair_candidate_limit = 32;
    int regret_repair_scope = 0;
    int regret_repair_min_block = 0;
    int regret_repair_max_block = 48;
    int adaptive_destroy_enabled = 1;
    int adaptive_repair_enabled = 1;
    int adaptive_segment_length = 12;
    double adaptive_learning_rate = 0.25;
    double adaptive_min_weight = 0.20;
    double adaptive_max_weight = 5.0;
    double adaptive_global_best_reward = 5.0;
    double adaptive_current_improve_reward = 2.0;
    double adaptive_initial_weight = 1.0;
};

struct PortfolioConfig {
    string name;
    TuningParams params;
    vector<int> enabled_order_types;
    double time_fraction = 1.0;
};

static void setTuningParam(TuningParams &params, const string &key, double value) {
    if (key == "wait_weight") params.wait_weight = value;
    else if (key == "memory_weight") params.memory_weight = value;
    else if (key == "finish_weight") params.finish_weight = value;
    else if (key == "fragment_weight") params.fragment_weight = value;
    else if (key == "scarcity_weight") params.scarcity_weight = value;
    else if (key == "tail_weight") params.tail_weight = value;
    else if (key == "duration_bias") params.duration_bias = value;
    else if (key == "random_repair_ratio") params.random_repair_ratio = value;
    else if (key == "repair_rounds") params.repair_rounds = max(0, static_cast<int>(value + 0.5));
    else if (key == "candidate_limit") params.candidate_limit = max(0, static_cast<int>(value + 0.5));
    else if (key == "local_search_time_small") params.local_search_time_small = value;
    else if (key == "local_search_time_medium") params.local_search_time_medium = value;
    else if (key == "local_search_time_large") params.local_search_time_large = value;
    else if (key == "local_search_time_huge") params.local_search_time_huge = value;
    else if (key == "proxy_wait_weight") params.proxy_wait_weight = value;
    else if (key == "proxy_memory_weight") params.proxy_memory_weight = value;
    else if (key == "proxy_finish_weight") params.proxy_finish_weight = value;
    else if (key == "tetris_weight") params.tetris_weight = value;
    else if (key == "tetris_tie_epsilon") params.tetris_tie_epsilon = value;
    else if (key == "late_acceptance_window") params.late_acceptance_window = max(0, static_cast<int>(value + 0.5));
    else if (key == "late_acceptance_margin") params.late_acceptance_margin = value;
    else if (key == "tabu_tenure") params.tabu_tenure = max(0, static_cast<int>(value + 0.5));
    else if (key == "tabu_block_tenure") params.tabu_block_tenure = max(0, static_cast<int>(value + 0.5));
    else if (key == "small_window_repair_rounds") params.small_window_repair_rounds = max(0, static_cast<int>(value + 0.5));
    else if (key == "small_window_repair_width") params.small_window_repair_width = max(0, static_cast<int>(value + 0.5));
    else if (key == "small_window_repair_cap") params.small_window_repair_cap = max(4, static_cast<int>(value + 0.5));
    else if (key == "regret_repair_enabled") params.regret_repair_enabled = value > 0.0 ? 1 : 0;
    else if (key == "regret_repair_k") params.regret_repair_k = max(0, static_cast<int>(value + 0.5));
    else if (key == "regret_repair_weight") params.regret_repair_weight = value;
    else if (key == "regret_repair_candidate_limit") params.regret_repair_candidate_limit = max(0, static_cast<int>(value + 0.5));
    else if (key == "regret_repair_scope") params.regret_repair_scope = max(0, static_cast<int>(value + 0.5));
    else if (key == "regret_repair_min_block") params.regret_repair_min_block = max(0, static_cast<int>(value + 0.5));
    else if (key == "regret_repair_max_block") params.regret_repair_max_block = max(0, static_cast<int>(value + 0.5));
    else if (key == "adaptive_destroy_enabled") params.adaptive_destroy_enabled = value > 0.0 ? 1 : 0;
    else if (key == "adaptive_repair_enabled") params.adaptive_repair_enabled = value > 0.0 ? 1 : 0;
    else if (key == "adaptive_segment_length") params.adaptive_segment_length = max(1, static_cast<int>(value + 0.5));
    else if (key == "adaptive_learning_rate") params.adaptive_learning_rate = value;
    else if (key == "adaptive_min_weight") params.adaptive_min_weight = value;
    else if (key == "adaptive_max_weight") params.adaptive_max_weight = value;
    else if (key == "adaptive_global_best_reward") params.adaptive_global_best_reward = value;
    else if (key == "adaptive_current_improve_reward") params.adaptive_current_improve_reward = value;
    else if (key == "adaptive_initial_weight") params.adaptive_initial_weight = value;
}

static void skipJsonWhitespace(const string &text, size_t &pos) {
    while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
}

static bool loadTuningParamsFromJson(const string &path, TuningParams &params) {
    ifstream input(path);
    if (!input) {
        return false;
    }

    string text((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    size_t pos = 0;
    skipJsonWhitespace(text, pos);
    if (pos < text.size() && text[pos] == '{') {
        ++pos;
    }

    while (pos < text.size()) {
        skipJsonWhitespace(text, pos);
        if (pos >= text.size() || text[pos] == '}') {
            break;
        }
        if (text[pos] != '"') {
            return false;
        }
        ++pos;
        size_t key_start = pos;
        while (pos < text.size() && text[pos] != '"') {
            ++pos;
        }
        if (pos >= text.size()) {
            return false;
        }
        string key = text.substr(key_start, pos - key_start);
        ++pos;

        skipJsonWhitespace(text, pos);
        if (pos >= text.size() || text[pos] != ':') {
            return false;
        }
        ++pos;
        skipJsonWhitespace(text, pos);

        const char *number_start = text.c_str() + pos;
        char *number_end = nullptr;
        double value = strtod(number_start, &number_end);
        if (number_end == number_start) {
            return false;
        }
        pos = static_cast<size_t>(number_end - text.c_str());
        setTuningParam(params, key, value);

        skipJsonWhitespace(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
        }
    }

    params.random_repair_ratio = min(1.0, max(0.0, params.random_repair_ratio));
    params.local_search_time_small = max(0.0, params.local_search_time_small);
    params.local_search_time_medium = max(0.0, params.local_search_time_medium);
    params.local_search_time_large = max(0.0, params.local_search_time_large);
    params.local_search_time_huge = max(0.0, params.local_search_time_huge);
    params.tetris_weight = max(0.0, params.tetris_weight);
    params.tetris_tie_epsilon = max(0.0, params.tetris_tie_epsilon);
    params.late_acceptance_margin = max(0.0, params.late_acceptance_margin);
    params.small_window_repair_rounds = max(0, params.small_window_repair_rounds);
    params.small_window_repair_width = max(0, params.small_window_repair_width);
    params.small_window_repair_cap = max(4, params.small_window_repair_cap);
    params.regret_repair_enabled = params.regret_repair_enabled > 0 ? 1 : 0;
    params.regret_repair_k = max(0, params.regret_repair_k);
    params.regret_repair_weight = max(0.0, params.regret_repair_weight);
    params.regret_repair_candidate_limit = max(0, params.regret_repair_candidate_limit);
    params.regret_repair_scope = min(2, max(0, params.regret_repair_scope));
    params.regret_repair_min_block = max(0, params.regret_repair_min_block);
    params.regret_repair_max_block = max(0, params.regret_repair_max_block);
    params.adaptive_destroy_enabled = params.adaptive_destroy_enabled > 0 ? 1 : 0;
    params.adaptive_repair_enabled = params.adaptive_repair_enabled > 0 ? 1 : 0;
    params.adaptive_segment_length = max(1, params.adaptive_segment_length);
    params.adaptive_learning_rate = min(1.0, max(0.0, params.adaptive_learning_rate));
    params.adaptive_min_weight = max(0.0, params.adaptive_min_weight);
    params.adaptive_max_weight = max(params.adaptive_min_weight, params.adaptive_max_weight);
    params.adaptive_global_best_reward = max(0.0, params.adaptive_global_best_reward);
    params.adaptive_current_improve_reward = max(0.0, params.adaptive_current_improve_reward);
    params.adaptive_initial_weight = max(params.adaptive_min_weight, params.adaptive_initial_weight);
    return true;
}

static TuningParams loadTuningParams() {
    TuningParams params;
    const char *config_path = getenv("SCHED_CONFIG");
    if (config_path != nullptr && config_path[0] != '\0') {
        loadTuningParamsFromJson(config_path, params);
    }
    return params;
}

static bool portfolioSingleMode() {
    const char *value = getenv("PORTFOLIO_SINGLE");
    return value != nullptr && value[0] != '\0' && string(value) != "0";
}

class ServerTimeline {
public:
    explicit ServerTimeline(const ServerSpec *server_spec = nullptr) : spec(server_spec) {}

    int requiredGpuCount(const Job &job) const {
        int by_memory = static_cast<int>((job.gpu_memory + spec->gpu_memory - 1LL) / spec->gpu_memory);
        return max(job.min_gpu, by_memory);
    }

    bool canEverRun(const Job &job, int gpu_used) const {
        return gpu_used <= spec->gpu_count &&
               job.cpu_cores <= spec->cpu_cores &&
               job.memory <= spec->memory;
    }

    bool feasibleAt(const Job &job, int gpu_used, long long start) const {
        long long finish = start + job.duration;
        vector<long long> points;
        points.reserve(placements.size() * 2 + 2);
        points.push_back(start);
        points.push_back(finish);

        for (const Placement &placement : placements) {
            if (placement.end <= start) {
                continue;
            }
            if (placement.start >= finish) {
                break;
            }
            points.push_back(max(start, placement.start));
            points.push_back(min(finish, placement.end));
        }

        sort(points.begin(), points.end());
        points.erase(unique(points.begin(), points.end()), points.end());

        for (size_t idx = 0; idx + 1 < points.size(); ++idx) {
            long long segment_start = points[idx];
            long long segment_end = points[idx + 1];
            if (segment_start >= segment_end) {
                continue;
            }

            int used_gpu = 0;
            int used_cpu = 0;
            int used_memory = 0;

            for (const Placement &placement : placements) {
                if (placement.end <= segment_start) {
                    continue;
                }
                if (placement.start >= segment_end) {
                    break;
                }
                used_gpu += placement.gpu_used;
                used_cpu += placement.cpu_used;
                used_memory += placement.memory_used;
            }

            if (used_gpu + gpu_used > spec->gpu_count ||
                used_cpu + job.cpu_cores > spec->cpu_cores ||
                used_memory + job.memory > spec->memory) {
                return false;
            }
        }

        return true;
    }

    ResourceFree minFreeResources(long long start, long long finish) const {
        ResourceFree result{spec->gpu_count, spec->cpu_cores, spec->memory};
        if (start >= finish) {
            return result;
        }

        vector<long long> points;
        points.reserve(placements.size() * 2 + 2);
        points.push_back(start);
        points.push_back(finish);

        for (const Placement &placement : placements) {
            if (placement.end <= start) {
                continue;
            }
            if (placement.start >= finish) {
                break;
            }
            points.push_back(max(start, placement.start));
            points.push_back(min(finish, placement.end));
        }

        sort(points.begin(), points.end());
        points.erase(unique(points.begin(), points.end()), points.end());

        for (size_t idx = 0; idx + 1 < points.size(); ++idx) {
            long long segment_start = points[idx];
            long long segment_end = points[idx + 1];
            if (segment_start >= segment_end) {
                continue;
            }

            int used_gpu = 0;
            int used_cpu = 0;
            int used_memory = 0;

            for (const Placement &placement : placements) {
                if (placement.end <= segment_start) {
                    continue;
                }
                if (placement.start >= segment_end) {
                    break;
                }
                used_gpu += placement.gpu_used;
                used_cpu += placement.cpu_used;
                used_memory += placement.memory_used;
            }

            result.gpu = min(result.gpu, spec->gpu_count - used_gpu);
            result.cpu = min(result.cpu, spec->cpu_cores - used_cpu);
            result.memory = min(result.memory, spec->memory - used_memory);
        }

        result.gpu = max(0, result.gpu);
        result.cpu = max(0, result.cpu);
        result.memory = max(0, result.memory);
        return result;
    }

    long long earliestStart(const Job &job, int gpu_used) const {
        vector<long long> candidates;
        candidates.reserve(placements.size() + 1);
        candidates.push_back(job.release_time);

        for (const Placement &placement : placements) {
            if (placement.end >= job.release_time) {
                candidates.push_back(placement.end);
            }
        }

        sort(candidates.begin(), candidates.end());
        candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

        for (long long start : candidates) {
            if (feasibleAt(job, gpu_used, start)) {
                return start;
            }
        }

        long long fallback = candidates.empty() ? job.release_time : candidates.back();
        while (!feasibleAt(job, gpu_used, fallback)) {
            long long next_time = -1;
            for (const Placement &placement : placements) {
                if (placement.end > fallback && (next_time == -1 || placement.end < next_time)) {
                    next_time = placement.end;
                }
            }
            if (next_time == -1) {
                fallback = max(fallback + 1, job.release_time);
            } else {
                fallback = next_time;
            }
        }
        return fallback;
    }

    void addPlacement(const Placement &placement) {
        auto pos = lower_bound(
            placements.begin(),
            placements.end(),
            placement.start,
            [](const Placement &lhs, long long value) {
                return lhs.start < value;
            }
        );
        placements.insert(pos, placement);
    }

private:
    const ServerSpec *spec = nullptr;
    vector<Placement> placements;
};

class Scheduler {
public:
    Scheduler(vector<ServerSpec> input_servers, vector<Job> input_jobs, TuningParams tuning_params)
        : servers(move(input_servers)), jobs(move(input_jobs)),
          base_params(tuning_params), params(tuning_params) {
        sort(servers.begin(), servers.end(), [](const ServerSpec &a, const ServerSpec &b) {
            return a.server_id < b.server_id;
        });
        sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) {
            return a.job_id < b.job_id;
        });

        int max_gpu = 1;
        int max_gpu_memory = 1;
        int max_cpu = 1;
        int max_memory = 1;
        for (const ServerSpec &server : servers) {
            max_gpu = max(max_gpu, server.gpu_count);
            max_gpu_memory = max(max_gpu_memory, server.gpu_memory);
            max_cpu = max(max_cpu, server.cpu_cores);
            max_memory = max(max_memory, server.memory);
            total_gpu_memory += 1LL * server.gpu_count * server.gpu_memory;
        }

        for (ServerSpec &server : servers) {
            server.power_score =
                0.35 * static_cast<double>(server.gpu_count) / max_gpu +
                0.30 * static_cast<double>(server.gpu_memory) / max_gpu_memory +
                0.20 * static_cast<double>(server.cpu_cores) / max_cpu +
                0.15 * static_cast<double>(server.memory) / max_memory;
        }

        for (Job &job : jobs) {
            job.demand_score =
                0.35 * static_cast<double>(job.min_gpu) / max_gpu +
                0.30 * static_cast<double>(job.gpu_memory) / max(1, max_gpu_memory * max_gpu) +
                0.20 * static_cast<double>(job.cpu_cores) / max_cpu +
                0.15 * static_cast<double>(job.memory) / max_memory;
            total_weight += max(1, job.weight);
            total_duration += job.duration;
            total_job_gpu_memory_time += 1LL * job.gpu_memory * job.duration;
            min_release = min(min_release, job.release_time);
        }
        if (jobs.empty()) {
            min_release = 0;
        }

        buildFeasibleEntries();
    }

    Solution solve() {
        auto deadline = chrono::steady_clock::now() + chrono::milliseconds(52000);

        if (portfolioSingleMode()) {
            params = base_params;
            return solveWithCurrentParams({}, deadline, 1.0);
        }

        vector<PortfolioConfig> portfolio = selectedPortfolioConfigs();
        vector<Solution> candidates;
        candidates.reserve(portfolio.size());

        for (const PortfolioConfig &config : portfolio) {
            if (secondsLeft(deadline) < 0.25) {
                break;
            }

            params = sizedParams(config.params);
            Solution candidate = solveWithCurrentParams(config.enabled_order_types, deadline, config.time_fraction);
            if (candidate.valid) {
                candidates.push_back(move(candidate));
            }
        }

        if (candidates.empty()) {
            params = base_params;
            return solveWithCurrentParams({}, deadline, 0.0);
        }

        params = base_params;
        initializeMetricScales(candidates);

        Solution best;
        for (Solution &candidate : candidates) {
            if (candidate.valid && candidate.score < best.score) {
                best = move(candidate);
            }
        }

        return best;
    }

private:
    double secondsLeft(chrono::steady_clock::time_point deadline) const {
        return chrono::duration<double>(deadline - chrono::steady_clock::now()).count();
    }

    vector<StrategyConfig> strategiesForOrderTypes(const vector<int> &enabled_order_types) const {
        vector<StrategyConfig> all = buildStrategies();
        if (enabled_order_types.empty()) {
            return all;
        }

        vector<StrategyConfig> selected;
        selected.reserve(enabled_order_types.size());
        for (int order_type : enabled_order_types) {
            for (const StrategyConfig &strategy : all) {
                if (strategy.order_type == order_type) {
                    selected.push_back(strategy);
                    break;
                }
            }
        }
        return selected.empty() ? all : selected;
    }

    TuningParams sizedParams(TuningParams tuned) const {
        int n = static_cast<int>(jobs.size());
        auto scale_times = [&](double scale) {
            tuned.local_search_time_small *= scale;
            tuned.local_search_time_medium *= scale;
            tuned.local_search_time_large *= scale;
            tuned.local_search_time_huge *= scale;
        };

        if (n <= 250) {
            scale_times(1.35);
            if (tuned.repair_rounds == 0) tuned.repair_rounds = 100;
        } else if (n <= 1000) {
            scale_times(1.00);
            if (tuned.repair_rounds == 0) tuned.repair_rounds = 70;
        } else if (n <= 3000) {
            scale_times(0.70);
            if (tuned.repair_rounds == 0) tuned.repair_rounds = 35;
            if (tuned.candidate_limit == 0) tuned.candidate_limit = 60;
        } else {
            scale_times(0.50);
            tuned.local_search_time_huge = min(tuned.local_search_time_huge, 4.2);
            if (tuned.repair_rounds == 0) tuned.repair_rounds = 18;
            if (tuned.candidate_limit == 0) tuned.candidate_limit = 36;
        }

        return tuned;
    }

    vector<PortfolioConfig> buildPortfolioConfigs() const {
        vector<PortfolioConfig> configs;
        auto add = [&](string name, vector<int> order_types, double time_fraction, double wait_mul,
                       double memory_mul, double finish_mul, double fragment_mul, double scarcity_mul,
                       double tail_mul, double proxy_wait_mul, double proxy_memory_mul,
                       double proxy_finish_mul, double random_ratio) {
            TuningParams tuned = base_params;
            tuned.wait_weight *= wait_mul;
            tuned.memory_weight *= memory_mul;
            tuned.finish_weight *= finish_mul;
            tuned.fragment_weight *= fragment_mul;
            tuned.scarcity_weight *= scarcity_mul;
            tuned.tail_weight *= tail_mul;
            tuned.proxy_wait_weight *= proxy_wait_mul;
            tuned.proxy_memory_weight *= proxy_memory_mul;
            tuned.proxy_finish_weight *= proxy_finish_mul;
            tuned.random_repair_ratio = min(1.0, max(0.0, random_ratio));
            PortfolioConfig config;
            config.name = move(name);
            config.params = tuned;
            config.enabled_order_types = move(order_types);
            config.time_fraction = time_fraction;
            configs.push_back(move(config));
        };

        add("wait_heavy", {1, 0, 5, 2}, 0.90, 1.75, 0.70, 0.80, 0.85, 0.85, 0.80, 1.45, 0.75, 0.80, 0.18);
        add("memory_heavy", {6, 2, 5, 0}, 0.85, 0.95, 1.85, 0.80, 1.15, 0.95, 0.85, 0.85, 1.55, 0.85, 0.22);
        add("finish_heavy", {4, 3, 5, 2}, 0.85, 0.80, 0.75, 1.55, 0.85, 0.95, 1.90, 0.80, 0.80, 1.55, 0.20);
        add("scarce_job_first", {2, 3, 5, 6}, 0.80, 1.10, 1.15, 1.00, 1.00, 1.85, 1.10, 1.05, 1.05, 1.00, 0.16);
        add("short_job_first", {1, 0, 5}, 0.75, 1.55, 0.90, 0.85, 0.90, 0.85, 0.75, 1.25, 0.90, 0.85, 0.24);
        add("long_job_first", {4, 3, 2}, 0.75, 0.85, 0.85, 1.35, 0.90, 1.00, 1.65, 0.85, 0.85, 1.35, 0.18);
        add("balanced_safe", {5, 1, 6, 2, 0, 3, 4}, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.25);
        return configs;
    }

    vector<PortfolioConfig> selectedPortfolioConfigs() const {
        vector<PortfolioConfig> all = buildPortfolioConfigs();
        int n = static_cast<int>(jobs.size());
        if (n <= 1000) {
            return all;
        }

        vector<string> names;
        if (n <= 3000) {
            names = {"wait_heavy", "memory_heavy", "finish_heavy", "scarce_job_first", "balanced_safe"};
        } else {
            names = {"scarce_job_first", "memory_heavy", "wait_heavy", "balanced_safe"};
        }

        vector<PortfolioConfig> selected;
        selected.reserve(names.size());
        for (const string &name : names) {
            for (const PortfolioConfig &config : all) {
                if (config.name == name) {
                    selected.push_back(config);
                    break;
                }
            }
        }
        return selected;
    }

    Solution solveWithCurrentParams(
        const vector<int> &enabled_order_types,
        chrono::steady_clock::time_point deadline,
        double local_time_fraction
    ) {
        vector<StrategyConfig> strategies = strategiesForOrderTypes(enabled_order_types);
        vector<Solution> initial_solutions;
        initial_solutions.reserve(strategies.size());

        for (const StrategyConfig &strategy : strategies) {
            if (secondsLeft(deadline) < 0.15) {
                break;
            }
            Solution candidate = buildGreedySolution(strategy, {});
            if (candidate.valid) {
                initial_solutions.push_back(move(candidate));
            }
        }

        if (initial_solutions.empty()) {
            return Solution{};
        }

        initializeMetricScales(initial_solutions);
        Solution best;
        for (Solution &candidate : initial_solutions) {
            if (candidate.valid && candidate.score < best.score) {
                best = move(candidate);
            }
        }

        if (best.valid && local_time_fraction > 0.0 && secondsLeft(deadline) > 0.15) {
            improveLocally(best, deadline, local_time_fraction);
        }
        return best;
    }

    vector<StrategyConfig> buildStrategies() const {
        return {
            {"release", 0, 1.30, 0.020, 0.020, 0.010, 0.010, 0.040},
            {"weighted_short", 1, 1.90, 0.015, 0.015, 0.010, 0.010, 0.030},
            {"scarcity", 2, 1.40, 0.020, 0.020, 0.012, 0.004, 0.050},
            {"large_resource", 3, 1.10, 0.025, 0.015, 0.006, 0.004, 0.070},
            {"long_duration", 4, 0.95, 0.035, 0.010, 0.005, 0.006, 0.100},
            {"hybrid", 5, 1.55, 0.025, 0.018, 0.008, 0.006, 0.070},
            {"memory_fit", 6, 1.25, 0.018, 0.045, 0.016, 0.006, 0.045},
        };
    }

    void buildFeasibleEntries() {
        feasible_by_job.assign(jobs.size() + 1, {});

        for (const Job &job : jobs) {
            vector<FeasibleEntry> entries;
            entries.reserve(servers.size());

            for (int idx = 0; idx < static_cast<int>(servers.size()); ++idx) {
                const ServerSpec &server = servers[idx];
                int gpu_used = max(
                    job.min_gpu,
                    static_cast<int>((job.gpu_memory + server.gpu_memory - 1LL) / server.gpu_memory)
                );
                if (gpu_used <= server.gpu_count &&
                    job.cpu_cores <= server.cpu_cores &&
                    job.memory <= server.memory) {
                    entries.push_back(FeasibleEntry{
                        idx,
                        gpu_used,
                        gpu_used * server.gpu_memory - job.gpu_memory,
                    });
                }
            }

            sort(entries.begin(), entries.end(), [&](const FeasibleEntry &a, const FeasibleEntry &b) {
                if (a.memory_waste != b.memory_waste) {
                    return a.memory_waste < b.memory_waste;
                }
                if (a.gpu_used != b.gpu_used) {
                    return a.gpu_used < b.gpu_used;
                }
                return servers[a.server_index].server_id < servers[b.server_index].server_id;
            });

            feasible_by_job[job.job_id] = move(entries);
        }
    }

    vector<int> orderedJobs(const StrategyConfig &strategy, const vector<char> &skip) const {
        vector<int> order;
        order.reserve(jobs.size());
        for (const Job &job : jobs) {
            if (skip.empty() || !skip[job.job_id]) {
                order.push_back(job.job_id);
            }
        }

        sort(order.begin(), order.end(), [&](int lhs_id, int rhs_id) {
            const Job &a = jobs[lhs_id - 1];
            const Job &b = jobs[rhs_id - 1];

            auto scarcity_a = feasible_by_job[a.job_id].size();
            auto scarcity_b = feasible_by_job[b.job_id].size();

            switch (strategy.order_type) {
                case 0:
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    if (a.duration != b.duration) return a.duration < b.duration;
                    return a.job_id < b.job_id;
                case 1: {
                    double ka = static_cast<double>(max(1, a.weight)) / max(1LL, a.duration);
                    double kb = static_cast<double>(max(1, b.weight)) / max(1LL, b.duration);
                    if (fabs(ka - kb) > 1e-12) return ka > kb;
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    return a.job_id < b.job_id;
                }
                case 2:
                    if (scarcity_a != scarcity_b) return scarcity_a < scarcity_b;
                    if (a.weight != b.weight) return a.weight > b.weight;
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    return a.job_id < b.job_id;
                case 3:
                    if (fabs(a.demand_score - b.demand_score) > 1e-12) return a.demand_score > b.demand_score;
                    if (a.weight != b.weight) return a.weight > b.weight;
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    return a.job_id < b.job_id;
                case 4:
                    if (a.duration != b.duration) return a.duration > b.duration;
                    if (a.weight != b.weight) return a.weight > b.weight;
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    return a.job_id < b.job_id;
                case 5: {
                    double ka =
                        2.5 * static_cast<double>(max(1, a.weight)) / max(1LL, a.duration) +
                        1.5 / max<size_t>(1, scarcity_a) +
                        1.2 * a.demand_score;
                    double kb =
                        2.5 * static_cast<double>(max(1, b.weight)) / max(1LL, b.duration) +
                        1.5 / max<size_t>(1, scarcity_b) +
                        1.2 * b.demand_score;
                    if (fabs(ka - kb) > 1e-12) return ka > kb;
                    if (a.release_time != b.release_time) return a.release_time < b.release_time;
                    return a.job_id < b.job_id;
                }
                case 6: {
                    int wa = feasible_by_job[a.job_id].empty() ? 0 : feasible_by_job[a.job_id].front().memory_waste;
                    int wb = feasible_by_job[b.job_id].empty() ? 0 : feasible_by_job[b.job_id].front().memory_waste;
                    if (wa != wb) return wa > wb;
                    if (scarcity_a != scarcity_b) return scarcity_a < scarcity_b;
                    if (a.weight != b.weight) return a.weight > b.weight;
                    return a.job_id < b.job_id;
                }
                default:
                    return a.job_id < b.job_id;
            }
        });

        return order;
    }

    double tetrisScore(
        const Job &job,
        const ServerTimeline &timeline,
        const ServerSpec &server,
        int gpu_used,
        long long start,
        long long finish
    ) const {
        ResourceFree free = timeline.minFreeResources(start, finish);
        double server_gpu = max(1, server.gpu_count);
        double server_cpu = max(1, server.cpu_cores);
        double server_memory = max(1, server.memory);
        double server_vmem = max(1.0, static_cast<double>(server.gpu_count) * server.gpu_memory);

        double demand_gpu = static_cast<double>(gpu_used) / server_gpu;
        double demand_cpu = static_cast<double>(job.cpu_cores) / server_cpu;
        double demand_memory = static_cast<double>(job.memory) / server_memory;
        double demand_vmem = static_cast<double>(gpu_used) * server.gpu_memory / server_vmem;

        double free_gpu = static_cast<double>(free.gpu) / server_gpu;
        double free_cpu = static_cast<double>(free.cpu) / server_cpu;
        double free_memory = static_cast<double>(free.memory) / server_memory;
        double free_vmem = static_cast<double>(free.gpu) * server.gpu_memory / server_vmem;

        return demand_gpu * free_gpu +
               demand_cpu * free_cpu +
               demand_memory * free_memory +
               demand_vmem * free_vmem;
    }

    bool candidateLess(const Candidate &lhs, const Candidate &rhs, const Job &job) const {
        if (!rhs.valid) {
            return lhs.valid;
        }
        if (!lhs.valid) {
            return false;
        }
        if (fabs(lhs.cost - rhs.cost) > max(1e-9, params.tetris_tie_epsilon)) {
            return lhs.cost < rhs.cost;
        }
        if (lhs.start != rhs.start) {
            return lhs.start < rhs.start;
        }
        if (lhs.finish != rhs.finish) {
            return lhs.finish < rhs.finish;
        }
        int lhs_waste = lhs.gpu_used * servers[lhs.server_index].gpu_memory - job.gpu_memory;
        int rhs_waste = rhs.gpu_used * servers[rhs.server_index].gpu_memory - job.gpu_memory;
        if (lhs_waste != rhs_waste) {
            return lhs_waste < rhs_waste;
        }
        if ((params.tetris_weight > 0.0 || params.tetris_tie_epsilon > 0.0) &&
            fabs(lhs.tetris_score - rhs.tetris_score) > 1e-12) {
            return lhs.tetris_score > rhs.tetris_score;
        }
        return servers[lhs.server_index].server_id < servers[rhs.server_index].server_id;
    }

    vector<Candidate> topCandidates(
        const Job &job,
        const vector<ServerTimeline> &timelines,
        const StrategyConfig &strategy,
        long long current_max_finish,
        int keep_count,
        int entry_limit
    ) const {
        vector<Candidate> candidates;
        if (keep_count <= 0) {
            return candidates;
        }
        const vector<FeasibleEntry> &entries = feasible_by_job[job.job_id];
        int entry_count = static_cast<int>(entries.size());
        if (params.candidate_limit > 0) {
            entry_count = min(entry_count, params.candidate_limit);
        }
        if (entry_limit > 0) {
            entry_count = min(entry_count, entry_limit);
        }
        candidates.reserve(min(entry_count, keep_count));

        for (int entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
            const FeasibleEntry &entry = entries[entry_idx];
            const ServerSpec &server = servers[entry.server_index];
            long long start = timelines[entry.server_index].earliestStart(job, entry.gpu_used);
            long long finish = start + job.duration;

            double wait_cost = static_cast<double>(start - job.release_time) * max(1, job.weight);
            double finish_cost = static_cast<double>(finish);
            double memory_cost = static_cast<double>(entry.memory_waste) * static_cast<double>(max(1LL, job.duration));
            double leftover_gpu = static_cast<double>(server.gpu_count - entry.gpu_used) / max(1, server.gpu_count);
            double leftover_cpu = static_cast<double>(server.cpu_cores - job.cpu_cores) / max(1, server.cpu_cores);
            double leftover_memory = static_cast<double>(server.memory - job.memory) / max(1, server.memory);
            double fragmentation_cost = 1000.0 * (0.50 * leftover_gpu + 0.25 * leftover_cpu + 0.25 * leftover_memory);
            double scarcity_cost = 1000.0 * server.power_score *
                static_cast<double>(max<int>(0, static_cast<int>(entries.size()) - 2)) / max<int>(1, servers.size());
            double tail_cost = static_cast<double>(max(0LL, finish - current_max_finish));
            double tetris_score = 0.0;
            if (params.tetris_weight > 0.0 || params.tetris_tie_epsilon > 0.0) {
                tetris_score = tetrisScore(job, timelines[entry.server_index], server, entry.gpu_used, start, finish);
            }

            double cost =
                params.wait_weight * strategy.wait_weight * wait_cost +
                params.finish_weight * strategy.finish_weight * finish_cost +
                params.memory_weight * strategy.memory_weight * memory_cost +
                params.fragment_weight * strategy.fragmentation_weight * fragmentation_cost +
                params.scarcity_weight * strategy.scarcity_weight * scarcity_cost +
                params.tail_weight * strategy.tail_weight * tail_cost -
                1000.0 * params.tetris_weight * tetris_score;

            Candidate candidate{true, entry.server_index, entry.gpu_used, start, finish, cost, tetris_score};
            candidates.push_back(candidate);
            sort(candidates.begin(), candidates.end(), [&](const Candidate &lhs, const Candidate &rhs) {
                return candidateLess(lhs, rhs, job);
            });
            if (static_cast<int>(candidates.size()) > keep_count) {
                candidates.pop_back();
            }
        }

        return candidates;
    }

    Candidate bestCandidate(
        const Job &job,
        const vector<ServerTimeline> &timelines,
        const StrategyConfig &strategy,
        long long current_max_finish
    ) const {
        vector<Candidate> candidates = topCandidates(job, timelines, strategy, current_max_finish, 1, 0);
        if (candidates.empty()) {
            return Candidate{};
        }
        return candidates.front();
    }

    Solution buildGreedySolution(const StrategyConfig &strategy, const vector<char> &skip) const {
        vector<ServerTimeline> timelines;
        timelines.reserve(servers.size());
        for (const ServerSpec &server : servers) {
            timelines.emplace_back(&server);
        }

        vector<ScheduleRecord> records(jobs.size() + 1);
        long long current_max_finish = 0;

        vector<int> order = orderedJobs(strategy, skip);
        for (int job_id : order) {
            const Job &job = jobs[job_id - 1];
            Candidate candidate = bestCandidate(job, timelines, strategy, current_max_finish);
            if (!candidate.valid) {
                return Solution{};
            }

            const ServerSpec &server = servers[candidate.server_index];
            timelines[candidate.server_index].addPlacement(Placement{
                job.job_id,
                candidate.start,
                candidate.finish,
                candidate.gpu_used,
                job.cpu_cores,
                job.memory,
            });

            records[job.job_id] = ScheduleRecord{
                job.job_id,
                server.server_id,
                candidate.start,
                candidate.gpu_used,
                candidate.finish,
            };
            current_max_finish = max(current_max_finish, candidate.finish);
        }

        records.erase(records.begin());
        Solution solution;
        solution.records = move(records);
        solution.valid = true;
        solution.score = proxyScore(solution.records);
        return solution;
    }

    struct DestroyOperatorConfig {
        string name;
        bool window = false;
        int mode = 0;
    };

    struct RepairOperatorConfig {
        string name;
        bool allow_regret = false;
        int regret_k = 0;
        double regret_weight = 0.0;
        int regret_candidate_limit = 0;
        int regret_max_block = 0;
    };

    struct AdaptiveState {
        vector<double> weights;
        vector<double> scores;
        vector<int> uses;

        AdaptiveState() = default;
        AdaptiveState(int count, double initial_weight)
            : weights(count, initial_weight),
              scores(count, 0.0),
              uses(count, 0) {}
    };

    vector<DestroyOperatorConfig> destroyOperators() const {
        return {
            {"wait", false, 0},
            {"memory", false, 1},
            {"tail_blocker", false, 2},
            {"scarce", false, 3},
            {"ideal_blocker", false, 4},
            {"random", false, 5},
            {"window_tail", true, 0},
            {"window_wait", true, 1},
            {"window_random", true, 2},
        };
    }

    vector<RepairOperatorConfig> repairOperators() const {
        return {
            {"fixed", false, 0, 0.0, 0, 0},
            {"regret2_cap48", true, 2, 2.0, 16, 48},
            {"regret3_cap32", true, 3, 2.0, 32, 32},
            {"regret3_cap48", true, 3, 2.0, 32, 48},
        };
    }

    RepairOperatorConfig repairOperatorFromParams() const {
        return {
            "params",
            params.regret_repair_enabled > 0 && params.regret_repair_k >= 2,
            max(0, params.regret_repair_k),
            params.regret_repair_weight,
            params.regret_repair_candidate_limit,
            params.regret_repair_max_block,
        };
    }

    int chooseAdaptiveIndex(const AdaptiveState &state, mt19937 &rng) const {
        double total = 0.0;
        for (double weight : state.weights) {
            total += max(0.0, weight);
        }
        if (total <= 0.0) {
            uniform_int_distribution<int> pick(0, static_cast<int>(state.weights.size()) - 1);
            return pick(rng);
        }
        uniform_real_distribution<double> pick(0.0, total);
        double target = pick(rng);
        double cumulative = 0.0;
        for (int idx = 0; idx < static_cast<int>(state.weights.size()); ++idx) {
            cumulative += max(0.0, state.weights[idx]);
            if (target <= cumulative) {
                return idx;
            }
        }
        return static_cast<int>(state.weights.size()) - 1;
    }

    void recordAdaptiveReward(AdaptiveState &state, int index, double reward) const {
        if (index < 0 || index >= static_cast<int>(state.weights.size())) {
            return;
        }
        state.uses[index] += 1;
        state.scores[index] += reward;
    }

    void updateAdaptiveWeights(AdaptiveState &state) const {
        double rho = params.adaptive_learning_rate;
        for (int idx = 0; idx < static_cast<int>(state.weights.size()); ++idx) {
            if (state.uses[idx] > 0) {
                double average = state.scores[idx] / max(1, state.uses[idx]);
                state.weights[idx] = (1.0 - rho) * state.weights[idx] + rho * average;
                state.weights[idx] = min(params.adaptive_max_weight, max(params.adaptive_min_weight, state.weights[idx]));
            }
            state.uses[idx] = 0;
            state.scores[idx] = 0.0;
        }
    }

    Solution rebuildWithRemoved(
        const Solution &base,
        const vector<int> &removed_jobs,
        const StrategyConfig &repair_strategy,
        const RepairOperatorConfig &repair_operator
    ) const {
        vector<char> removed(jobs.size() + 1, false);
        for (int job_id : removed_jobs) {
            removed[job_id] = true;
        }

        vector<ServerTimeline> timelines;
        timelines.reserve(servers.size());
        for (const ServerSpec &server : servers) {
            timelines.emplace_back(&server);
        }

        vector<int> server_index_by_id(servers.size() + 1, -1);
        int max_server_id = 0;
        for (const ServerSpec &server : servers) {
            max_server_id = max(max_server_id, server.server_id);
        }
        server_index_by_id.assign(max_server_id + 1, -1);
        for (int idx = 0; idx < static_cast<int>(servers.size()); ++idx) {
            server_index_by_id[servers[idx].server_id] = idx;
        }

        vector<ScheduleRecord> records(jobs.size() + 1);
        long long current_max_finish = 0;

        vector<int> kept_order;
        kept_order.reserve(jobs.size());
        for (const ScheduleRecord &record : base.records) {
            if (!removed[record.job_id]) {
                kept_order.push_back(record.job_id);
            }
        }
        sort(kept_order.begin(), kept_order.end(), [&](int lhs, int rhs) {
            const ScheduleRecord &a = base.records[lhs - 1];
            const ScheduleRecord &b = base.records[rhs - 1];
            if (a.start_time != b.start_time) return a.start_time < b.start_time;
            return a.job_id < b.job_id;
        });

        for (int job_id : kept_order) {
            const Job &job = jobs[job_id - 1];
            const ScheduleRecord &record = base.records[job_id - 1];
            int server_index = server_index_by_id[record.server_id];
            timelines[server_index].addPlacement(Placement{
                job.job_id,
                record.start_time,
                record.finish_time,
                record.gpu_used,
                job.cpu_cores,
                job.memory,
            });
            records[job.job_id] = record;
            current_max_finish = max(current_max_finish, record.finish_time);
        }

        vector<int> repair_order = removed_jobs;
        sort(repair_order.begin(), repair_order.end(), [&](int lhs_id, int rhs_id) {
            const Job &a = jobs[lhs_id - 1];
            const Job &b = jobs[rhs_id - 1];
            size_t sa = feasible_by_job[a.job_id].size();
            size_t sb = feasible_by_job[b.job_id].size();
            double ka =
                4.0 / max<size_t>(1, sa) +
                2.0 * static_cast<double>(a.weight) / max(1LL, a.duration) +
                1.5 * a.demand_score +
                params.duration_bias * a.duration;
            double kb =
                4.0 / max<size_t>(1, sb) +
                2.0 * static_cast<double>(b.weight) / max(1LL, b.duration) +
                1.5 * b.demand_score +
                params.duration_bias * b.duration;
            if (fabs(ka - kb) > 1e-12) return ka > kb;
            if (a.release_time != b.release_time) return a.release_time < b.release_time;
            return a.job_id < b.job_id;
        });

        auto place_job = [&](int job_id, const Candidate &candidate) -> bool {
            if (!candidate.valid) {
                return false;
            }
            const Job &job = jobs[job_id - 1];
            const ServerSpec &server = servers[candidate.server_index];
            timelines[candidate.server_index].addPlacement(Placement{
                job.job_id,
                candidate.start,
                candidate.finish,
                candidate.gpu_used,
                job.cpu_cores,
                job.memory,
            });
            records[job.job_id] = ScheduleRecord{
                job.job_id,
                server.server_id,
                candidate.start,
                candidate.gpu_used,
                candidate.finish,
            };
            current_max_finish = max(current_max_finish, candidate.finish);
            return true;
        };

        if (repair_operator.allow_regret && repair_operator.regret_k >= 2 &&
            static_cast<int>(repair_order.size()) > 1) {
            vector<int> rank_by_job(jobs.size() + 1, 0);
            for (int idx = 0; idx < static_cast<int>(repair_order.size()); ++idx) {
                rank_by_job[repair_order[idx]] = idx;
            }

            vector<int> remaining = repair_order;
            int regret_k = max(2, repair_operator.regret_k);
            while (!remaining.empty()) {
                int selected_index = -1;
                int selected_job = -1;
                Candidate selected_candidate;
                double selected_priority = -numeric_limits<double>::infinity();
                double selected_regret = -numeric_limits<double>::infinity();
                double selected_best_cost = numeric_limits<double>::infinity();

                for (int idx = 0; idx < static_cast<int>(remaining.size()); ++idx) {
                    int job_id = remaining[idx];
                    const Job &job = jobs[job_id - 1];
                    vector<Candidate> options = topCandidates(
                        job,
                        timelines,
                        repair_strategy,
                        current_max_finish,
                        regret_k,
                        repair_operator.regret_candidate_limit
                    );
                    if (options.empty()) {
                        return Solution{};
                    }

                    double best_cost = options.front().cost;
                    double missing_penalty = max(1000.0, fabs(best_cost) * 0.25 + 0.1 * static_cast<double>(job.duration));
                    double regret = 0.0;
                    for (int option_idx = 1; option_idx < regret_k; ++option_idx) {
                        double other_cost = option_idx < static_cast<int>(options.size())
                            ? options[option_idx].cost
                            : best_cost + missing_penalty;
                        regret += max(0.0, other_cost - best_cost);
                    }
                    double priority = repair_operator.regret_weight * regret - best_cost;

                    bool choose = false;
                    if (selected_index < 0 || priority > selected_priority + 1e-9) {
                        choose = true;
                    } else if (fabs(priority - selected_priority) <= 1e-9) {
                        if (regret > selected_regret + 1e-9) {
                            choose = true;
                        } else if (fabs(regret - selected_regret) <= 1e-9) {
                            if (best_cost < selected_best_cost - 1e-9) {
                                choose = true;
                            } else if (fabs(best_cost - selected_best_cost) <= 1e-9) {
                                if (options.front().start != selected_candidate.start) {
                                    choose = options.front().start < selected_candidate.start;
                                } else if (options.front().finish != selected_candidate.finish) {
                                    choose = options.front().finish < selected_candidate.finish;
                                } else if (servers[options.front().server_index].server_id !=
                                           servers[selected_candidate.server_index].server_id) {
                                    choose = servers[options.front().server_index].server_id <
                                             servers[selected_candidate.server_index].server_id;
                                } else if (rank_by_job[job_id] < rank_by_job[selected_job]) {
                                    choose = true;
                                }
                            }
                        }
                    }

                    if (choose) {
                        selected_index = idx;
                        selected_job = job_id;
                        selected_candidate = options.front();
                        selected_priority = priority;
                        selected_regret = regret;
                        selected_best_cost = best_cost;
                    }
                }

                if (selected_index < 0 || !place_job(selected_job, selected_candidate)) {
                    return Solution{};
                }
                remaining.erase(remaining.begin() + selected_index);
            }
        } else {
            for (int job_id : repair_order) {
                const Job &job = jobs[job_id - 1];
                Candidate candidate = bestCandidate(job, timelines, repair_strategy, current_max_finish);
                if (!place_job(job.job_id, candidate)) {
                    return Solution{};
                }
            }
        }

        records.erase(records.begin());
        Solution solution;
        solution.records = move(records);
        solution.valid = true;
        solution.score = proxyScore(solution.records);
        return solution;
    }

    string repairBlockSignature(vector<int> job_ids) const {
        sort(job_ids.begin(), job_ids.end());
        string signature;
        for (int job_id : job_ids) {
            signature += "#";
            signature += to_string(job_id);
        }
        return signature;
    }

    bool isBlockTabu(const string &signature, const vector<pair<string, int>> &block_tabu, int iter) const {
        if (signature.empty() || params.tabu_block_tenure <= 0) {
            return false;
        }
        for (const auto &entry : block_tabu) {
            if (entry.first == signature && entry.second > iter) {
                return true;
            }
        }
        return false;
    }

    void markBlockTabu(vector<pair<string, int>> &block_tabu, const string &signature, int iter) const {
        if (signature.empty() || params.tabu_block_tenure <= 0) {
            return;
        }
        int until = iter + params.tabu_block_tenure;
        for (auto &entry : block_tabu) {
            if (entry.first == signature) {
                entry.second = until;
                return;
            }
        }
        block_tabu.push_back({signature, until});
    }

    void markJobsTabu(vector<int> &job_tabu_until, const vector<int> &job_ids, int iter) const {
        if (params.tabu_tenure <= 0) {
            return;
        }
        int until = iter + params.tabu_tenure;
        for (int job_id : job_ids) {
            if (job_id >= 1 && job_id < static_cast<int>(job_tabu_until.size())) {
                job_tabu_until[job_id] = max(job_tabu_until[job_id], until);
            }
        }
    }

    vector<int> filterTabuJobs(const vector<int> &job_ids, const vector<int> &job_tabu_until, int iter) const {
        if (params.tabu_tenure <= 0 || job_ids.empty()) {
            return job_ids;
        }

        vector<int> filtered;
        filtered.reserve(job_ids.size());
        for (int job_id : job_ids) {
            if (job_id >= 1 && job_id < static_cast<int>(job_tabu_until.size()) &&
                job_tabu_until[job_id] > iter) {
                continue;
            }
            filtered.push_back(job_id);
        }

        int minimum_size = max(1, static_cast<int>(job_ids.size()) / 2);
        if (static_cast<int>(filtered.size()) < minimum_size) {
            return job_ids;
        }
        return filtered;
    }

    vector<int> selectRemovedJobs(const Solution &solution, int mode, mt19937 &rng) const {
        int n = static_cast<int>(jobs.size());
        int q = max(4, n / 50);
        q = min(q, 120);
        q = min(q, max(1, n / 4));
        int cap = min(n, max(q, min(180, q * 2)));

        vector<int> ids(n);
        iota(ids.begin(), ids.end(), 1);
        vector<char> picked(n + 1, false);
        vector<int> result;
        result.reserve(cap);

        auto add_unique = [&](int job_id) {
            if (job_id >= 1 && job_id <= n && !picked[job_id] && static_cast<int>(result.size()) < cap) {
                picked[job_id] = true;
                result.push_back(job_id);
            }
        };

        auto finish_result = [&]() {
            if (result.empty()) {
                shuffle(ids.begin(), ids.end(), rng);
                for (int job_id : ids) {
                    add_unique(job_id);
                    if (static_cast<int>(result.size()) >= q) break;
                }
            }
            if (static_cast<int>(result.size()) > cap) {
                result.resize(cap);
            }
            return result;
        };

        if (mode == 0) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                const Job &a = jobs[lhs - 1];
                const Job &b = jobs[rhs - 1];
                const ScheduleRecord &ra = solution.records[lhs - 1];
                const ScheduleRecord &rb = solution.records[rhs - 1];
                long long ca = (ra.start_time - a.release_time) * max(1, a.weight);
                long long cb = (rb.start_time - b.release_time) * max(1, b.weight);
                if (ca != cb) return ca > cb;
                return lhs < rhs;
            });
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        } else if (mode == 1) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                const Job &a = jobs[lhs - 1];
                const Job &b = jobs[rhs - 1];
                const ScheduleRecord &ra = solution.records[lhs - 1];
                const ScheduleRecord &rb = solution.records[rhs - 1];
                const ServerSpec &sa = serverById(ra.server_id);
                const ServerSpec &sb = serverById(rb.server_id);
                long long wa = 1LL * (ra.gpu_used * sa.gpu_memory - a.gpu_memory) * a.duration;
                long long wb = 1LL * (rb.gpu_used * sb.gpu_memory - b.gpu_memory) * b.duration;
                if (wa != wb) return wa > wb;
                return lhs < rhs;
            });
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        } else if (mode == 2) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                const ScheduleRecord &a = solution.records[lhs - 1];
                const ScheduleRecord &b = solution.records[rhs - 1];
                if (a.finish_time != b.finish_time) return a.finish_time > b.finish_time;
                return lhs < rhs;
            });
            int seed_count = max(2, q / 3);
            seed_count = min(seed_count, static_cast<int>(ids.size()));
            for (int idx = 0; idx < seed_count; ++idx) {
                int tail_id = ids[idx];
                add_unique(tail_id);
                const Job &tail_job = jobs[tail_id - 1];
                const ScheduleRecord &tail_record = solution.records[tail_id - 1];
                vector<pair<long long, int>> blockers;
                for (const ScheduleRecord &other : solution.records) {
                    if (other.job_id == tail_id || other.server_id != tail_record.server_id) {
                        continue;
                    }
                    if (other.start_time < tail_record.start_time && other.finish_time > tail_job.release_time) {
                        long long closeness = other.finish_time;
                        blockers.push_back({-closeness, other.job_id});
                    }
                }
                sort(blockers.begin(), blockers.end());
                for (const auto &item : blockers) {
                    add_unique(item.second);
                    if (static_cast<int>(result.size()) >= cap) break;
                }
                if (static_cast<int>(result.size()) >= cap) break;
            }
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        } else if (mode == 3) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                size_t sl = feasible_by_job[lhs].size();
                size_t sr = feasible_by_job[rhs].size();
                if (sl != sr) return sl < sr;
                const Job &a = jobs[lhs - 1];
                const Job &b = jobs[rhs - 1];
                const ScheduleRecord &ra = solution.records[lhs - 1];
                const ScheduleRecord &rb = solution.records[rhs - 1];
                long long wa = (ra.start_time - a.release_time) * max(1, a.weight);
                long long wb = (rb.start_time - b.release_time) * max(1, b.weight);
                if (wa != wb) return wa > wb;
                return lhs < rhs;
            });
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        } else if (mode == 4) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                const Job &a = jobs[lhs - 1];
                const Job &b = jobs[rhs - 1];
                const ScheduleRecord &ra = solution.records[lhs - 1];
                const ScheduleRecord &rb = solution.records[rhs - 1];
                long long wa = (ra.start_time - a.release_time) * max(1, a.weight);
                long long wb = (rb.start_time - b.release_time) * max(1, b.weight);
                if (wa != wb) return wa > wb;
                return lhs < rhs;
            });

            int seed_count = max(1, q / 12);
            seed_count = min(seed_count, static_cast<int>(ids.size()));
            for (int idx = 0; idx < seed_count; ++idx) {
                int seed_id = ids[idx];
                const Job &job = jobs[seed_id - 1];
                const ScheduleRecord &record = solution.records[seed_id - 1];
                add_unique(seed_id);
                if (feasible_by_job[seed_id].empty() || record.start_time <= job.release_time) {
                    continue;
                }

                int ideal_server_id = servers[feasible_by_job[seed_id].front().server_index].server_id;
                vector<pair<long long, int>> blockers;
                for (const ScheduleRecord &other : solution.records) {
                    if (other.job_id == seed_id || other.server_id != ideal_server_id) {
                        continue;
                    }
                    if (other.start_time < record.start_time && other.finish_time > job.release_time) {
                        long long overlap =
                            min(record.start_time, other.finish_time) - max(job.release_time, other.start_time);
                        if (overlap > 0) {
                            blockers.push_back({-overlap, other.job_id});
                        }
                    }
                }
                sort(blockers.begin(), blockers.end());
                for (const auto &item : blockers) {
                    add_unique(item.second);
                    if (static_cast<int>(result.size()) >= cap) break;
                }
                if (static_cast<int>(result.size()) >= cap) break;
            }
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        } else {
            shuffle(ids.begin(), ids.end(), rng);
            for (int job_id : ids) {
                add_unique(job_id);
                if (static_cast<int>(result.size()) >= q) break;
            }
        }

        return finish_result();
    }

    vector<int> selectSmallWindowJobs(
        const Solution &solution,
        int center_mode,
        mt19937 &rng,
        int fallback_mode = -1
    ) const {
        int n = static_cast<int>(jobs.size());
        if (n <= 1) {
            return {};
        }

        int q = max(4, n / 80);
        q = min(q, max(1, n / 5));
        int cap = min(n, max(q, min(params.small_window_repair_cap, q * 3)));

        int center_id = 1;
        if (center_mode == 0) {
            center_id = max_element(solution.records.begin(), solution.records.end(),
                                    [](const ScheduleRecord &lhs, const ScheduleRecord &rhs) {
                                        if (lhs.finish_time != rhs.finish_time) return lhs.finish_time < rhs.finish_time;
                                        return lhs.job_id > rhs.job_id;
                                    })->job_id;
        } else if (center_mode == 1) {
            center_id = max_element(solution.records.begin(), solution.records.end(),
                                    [&](const ScheduleRecord &lhs, const ScheduleRecord &rhs) {
                                        const Job &lj = jobs[lhs.job_id - 1];
                                        const Job &rj = jobs[rhs.job_id - 1];
                                        long long lw = 1LL * (lhs.start_time - lj.release_time) * max(1, lj.weight);
                                        long long rw = 1LL * (rhs.start_time - rj.release_time) * max(1, rj.weight);
                                        if (lw != rw) return lw < rw;
                                        return lhs.job_id > rhs.job_id;
                                    })->job_id;
        } else {
            uniform_int_distribution<int> pick(1, n);
            center_id = pick(rng);
        }

        const ScheduleRecord &center = solution.records[center_id - 1];
        const Job &center_job = jobs[center_id - 1];
        int width = params.small_window_repair_width;
        if (width == 0) {
            width = static_cast<int>(max(1LL, center_job.duration));
        }
        long long left = max(0LL, center.start_time - width);
        long long right = center.finish_time + width;

        vector<pair<long long, int>> candidates;
        candidates.reserve(n);
        for (const ScheduleRecord &record : solution.records) {
            if (record.server_id != center.server_id) {
                continue;
            }
            if (record.finish_time < left || record.start_time > right) {
                continue;
            }
            long long overlap = min(record.finish_time, right) - max(record.start_time, left);
            long long priority = -max(0LL, overlap);
            if (record.job_id == center_id) {
                priority -= 1LL << 30;
            }
            candidates.push_back({priority, record.job_id});
        }
        sort(candidates.begin(), candidates.end());

        vector<int> result;
        vector<char> picked(n + 1, false);
        result.reserve(cap);
        for (const auto &item : candidates) {
            int job_id = item.second;
            if (job_id >= 1 && job_id <= n && !picked[job_id]) {
                picked[job_id] = true;
                result.push_back(job_id);
                if (static_cast<int>(result.size()) >= cap) {
                    break;
                }
            }
        }

        if (static_cast<int>(result.size()) < q) {
            if (fallback_mode < 0) {
                fallback_mode = center_mode % 5;
            }
            vector<int> fallback = selectRemovedJobs(solution, fallback_mode, rng);
            for (int job_id : fallback) {
                if (job_id >= 1 && job_id <= n && !picked[job_id]) {
                    picked[job_id] = true;
                    result.push_back(job_id);
                    if (static_cast<int>(result.size()) >= q) {
                        break;
                    }
                }
            }
        }
        return result;
    }

    void improveLocally(
        Solution &best,
        chrono::steady_clock::time_point deadline,
        double local_time_fraction
    ) const {
        int n = static_cast<int>(jobs.size());
        if (n <= 1) {
            return;
        }

        auto start = chrono::steady_clock::now();
        double time_limit = params.local_search_time_small;
        int max_iterations = 80;
        if (n > 250) {
            time_limit = params.local_search_time_medium;
            max_iterations = 70;
        }
        if (n > 1000) {
            time_limit = params.local_search_time_large;
            max_iterations = 45;
        }
        if (n > 3000) {
            time_limit = params.local_search_time_huge;
            max_iterations = 30;
        }
        if (params.repair_rounds > 0) {
            max_iterations = params.repair_rounds;
        }
        time_limit *= max(0.05, local_time_fraction);

        StrategyConfig repair_strategy{"repair", 5, 1.90, 0.020, 0.020, 0.008, 0.004, 0.090};
        mt19937 rng(712367 + n);
        uniform_real_distribution<double> probability(0.0, 1.0);
        Solution best_solution = best;
        Solution current_solution = best;
        vector<double> history;
        int history_index = 0;
        if (params.late_acceptance_window > 0) {
            history.assign(params.late_acceptance_window, current_solution.score);
        }
        vector<int> job_tabu_until(jobs.size() + 1, 0);
        vector<pair<string, int>> block_tabu;

        int standard_iterations = max_iterations;
        int total_iterations = standard_iterations + params.small_window_repair_rounds;
        vector<DestroyOperatorConfig> destroy_ops = destroyOperators();
        vector<RepairOperatorConfig> repair_ops = repairOperators();
        AdaptiveState destroy_state(static_cast<int>(destroy_ops.size()), params.adaptive_initial_weight);
        AdaptiveState repair_state(static_cast<int>(repair_ops.size()), params.adaptive_initial_weight);

        for (int iter = 0; iter < total_iterations; ++iter) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            if (elapsed > time_limit || secondsLeft(deadline) < 0.05) {
                break;
            }

            int destroy_index = -1;
            bool window_iter = false;
            int mode = 0;
            vector<int> original_removed;
            auto choose_destroy_jobs = [&](int selected_index, bool &is_window, int &selected_mode) {
                const DestroyOperatorConfig &op = destroy_ops[selected_index];
                is_window = op.window;
                selected_mode = op.mode;
                if (op.window) {
                    return selectSmallWindowJobs(current_solution, op.mode, rng);
                }
                return selectRemovedJobs(current_solution, op.mode, rng);
            };

            if (params.adaptive_destroy_enabled > 0) {
                destroy_index = chooseAdaptiveIndex(destroy_state, rng);
                original_removed = choose_destroy_jobs(destroy_index, window_iter, mode);
            } else {
                window_iter = iter >= standard_iterations;
                mode = (probability(rng) < params.random_repair_ratio) ? 5 : (iter % 5);
                original_removed = window_iter
                    ? selectSmallWindowJobs(
                        current_solution,
                        (iter - standard_iterations) % 3,
                        rng,
                        (iter - standard_iterations) % 5
                    )
                    : selectRemovedJobs(current_solution, mode, rng);
            }

            vector<int> removed = filterTabuJobs(original_removed, job_tabu_until, iter);
            if (removed.empty()) {
                removed = original_removed;
            }
            string block_signature = repairBlockSignature(removed);

            if (isBlockTabu(block_signature, block_tabu, iter)) {
                bool replaced = false;
                for (int attempt = 1; attempt <= 4; ++attempt) {
                    int alternate_destroy_index = -1;
                    bool alternate_window = false;
                    int alternate_mode = 0;
                    vector<int> alternate_original;
                    if (params.adaptive_destroy_enabled > 0) {
                        alternate_destroy_index = chooseAdaptiveIndex(destroy_state, rng);
                        alternate_original = choose_destroy_jobs(alternate_destroy_index, alternate_window, alternate_mode);
                    } else {
                        alternate_mode = (mode + attempt) % 6;
                        alternate_original = selectRemovedJobs(current_solution, alternate_mode, rng);
                    }
                    vector<int> alternate_removed =
                        filterTabuJobs(alternate_original, job_tabu_until, iter);
                    string alternate_signature = repairBlockSignature(alternate_removed);
                    if (!isBlockTabu(alternate_signature, block_tabu, iter)) {
                        removed = move(alternate_removed);
                        block_signature = move(alternate_signature);
                        if (params.adaptive_destroy_enabled > 0) {
                            destroy_index = alternate_destroy_index;
                            window_iter = alternate_window;
                            mode = alternate_mode;
                        }
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    removed = move(original_removed);
                    block_signature = repairBlockSignature(removed);
                }
            }

            int repair_index = -1;
            RepairOperatorConfig repair_operator = repairOperatorFromParams();
            if (params.adaptive_repair_enabled > 0) {
                repair_index = chooseAdaptiveIndex(repair_state, rng);
                repair_operator = repair_ops[repair_index];
            }

            bool allow_regret = repair_operator.allow_regret;
            if (params.adaptive_repair_enabled <= 0) {
                if (allow_regret && params.regret_repair_scope == 1 && window_iter) {
                    allow_regret = false;
                }
                if (allow_regret && params.regret_repair_scope == 2 && !window_iter) {
                    allow_regret = false;
                }
                if (allow_regret && params.regret_repair_min_block > 0 &&
                    static_cast<int>(removed.size()) < params.regret_repair_min_block) {
                    allow_regret = false;
                }
            }
            if (allow_regret && repair_operator.regret_max_block > 0 &&
                static_cast<int>(removed.size()) > repair_operator.regret_max_block) {
                allow_regret = false;
            }
            repair_operator.allow_regret = allow_regret;

            Solution candidate = rebuildWithRemoved(current_solution, removed, repair_strategy, repair_operator);
            bool accepted = false;
            bool global_best = false;
            bool current_improve = false;
            if (candidate.valid) {
                global_best = candidate.score < best_solution.score - 1e-7;
                current_improve = candidate.score < current_solution.score - 1e-7;
                if (global_best || current_improve) {
                    accepted = true;
                } else if (!history.empty() &&
                           candidate.score <= history[history_index] + params.late_acceptance_margin) {
                    accepted = true;
                }
            }

            double reward = 0.0;
            if (global_best) {
                reward = params.adaptive_global_best_reward;
            } else if (accepted && current_improve) {
                reward = params.adaptive_current_improve_reward;
            }
            if (params.adaptive_destroy_enabled > 0) {
                recordAdaptiveReward(destroy_state, destroy_index, reward);
            }
            if (params.adaptive_repair_enabled > 0) {
                recordAdaptiveReward(repair_state, repair_index, reward);
            }

            if (accepted) {
                current_solution = move(candidate);
                if (current_solution.score < best_solution.score - 1e-7) {
                    best_solution = current_solution;
                }
            } else {
                markJobsTabu(job_tabu_until, removed, iter);
                markBlockTabu(block_tabu, block_signature, iter);
            }

            if (!history.empty()) {
                history[history_index] = current_solution.score;
                history_index = (history_index + 1) % static_cast<int>(history.size());
            }

            if ((iter + 1) % params.adaptive_segment_length == 0) {
                if (params.adaptive_destroy_enabled > 0) {
                    updateAdaptiveWeights(destroy_state);
                }
                if (params.adaptive_repair_enabled > 0) {
                    updateAdaptiveWeights(repair_state);
                }
            }
        }

        best = move(best_solution);
    }

    const ServerSpec &serverById(int server_id) const {
        for (const ServerSpec &server : servers) {
            if (server.server_id == server_id) {
                return server;
            }
        }
        return servers.front();
    }

    RawMetrics rawMetrics(const vector<ScheduleRecord> &records) const {
        RawMetrics metrics;
        for (const ScheduleRecord &record : records) {
            const Job &job = jobs[record.job_id - 1];
            const ServerSpec &server = serverById(record.server_id);
            metrics.weighted_wait +=
                static_cast<long double>(record.start_time - job.release_time) * max(1, job.weight);
            metrics.memory_waste +=
                static_cast<long double>(record.gpu_used * server.gpu_memory - job.gpu_memory) * job.duration;
            metrics.max_finish = max(metrics.max_finish, record.finish_time);
        }
        return metrics;
    }

    void initializeMetricScales(vector<Solution> &solutions) {
        long double best_wait = numeric_limits<long double>::infinity();
        long double best_memory = numeric_limits<long double>::infinity();
        long double best_finish = numeric_limits<long double>::infinity();

        for (const Solution &solution : solutions) {
            if (!solution.valid) {
                continue;
            }
            RawMetrics metrics = rawMetrics(solution.records);
            best_wait = min(best_wait, metrics.weighted_wait);
            best_memory = min(best_memory, metrics.memory_waste);
            best_finish = min(best_finish, static_cast<long double>(metrics.max_finish));
        }

        wait_scale = max(1.0L, best_wait);
        memory_scale = max(1.0L, best_memory);
        finish_scale = max(1.0L, best_finish);

        for (Solution &solution : solutions) {
            if (solution.valid) {
                solution.score = proxyScore(solution.records);
            }
        }
    }

    double proxyScore(const vector<ScheduleRecord> &records) const {
        RawMetrics metrics = rawMetrics(records);
        long double wait_norm = metrics.weighted_wait / max(1.0L, wait_scale);
        long double memory_norm = metrics.memory_waste / max(1.0L, memory_scale);
        long double finish_norm = static_cast<long double>(metrics.max_finish) / max(1.0L, finish_scale);

        return static_cast<double>(
            static_cast<long double>(params.proxy_wait_weight) * wait_norm +
            static_cast<long double>(params.proxy_memory_weight) * memory_norm +
            static_cast<long double>(params.proxy_finish_weight) * finish_norm
        );
    }

    vector<ServerSpec> servers;
    vector<Job> jobs;
    vector<vector<FeasibleEntry>> feasible_by_job;
    TuningParams base_params;
    TuningParams params;
    long long total_weight = 0;
    long long total_duration = 0;
    long long total_gpu_memory = 0;
    long long total_job_gpu_memory_time = 0;
    long long min_release = numeric_limits<long long>::max();
    long double wait_scale = 1.0L;
    long double memory_scale = 1.0L;
    long double finish_scale = 1.0L;
};

pair<vector<ServerSpec>, vector<Job>> readInstance(istream &input) {
    int server_count;
    int job_count;
    if (!(input >> server_count >> job_count)) {
        return {{}, {}};
    }

    vector<ServerSpec> servers;
    servers.reserve(server_count);
    for (int server_id = 1; server_id <= server_count; ++server_id) {
        ServerSpec server;
        server.server_id = server_id;
        input >> server.gpu_count >> server.gpu_memory >> server.cpu_cores >> server.memory;
        servers.push_back(server);
    }

    vector<Job> jobs;
    jobs.reserve(job_count);
    for (int job_id = 1; job_id <= job_count; ++job_id) {
        Job job;
        job.job_id = job_id;
        input >> job.release_time >> job.duration >> job.min_gpu >> job.gpu_memory
              >> job.cpu_cores >> job.memory >> job.weight;
        jobs.push_back(job);
    }

    return {servers, jobs};
}

void writeRecords(ostream &output, vector<ScheduleRecord> records) {
    sort(records.begin(), records.end(), [](const ScheduleRecord &a, const ScheduleRecord &b) {
        return a.job_id < b.job_id;
    });

    for (const ScheduleRecord &record : records) {
        output << record.job_id << ' '
               << record.server_id << ' '
               << record.start_time << ' '
               << record.gpu_used << ' '
               << record.finish_time << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    pair<vector<ServerSpec>, vector<Job>> input = readInstance(cin);
    vector<ServerSpec> servers = move(input.first);
    vector<Job> jobs = move(input.second);
    if (jobs.empty()) {
        return 0;
    }

    TuningParams params = loadTuningParams();
    Scheduler scheduler(move(servers), move(jobs), params);
    Solution solution = scheduler.solve();
    if (!solution.valid) {
        return 0;
    }

    writeRecords(cout, move(solution.records));
    return 0;
}
