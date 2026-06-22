#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
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

    Candidate() = default;
    Candidate(bool valid, int server_index, int gpu_used, long long start, long long finish, double cost)
        : valid(valid),
          server_index(server_index),
          gpu_used(gpu_used),
          start(start),
          finish(finish),
          cost(cost) {}
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
    Scheduler(vector<ServerSpec> input_servers, vector<Job> input_jobs)
        : servers(move(input_servers)), jobs(move(input_jobs)) {
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
        vector<StrategyConfig> strategies = buildStrategies();
        Solution best;
        for (const StrategyConfig &strategy : strategies) {
            Solution candidate = buildGreedySolution(strategy, {});
            if (candidate.valid && candidate.score < best.score) {
                best = move(candidate);
            }
        }

        if (!best.valid) {
            return best;
        }

        improveLocally(best);
        return best;
    }

private:
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
                if (a.gpu_used != b.gpu_used) {
                    return a.gpu_used < b.gpu_used;
                }
                if (a.memory_waste != b.memory_waste) {
                    return a.memory_waste < b.memory_waste;
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

    Candidate bestCandidate(
        const Job &job,
        const vector<ServerTimeline> &timelines,
        const StrategyConfig &strategy,
        long long current_max_finish
    ) const {
        Candidate best;
        const vector<FeasibleEntry> &entries = feasible_by_job[job.job_id];

        for (const FeasibleEntry &entry : entries) {
            const ServerSpec &server = servers[entry.server_index];
            long long start = timelines[entry.server_index].earliestStart(job, entry.gpu_used);
            long long finish = start + job.duration;

            double wait_cost = static_cast<double>(start - job.release_time) * max(1, job.weight);
            double finish_cost = static_cast<double>(finish);
            double memory_cost = static_cast<double>(entry.memory_waste) * sqrt(static_cast<double>(max(1LL, job.duration)));
            double leftover_gpu = static_cast<double>(server.gpu_count - entry.gpu_used) / max(1, server.gpu_count);
            double leftover_cpu = static_cast<double>(server.cpu_cores - job.cpu_cores) / max(1, server.cpu_cores);
            double leftover_memory = static_cast<double>(server.memory - job.memory) / max(1, server.memory);
            double fragmentation_cost = 1000.0 * (0.50 * leftover_gpu + 0.25 * leftover_cpu + 0.25 * leftover_memory);
            double scarcity_cost = 1000.0 * server.power_score *
                static_cast<double>(max<int>(0, static_cast<int>(entries.size()) - 2)) / max<int>(1, servers.size());
            double tail_cost = static_cast<double>(max(0LL, finish - current_max_finish));

            double cost =
                strategy.wait_weight * wait_cost +
                strategy.finish_weight * finish_cost +
                strategy.memory_weight * memory_cost +
                strategy.fragmentation_weight * fragmentation_cost +
                strategy.scarcity_weight * scarcity_cost +
                strategy.tail_weight * tail_cost;

            bool better = false;
            if (!best.valid || cost < best.cost - 1e-9) {
                better = true;
            } else if (fabs(cost - best.cost) <= 1e-9) {
                if (start != best.start) {
                    better = start < best.start;
                } else if (finish != best.finish) {
                    better = finish < best.finish;
                } else if (entry.memory_waste != feasible_by_job[job.job_id][0].memory_waste) {
                    better = entry.memory_waste < feasible_by_job[job.job_id][0].memory_waste;
                } else {
                    better = server.server_id < servers[best.server_index].server_id;
                }
            }

            if (better) {
                best = Candidate{true, entry.server_index, entry.gpu_used, start, finish, cost};
            }
        }

        return best;
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

    Solution rebuildWithRemoved(
        const Solution &base,
        const vector<int> &removed_jobs,
        const StrategyConfig &repair_strategy
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
                0.0001 * a.duration;
            double kb =
                4.0 / max<size_t>(1, sb) +
                2.0 * static_cast<double>(b.weight) / max(1LL, b.duration) +
                1.5 * b.demand_score +
                0.0001 * b.duration;
            if (fabs(ka - kb) > 1e-12) return ka > kb;
            if (a.release_time != b.release_time) return a.release_time < b.release_time;
            return a.job_id < b.job_id;
        });

        for (int job_id : repair_order) {
            const Job &job = jobs[job_id - 1];
            Candidate candidate = bestCandidate(job, timelines, repair_strategy, current_max_finish);
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

    vector<int> selectRemovedJobs(const Solution &solution, int mode, mt19937 &rng) const {
        int n = static_cast<int>(jobs.size());
        int q = max(4, n / 50);
        q = min(q, 120);
        q = min(q, max(1, n / 4));

        vector<int> ids(n);
        iota(ids.begin(), ids.end(), 1);

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
        } else if (mode == 2) {
            sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
                const ScheduleRecord &a = solution.records[lhs - 1];
                const ScheduleRecord &b = solution.records[rhs - 1];
                if (a.finish_time != b.finish_time) return a.finish_time > b.finish_time;
                return lhs < rhs;
            });
        } else {
            shuffle(ids.begin(), ids.end(), rng);
        }

        ids.resize(q);
        return ids;
    }

    void improveLocally(Solution &best) const {
        int n = static_cast<int>(jobs.size());
        if (n <= 1) {
            return;
        }

        auto start = chrono::steady_clock::now();
        double time_limit = 1.2;
        int max_iterations = 80;
        if (n > 250) {
            time_limit = 2.5;
            max_iterations = 70;
        }
        if (n > 1000) {
            time_limit = 5.0;
            max_iterations = 45;
        }
        if (n > 3000) {
            time_limit = 8.0;
            max_iterations = 30;
        }

        StrategyConfig repair_strategy{"repair", 5, 1.90, 0.020, 0.020, 0.008, 0.004, 0.090};
        mt19937 rng(712367 + n);

        for (int iter = 0; iter < max_iterations; ++iter) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            if (elapsed > time_limit) {
                break;
            }

            vector<int> removed = selectRemovedJobs(best, iter % 4, rng);
            Solution candidate = rebuildWithRemoved(best, removed, repair_strategy);
            if (candidate.valid && candidate.score < best.score - 1e-7) {
                best = move(candidate);
            }
        }
    }

    const ServerSpec &serverById(int server_id) const {
        for (const ServerSpec &server : servers) {
            if (server.server_id == server_id) {
                return server;
            }
        }
        return servers.front();
    }

    double proxyScore(const vector<ScheduleRecord> &records) const {
        long double weighted_wait = 0.0L;
        long long max_finish = 0;
        long double waste_time = 0.0L;

        for (const ScheduleRecord &record : records) {
            const Job &job = jobs[record.job_id - 1];
            const ServerSpec &server = serverById(record.server_id);
            weighted_wait += static_cast<long double>(record.start_time - job.release_time) * max(1, job.weight);
            max_finish = max(max_finish, record.finish_time);
            waste_time += static_cast<long double>(record.gpu_used * server.gpu_memory - job.gpu_memory) * job.duration;
        }

        long double avg_wait = weighted_wait / max(1LL, total_weight);
        long double horizon = max(1LL, max_finish);
        long double avg_idle_memory = static_cast<long double>(total_gpu_memory) -
            static_cast<long double>(total_job_gpu_memory_time) / horizon;
        long double avg_waste = waste_time / max(1LL, total_duration);

        return static_cast<double>(
            1.00L * avg_wait +
            0.18L * static_cast<long double>(max_finish) +
            0.60L * avg_idle_memory +
            0.025L * avg_waste
        );
    }

    vector<ServerSpec> servers;
    vector<Job> jobs;
    vector<vector<FeasibleEntry>> feasible_by_job;
    long long total_weight = 0;
    long long total_duration = 0;
    long long total_gpu_memory = 0;
    long long total_job_gpu_memory_time = 0;
    long long min_release = numeric_limits<long long>::max();
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

    Scheduler scheduler(move(servers), move(jobs));
    Solution solution = scheduler.solve();
    if (!solution.valid) {
        return 0;
    }

    writeRecords(cout, move(solution.records));
    return 0;
}
