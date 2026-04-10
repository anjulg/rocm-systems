#pragma once

#include "mirage/simulator.h"

#include <stdexcept>
#include <unordered_map>

namespace mirage::testing {

/// A configurable mock Simulator for testing.
///
/// Stores sessions in-memory and returns predictable values.
class MockSimulator : public Simulator {
public:
    explicit MockSimulator(SimulatorInfo info) : info_(std::move(info)) {}

    SimulatorInfo info() const override { return info_; }

    std::vector<GpuDef> supported_gpus() const override {
        return info_.supported_gpus;
    }

    GpuDef set_custom_gpu(const CustomGpuDef& gpu) override {
        GpuDef def{gpu.name, "custom_arch", GpuFamily::AmdCdna, "custom"};
        info_.supported_gpus.push_back(def);
        return def;
    }

    ContainerDef create_session(const SessionDef& session,
                                const ProfileDef& profile) override {
        if (should_fail_create_) {
            throw std::runtime_error("mock create failure");
        }
        ContainerDef container;
        container.image =
            session.image.empty() ? "default:latest" : session.image;
        container.env.push_back({"SIMULATOR", info_.name});
        container.env.push_back({"GPU", profile.gpu});
        sessions_[session.name] = profile;
        return container;
    }

    void delete_session(const std::string& session_id) override {
        sessions_.erase(session_id);
    }

    SessionHealth get_session_health(
        const std::string& session_id) const override {
        SessionHealth h;
        h.session_id = session_id;
        h.status = sessions_.contains(session_id) ? HealthStatus::Healthy
                                                   : HealthStatus::Unhealthy;
        h.uptime = {42, 0};
        return h;
    }

    SessionPerf get_session_perf(
        const std::string& session_id) const override {
        SessionPerf p;
        p.session_id = session_id;
        p.ticks = 1000;
        p.ipc = 2.5;
        p.simulation_speed = 0.8;
        p.active_contexts = 64;
        return p;
    }

    ExecDef get_run_def(const RunDef& run) const override {
        ExecDef exec = run.exec;
        exec.env.push_back({"LD_PRELOAD", "/usr/lib/libsim.so"});
        return exec;
    }

    // Test controls
    void set_fail_create(bool fail) { should_fail_create_ = fail; }
    size_t session_count() const { return sessions_.size(); }

private:
    SimulatorInfo info_;
    bool should_fail_create_ = false;
    std::unordered_map<std::string, ProfileDef> sessions_;
};

/// Create a standard mock SimulatorInfo for testing.
inline SimulatorInfo make_test_simulator_info(
    const std::string& name = "test-sim") {
    SimulatorInfo info;
    info.name = name;
    info.version = "1.0.0";
    info.description = "Test simulator";
    info.supported_gpus = {
        {"MI300X", "gfx942", GpuFamily::AmdCdna, "AMD MI300X"},
        {"MI325X", "gfx950", GpuFamily::AmdCdna, "AMD MI325X"},
    };
    info.supports_custom_gpus = true;
    info.supported_modes = {SimulatorMode::Functional,
                            SimulatorMode::CycleAccurate};
    return info;
}

/// Create a standard test ProfileDef.
inline ProfileDef make_test_profile(
    const std::string& name = "test-profile",
    const std::string& simulator = "test-sim",
    const std::string& gpu = "MI300X") {
    ProfileDef p;
    p.name = name;
    p.simulator = simulator;
    p.gpu = gpu;
    p.mode = SimulatorMode::Functional;
    p.num_gpus = 4;
    p.num_nodes = 2;
    return p;
}

/// Create a standard test SessionDef.
inline SessionDef make_test_session(
    const std::string& name = "test-session",
    const std::string& profile = "test-profile") {
    SessionDef s;
    s.name = name;
    s.profile = profile;
    s.image = "ghcr.io/rocm/pytorch:latest";
    return s;
}

/// Create a MockSimulator with default test info.
inline std::shared_ptr<MockSimulator> make_mock_simulator(
    const std::string& name = "test-sim") {
    return std::make_shared<MockSimulator>(make_test_simulator_info(name));
}

} // namespace mirage::testing
