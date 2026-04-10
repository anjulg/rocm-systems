#include "mirage/simulator.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

namespace mirage {
namespace {

using mirage::testing::MockSimulator;
using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;
using mirage::testing::make_test_simulator_info;

// ── SimulatorInfo ──────────────────────────────────────────────────────────

TEST(SimulatorInfoTest, Construction) {
    auto info = make_test_simulator_info();
    EXPECT_EQ(info.name, "test-sim");
    EXPECT_EQ(info.version, "1.0.0");
    EXPECT_EQ(info.supported_gpus.size(), 2u);
    EXPECT_TRUE(info.supports_custom_gpus);
    EXPECT_EQ(info.supported_modes.size(), 2u);
}

TEST(SimulatorInfoTest, CustomName) {
    auto info = make_test_simulator_info("my-sim");
    EXPECT_EQ(info.name, "my-sim");
}

// ── CustomGpuDef ───────────────────────────────────────────────────────────

TEST(CustomGpuDefTest, Construction) {
    CustomGpuDef gpu{"custom-gpu", R"({"cu_count": 256})", "MI300X"};
    EXPECT_EQ(gpu.name, "custom-gpu");
    EXPECT_FALSE(gpu.topology_json.empty());
    EXPECT_EQ(gpu.base_gpu, "MI300X");
}

// ── SessionHealth ──────────────────────────────────────────────────────────

TEST(SessionHealthTest, Defaults) {
    SessionHealth h;
    EXPECT_TRUE(h.session_id.empty());
    EXPECT_EQ(h.status, HealthStatus::Unknown);
    EXPECT_EQ(h.uptime.seconds, 0u);
    EXPECT_TRUE(h.error_message.empty());
}

// ── SessionPerf ────────────────────────────────────────────────────────────

TEST(SessionPerfTest, Defaults) {
    SessionPerf p;
    EXPECT_TRUE(p.session_id.empty());
    EXPECT_EQ(p.ticks, 0u);
    EXPECT_DOUBLE_EQ(p.ipc, 0.0);
    EXPECT_DOUBLE_EQ(p.simulation_speed, 0.0);
    EXPECT_EQ(p.active_contexts, 0u);
}

// ── MockSimulator interface ────────────────────────────────────────────────

TEST(MockSimulatorTest, Info) {
    auto sim = make_mock_simulator();
    auto info = sim->info();
    EXPECT_EQ(info.name, "test-sim");
    EXPECT_EQ(info.version, "1.0.0");
}

TEST(MockSimulatorTest, SupportedGpus) {
    auto sim = make_mock_simulator();
    auto gpus = sim->supported_gpus();
    ASSERT_EQ(gpus.size(), 2u);
    EXPECT_EQ(gpus[0].name, "MI300X");
    EXPECT_EQ(gpus[1].name, "MI325X");
}

TEST(MockSimulatorTest, SetCustomGpu) {
    auto sim = make_mock_simulator();
    CustomGpuDef custom{"custom-1", R"({"cu": 128})", ""};
    auto gpu = sim->set_custom_gpu(custom);
    EXPECT_EQ(gpu.name, "custom-1");
    EXPECT_EQ(gpu.arch, "custom_arch");

    // Now supported_gpus should include the custom one
    auto gpus = sim->supported_gpus();
    EXPECT_EQ(gpus.size(), 3u);
    EXPECT_EQ(gpus.back().name, "custom-1");
}

TEST(MockSimulatorTest, CreateSession) {
    auto sim = make_mock_simulator();
    auto container =
        sim->create_session(make_test_session(), make_test_profile());
    EXPECT_EQ(container.image, "ghcr.io/rocm/pytorch:latest");
    ASSERT_GE(container.env.size(), 2u);
    EXPECT_EQ(container.env[0].value, "test-sim");
    EXPECT_EQ(sim->session_count(), 1u);
}

TEST(MockSimulatorTest, CreateSessionDefaultImage) {
    auto sim = make_mock_simulator();
    auto s = make_test_session();
    s.image = "";
    auto container = sim->create_session(s, make_test_profile());
    EXPECT_EQ(container.image, "default:latest");
}

TEST(MockSimulatorTest, DeleteSession) {
    auto sim = make_mock_simulator();
    sim->create_session(make_test_session("s1"), make_test_profile());
    EXPECT_EQ(sim->session_count(), 1u);
    sim->delete_session("s1");
    EXPECT_EQ(sim->session_count(), 0u);
}

TEST(MockSimulatorTest, GetSessionHealth) {
    auto sim = make_mock_simulator();
    sim->create_session(make_test_session("h1"), make_test_profile());
    auto health = sim->get_session_health("h1");
    EXPECT_EQ(health.status, HealthStatus::Healthy);

    auto unhealthy = sim->get_session_health("nonexistent");
    EXPECT_EQ(unhealthy.status, HealthStatus::Unhealthy);
}

TEST(MockSimulatorTest, GetSessionPerf) {
    auto sim = make_mock_simulator();
    auto perf = sim->get_session_perf("any");
    EXPECT_EQ(perf.ticks, 1000u);
    EXPECT_DOUBLE_EQ(perf.ipc, 2.5);
}

TEST(MockSimulatorTest, GetRunDef) {
    auto sim = make_mock_simulator();
    RunDef run;
    run.session = "s1";
    run.exec.command = "python";
    run.exec.args = {"train.py"};

    auto exec = sim->get_run_def(run);
    EXPECT_EQ(exec.command, "python");
    EXPECT_EQ(exec.args.size(), 1u);
    ASSERT_GE(exec.env.size(), 1u);
    EXPECT_EQ(exec.env.back().key, "LD_PRELOAD");
}

TEST(MockSimulatorTest, CreateSessionFailure) {
    auto sim = make_mock_simulator();
    sim->set_fail_create(true);
    EXPECT_THROW(sim->create_session(make_test_session(), make_test_profile()),
                 std::runtime_error);
}

} // namespace
} // namespace mirage
