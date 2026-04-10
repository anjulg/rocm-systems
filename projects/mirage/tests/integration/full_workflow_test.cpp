#include "mirage/daemon.h"
#include "mirage/time_utils.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace mirage {
namespace {

using mirage::testing::MockSimulator;
using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;
using mirage::testing::make_test_simulator_info;

/// End-to-end workflow: configure → register → create sessions → run → teardown
class FullWorkflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a daemon with several profiles and two simulators
        DaemonConfig config;
        config.profiles.push_back(
            {"func-mi300", "rocjitsu", SimulatorMode::Functional, "MI300X", 4,
             1});
        config.profiles.push_back(
            {"cycle-mi325", "rocjitsu", SimulatorMode::CycleAccurate, "MI325X",
             8, 2});
        config.profiles.push_back(
            {"riscv-func", "rv-sim", SimulatorMode::Functional, "RV64", 1, 1});

        daemon_ = std::make_unique<Daemon>(std::move(config));

        // Register two different simulators
        auto rj_info = make_test_simulator_info("rocjitsu");
        rj_info.supported_gpus = {
            {"MI300X", "gfx942", GpuFamily::AmdCdna, ""},
            {"MI325X", "gfx950", GpuFamily::AmdCdna, ""},
        };
        rocjitsu_ = std::make_shared<MockSimulator>(rj_info);
        daemon_->register_simulator(rocjitsu_);

        auto rv_info = make_test_simulator_info("rv-sim");
        rv_info.supported_gpus = {
            {"RV64", "rv64i", GpuFamily::RiscV, "RISC-V accelerator"},
        };
        rv_info.supports_custom_gpus = false;
        rvsim_ = std::make_shared<MockSimulator>(rv_info);
        daemon_->register_simulator(rvsim_);
    }

    std::unique_ptr<Daemon> daemon_;
    std::shared_ptr<MockSimulator> rocjitsu_;
    std::shared_ptr<MockSimulator> rvsim_;
};

TEST_F(FullWorkflowTest, CompleteAmdWorkflow) {
    // 1. Verify profiles loaded
    EXPECT_EQ(daemon_->profiles().size(), 3u);

    // 2. Create session for MI300X
    SessionDef sd{"pytorch-bench", "func-mi300", "pytorch:latest"};
    ASSERT_EQ(daemon_->create_session(sd), SessionManager::Error::Ok);

    // 3. Verify session state
    auto state = daemon_->find_session("pytorch-bench");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->simulator_name, "rocjitsu");
    EXPECT_EQ(state->profile.gpu, "MI300X");
    EXPECT_EQ(state->profile.num_gpus, 4u);
    EXPECT_EQ(state->definition.image, "pytorch:latest");

    // 4. Check container env
    bool has_sim_env = false;
    for (const auto& e : state->container.env) {
        if (e.key == "SIMULATOR" && e.value == "rocjitsu") {
            has_sim_env = true;
        }
    }
    EXPECT_TRUE(has_sim_env);

    // 5. Run a training script
    RunDef run;
    run.session = "pytorch-bench";
    run.exec.command = "python";
    run.exec.args = {"train.py", "--epochs=10"};
    auto exec = daemon_->get_run_def(run);
    ASSERT_TRUE(exec.has_value());
    EXPECT_EQ(exec->command, "python");

    // 6. Check health
    auto health = daemon_->session_health("pytorch-bench");
    ASSERT_TRUE(health.has_value());
    EXPECT_EQ(health->status, HealthStatus::Healthy);

    // 7. Check perf
    auto perf = daemon_->session_perf("pytorch-bench");
    ASSERT_TRUE(perf.has_value());
    EXPECT_GT(perf->ticks, 0u);

    // 8. Teardown
    EXPECT_EQ(daemon_->delete_session("pytorch-bench"),
              SessionManager::Error::Ok);
    EXPECT_EQ(rocjitsu_->session_count(), 0u);
}

TEST_F(FullWorkflowTest, CompleteRiscVWorkflow) {
    SessionDef sd{"riscv-test", "riscv-func", "riscv-sdk:v1"};
    ASSERT_EQ(daemon_->create_session(sd), SessionManager::Error::Ok);

    auto state = daemon_->find_session("riscv-test");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->simulator_name, "rv-sim");
    EXPECT_EQ(state->profile.gpu, "RV64");
    EXPECT_EQ(state->profile.num_gpus, 1u);
    EXPECT_EQ(state->profile.num_nodes, 1u);

    daemon_->delete_session("riscv-test");
    EXPECT_EQ(rvsim_->session_count(), 0u);
}

TEST_F(FullWorkflowTest, CycleAccurateSession) {
    SessionDef sd{"cycle-bench", "cycle-mi325", "vllm:latest"};
    ASSERT_EQ(daemon_->create_session(sd), SessionManager::Error::Ok);

    auto state = daemon_->find_session("cycle-bench");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->profile.mode, SimulatorMode::CycleAccurate);
    EXPECT_EQ(state->profile.gpu, "MI325X");
    EXPECT_EQ(state->profile.num_gpus, 8u);
    EXPECT_EQ(state->profile.num_nodes, 2u);
}

TEST_F(FullWorkflowTest, SimultaneousSessionsMultipleSimulators) {
    // Create sessions across both simulators
    daemon_->create_session({"amd-1", "func-mi300", "img:1"});
    daemon_->create_session({"amd-2", "cycle-mi325", "img:2"});
    daemon_->create_session({"rv-1", "riscv-func", "img:3"});

    EXPECT_EQ(daemon_->sessions().size(), 3u);
    EXPECT_EQ(rocjitsu_->session_count(), 2u);
    EXPECT_EQ(rvsim_->session_count(), 1u);

    // Delete one AMD session
    daemon_->delete_session("amd-1");
    EXPECT_EQ(daemon_->sessions().size(), 2u);
    EXPECT_EQ(rocjitsu_->session_count(), 1u);
    EXPECT_EQ(rvsim_->session_count(), 1u);
}

TEST_F(FullWorkflowTest, DynamicProfileAddAndUse) {
    // Add a new profile at runtime
    ProfileDef new_profile{"dynamic-prof", "rocjitsu",
                           SimulatorMode::Clocked, "MI300X", 2, 1};
    EXPECT_EQ(daemon_->add_profile(new_profile), ProfileRegistry::Error::Ok);

    // Use it immediately
    SessionDef sd{"dynamic-session", "dynamic-prof", "pytorch:latest"};
    ASSERT_EQ(daemon_->create_session(sd), SessionManager::Error::Ok);

    auto state = daemon_->find_session("dynamic-session");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->profile.name, "dynamic-prof");
    EXPECT_EQ(state->profile.mode, SimulatorMode::Clocked);
}

TEST_F(FullWorkflowTest, ProfileRemovalDoesNotAffectExistingSessions) {
    daemon_->create_session({"s1", "func-mi300", "img:1"});

    // Remove the profile
    daemon_->remove_profile("func-mi300");

    // Session should still exist
    EXPECT_TRUE(daemon_->find_session("s1").has_value());

    // But can't create new sessions with that profile
    auto result =
        daemon_->create_session({"s2", "func-mi300", "img:2"});
    EXPECT_NE(result, SessionManager::Error::Ok);
}

TEST_F(FullWorkflowTest, CustomGpuWorkflow) {
    auto sim = daemon_->find_simulator("rocjitsu");
    ASSERT_NE(sim, nullptr);

    // Add custom GPU
    CustomGpuDef custom{"my-gpu", R"({"cu": 256, "mem_gb": 192})", "MI300X"};
    auto gpu = sim->set_custom_gpu(custom);
    EXPECT_EQ(gpu.name, "my-gpu");

    // Verify it's now in the GPU list
    auto gpus = sim->supported_gpus();
    bool found = false;
    for (const auto& g : gpus) {
        if (g.name == "my-gpu") found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(FullWorkflowTest, RunWithEnvironmentOverrides) {
    daemon_->create_session({"env-test", "func-mi300", "pytorch:latest"});

    RunDef run;
    run.session = "env-test";
    run.exec.command = "python";
    run.exec.args = {"benchmark.py"};
    run.exec.env = {
        {"HIP_VISIBLE_DEVICES", "0,1,2,3"},
        {"PYTORCH_NO_CUDA_MEMORY_CACHING", "1"},
    };

    auto exec = daemon_->get_run_def(run);
    ASSERT_TRUE(exec.has_value());

    // Should have the original env vars plus simulator-injected ones
    EXPECT_GE(exec->env.size(), 3u);
}

TEST_F(FullWorkflowTest, ConcurrentSessionOperations) {
    constexpr int kSessions = 20;

    // Create sessions concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([this, i]() {
            auto name = "concurrent-" + std::to_string(i);
            daemon_->create_session({name, "func-mi300", "img"});
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(daemon_->sessions().size(), kSessions);

    // Query all health concurrently
    threads.clear();
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([this, i]() {
            auto name = "concurrent-" + std::to_string(i);
            auto h = daemon_->session_health(name);
            EXPECT_TRUE(h.has_value());
        });
    }
    for (auto& t : threads) t.join();

    // Delete all concurrently
    threads.clear();
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([this, i]() {
            auto name = "concurrent-" + std::to_string(i);
            daemon_->delete_session(name);
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(daemon_->sessions().size(), 0u);
}

TEST_F(FullWorkflowTest, TimeUtilsWithSessionPerf) {
    daemon_->create_session({"time-test", "func-mi300", "pytorch:latest"});
    auto perf = daemon_->session_perf("time-test");
    ASSERT_TRUE(perf.has_value());

    // The mock returns zero times, verify they're valid
    EXPECT_TRUE(time_is_zero(perf->simulated_time));
    EXPECT_TRUE(time_is_zero(perf->wall_time));

    // Verify time arithmetic still works with session perf values
    auto combined = time_add(perf->simulated_time, time_from_seconds(1));
    EXPECT_EQ(combined.seconds, 1u);
    EXPECT_EQ(combined.picoseconds, 0u);
}

TEST_F(FullWorkflowTest, FullTeardown) {
    // Create sessions on both simulators
    daemon_->create_session({"s1", "func-mi300", "img"});
    daemon_->create_session({"s2", "cycle-mi325", "img"});
    daemon_->create_session({"s3", "riscv-func", "img"});
    EXPECT_EQ(daemon_->sessions().size(), 3u);

    // Delete all
    daemon_->delete_session("s1");
    daemon_->delete_session("s2");
    daemon_->delete_session("s3");
    EXPECT_EQ(daemon_->sessions().size(), 0u);
    EXPECT_EQ(rocjitsu_->session_count(), 0u);
    EXPECT_EQ(rvsim_->session_count(), 0u);

    // Unregister simulators
    daemon_->unregister_simulator("rocjitsu");
    daemon_->unregister_simulator("rv-sim");
    EXPECT_TRUE(daemon_->simulator_names().empty());

    // Remove profiles
    daemon_->remove_profile("func-mi300");
    daemon_->remove_profile("cycle-mi325");
    daemon_->remove_profile("riscv-func");
    EXPECT_TRUE(daemon_->profiles().empty());
}

} // namespace
} // namespace mirage
