#include "mirage/daemon.h"
#include "mirage/simulator.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

#include <memory>

namespace mirage {
namespace {

using mirage::testing::MockSimulator;
using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;
using mirage::testing::make_test_simulator_info;

// ---------------------------------------------------------------------------
//  Daemon ↔ Simulator integration
// ---------------------------------------------------------------------------

class DaemonSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto p = make_test_profile();
        config_.profiles.push_back(p);
        daemon_ = std::make_unique<Daemon>(config_);
        mock_ = make_mock_simulator();
        daemon_->register_simulator(mock_);
    }

    DaemonConfig config_;
    std::unique_ptr<Daemon> daemon_;
    std::shared_ptr<MockSimulator> mock_;
};

TEST_F(DaemonSimulatorTest, SimulatorInfoAccessibleThroughDaemon) {
    auto sim = daemon_->find_simulator("test-sim");
    ASSERT_NE(sim, nullptr);
    auto info = sim->info();
    EXPECT_EQ(info.name, "test-sim");
    EXPECT_EQ(info.version, "1.0.0");
    EXPECT_EQ(info.supported_gpus.size(), 2u);
}

TEST_F(DaemonSimulatorTest, DaemonCreatesSessionUsingSimulator) {
    EXPECT_EQ(daemon_->create_session(make_test_session()),
              SessionManager::Error::Ok);
    // Simulator should have tracked the session
    EXPECT_EQ(mock_->session_count(), 1u);
}

TEST_F(DaemonSimulatorTest, DaemonDeletesSessionUsingSimulator) {
    daemon_->create_session(make_test_session());
    EXPECT_EQ(mock_->session_count(), 1u);
    daemon_->delete_session("test-session");
    EXPECT_EQ(mock_->session_count(), 0u);
}

TEST_F(DaemonSimulatorTest, ContainerEnvContainsSimulatorName) {
    daemon_->create_session(make_test_session());
    auto state = daemon_->find_session("test-session");
    ASSERT_TRUE(state.has_value());

    bool found_sim = false;
    for (const auto& e : state->container.env) {
        if (e.key == "SIMULATOR" && e.value == "test-sim") {
            found_sim = true;
        }
    }
    EXPECT_TRUE(found_sim) << "Container env should include SIMULATOR name";
}

TEST_F(DaemonSimulatorTest, ContainerEnvContainsGpuName) {
    daemon_->create_session(make_test_session());
    auto state = daemon_->find_session("test-session");
    ASSERT_TRUE(state.has_value());

    bool found_gpu = false;
    for (const auto& e : state->container.env) {
        if (e.key == "GPU" && e.value == "MI300X") {
            found_gpu = true;
        }
    }
    EXPECT_TRUE(found_gpu) << "Container env should include GPU name";
}

TEST_F(DaemonSimulatorTest, HealthDelegatesToSimulator) {
    daemon_->create_session(make_test_session());
    auto health = daemon_->session_health("test-session");
    ASSERT_TRUE(health.has_value());
    EXPECT_EQ(health->status, HealthStatus::Healthy);
    EXPECT_EQ(health->uptime.seconds, 42u);
}

TEST_F(DaemonSimulatorTest, PerfDelegatesToSimulator) {
    daemon_->create_session(make_test_session());
    auto perf = daemon_->session_perf("test-session");
    ASSERT_TRUE(perf.has_value());
    EXPECT_EQ(perf->ticks, 1000u);
    EXPECT_DOUBLE_EQ(perf->ipc, 2.5);
    EXPECT_DOUBLE_EQ(perf->simulation_speed, 0.8);
    EXPECT_EQ(perf->active_contexts, 64u);
}

TEST_F(DaemonSimulatorTest, RunDefDelegatesToSimulator) {
    daemon_->create_session(make_test_session());

    RunDef run;
    run.session = "test-session";
    run.exec.command = "python";
    run.exec.args = {"train.py"};

    auto exec = daemon_->get_run_def(run);
    ASSERT_TRUE(exec.has_value());
    EXPECT_EQ(exec->command, "python");
    // Simulator should have injected LD_PRELOAD
    bool has_preload = false;
    for (const auto& e : exec->env) {
        if (e.key == "LD_PRELOAD") has_preload = true;
    }
    EXPECT_TRUE(has_preload);
}

// ---------------------------------------------------------------------------
//  Multiple simulators
// ---------------------------------------------------------------------------

TEST_F(DaemonSimulatorTest, MultipleSimulatorRegistration) {
    auto sim2 = make_mock_simulator("alt-sim");
    daemon_->register_simulator(sim2);

    EXPECT_EQ(daemon_->simulator_names().size(), 2u);
    EXPECT_NE(daemon_->find_simulator("test-sim"), nullptr);
    EXPECT_NE(daemon_->find_simulator("alt-sim"), nullptr);
}

TEST_F(DaemonSimulatorTest, SessionUsesCorrectSimulator) {
    auto sim2_info = make_test_simulator_info("alt-sim");
    auto sim2 = std::make_shared<MockSimulator>(sim2_info);
    daemon_->register_simulator(sim2);

    auto alt_profile =
        make_test_profile("alt-profile", "alt-sim", "MI300X");
    daemon_->add_profile(alt_profile);

    auto session = make_test_session("alt-session", "alt-profile");
    EXPECT_EQ(daemon_->create_session(session), SessionManager::Error::Ok);

    auto state = daemon_->find_session("alt-session");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->simulator_name, "alt-sim");

    // alt-sim should have the session, original should not
    EXPECT_EQ(sim2->session_count(), 1u);
    EXPECT_EQ(mock_->session_count(), 0u);
}

// ---------------------------------------------------------------------------
//  Custom GPU through daemon
// ---------------------------------------------------------------------------

TEST_F(DaemonSimulatorTest, CustomGpuThroughSimulator) {
    auto sim = daemon_->find_simulator("test-sim");
    ASSERT_NE(sim, nullptr);

    auto initial_gpus = sim->supported_gpus();
    EXPECT_EQ(initial_gpus.size(), 2u);

    CustomGpuDef custom{"my-custom-gpu", R"({"cu_count": 512})", "MI300X"};
    auto gpu = sim->set_custom_gpu(custom);
    EXPECT_EQ(gpu.name, "my-custom-gpu");

    auto updated_gpus = sim->supported_gpus();
    EXPECT_EQ(updated_gpus.size(), 3u);
}

// ---------------------------------------------------------------------------
//  Simulator failure handling
// ---------------------------------------------------------------------------

TEST_F(DaemonSimulatorTest, SimulatorCreateFailure) {
    mock_->set_fail_create(true);
    EXPECT_EQ(daemon_->create_session(make_test_session()),
              SessionManager::Error::SimulatorError);
    EXPECT_TRUE(daemon_->sessions().empty());
}

TEST_F(DaemonSimulatorTest, UnregisterSimulatorDoesNotAffectSessions) {
    daemon_->create_session(make_test_session());
    daemon_->unregister_simulator("test-sim");

    // Session should still exist (orphaned but findable)
    auto state = daemon_->find_session("test-session");
    EXPECT_TRUE(state.has_value());

    // But health/perf queries should return nullopt (no simulator)
    EXPECT_FALSE(daemon_->session_health("test-session").has_value());
    EXPECT_FALSE(daemon_->session_perf("test-session").has_value());
}

} // namespace
} // namespace mirage
