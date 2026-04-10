#include "mirage/daemon.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

namespace mirage {
namespace {

using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;

/// Tests the full lifecycle of sessions: create → query → delete
class SessionLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        DaemonConfig config;
        config.profiles.push_back(make_test_profile("fast", "test-sim"));
        config.profiles.push_back(
            make_test_profile("accurate", "test-sim", "MI325X"));
        daemon_ = std::make_unique<Daemon>(std::move(config));
        daemon_->register_simulator(make_mock_simulator());
    }

    std::unique_ptr<Daemon> daemon_;
};

TEST_F(SessionLifecycleTest, CreateQueryDelete) {
    // 1. Create
    auto s = make_test_session("lifecycle-1", "fast");
    ASSERT_EQ(daemon_->create_session(s), SessionManager::Error::Ok);

    // 2. Find
    auto state = daemon_->find_session("lifecycle-1");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->definition.name, "lifecycle-1");
    EXPECT_EQ(state->profile.name, "fast");
    EXPECT_EQ(state->health, HealthStatus::Unknown);

    // 3. Check health
    auto health = daemon_->session_health("lifecycle-1");
    ASSERT_TRUE(health.has_value());
    EXPECT_EQ(health->status, HealthStatus::Healthy);

    // 4. Check perf
    auto perf = daemon_->session_perf("lifecycle-1");
    ASSERT_TRUE(perf.has_value());
    EXPECT_GT(perf->ticks, 0u);

    // 5. Delete
    ASSERT_EQ(daemon_->delete_session("lifecycle-1"),
              SessionManager::Error::Ok);
    EXPECT_FALSE(daemon_->find_session("lifecycle-1").has_value());
    EXPECT_FALSE(daemon_->session_health("lifecycle-1").has_value());
}

TEST_F(SessionLifecycleTest, MultipleSessionsSameProfile) {
    auto s1 = make_test_session("s1", "fast");
    auto s2 = make_test_session("s2", "fast");
    auto s3 = make_test_session("s3", "fast");

    EXPECT_EQ(daemon_->create_session(s1), SessionManager::Error::Ok);
    EXPECT_EQ(daemon_->create_session(s2), SessionManager::Error::Ok);
    EXPECT_EQ(daemon_->create_session(s3), SessionManager::Error::Ok);

    EXPECT_EQ(daemon_->sessions().size(), 3u);

    // All share the same profile
    for (const auto& session : daemon_->sessions()) {
        EXPECT_EQ(session.profile.name, "fast");
    }
}

TEST_F(SessionLifecycleTest, SessionsDifferentProfiles) {
    auto s1 = make_test_session("func-session", "fast");
    auto s2 = make_test_session("cycle-session", "accurate");

    EXPECT_EQ(daemon_->create_session(s1), SessionManager::Error::Ok);
    EXPECT_EQ(daemon_->create_session(s2), SessionManager::Error::Ok);

    auto st1 = daemon_->find_session("func-session");
    auto st2 = daemon_->find_session("cycle-session");
    ASSERT_TRUE(st1.has_value());
    ASSERT_TRUE(st2.has_value());
    EXPECT_EQ(st1->profile.gpu, "MI300X");
    EXPECT_EQ(st2->profile.gpu, "MI325X");
}

TEST_F(SessionLifecycleTest, RecreateAfterDelete) {
    auto s = make_test_session("reuse", "fast");
    EXPECT_EQ(daemon_->create_session(s), SessionManager::Error::Ok);
    EXPECT_EQ(daemon_->delete_session("reuse"), SessionManager::Error::Ok);

    // Re-create with the same name
    EXPECT_EQ(daemon_->create_session(s), SessionManager::Error::Ok);
    auto state = daemon_->find_session("reuse");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->definition.name, "reuse");
}

TEST_F(SessionLifecycleTest, DeleteOneOfMany) {
    for (int i = 0; i < 5; ++i) {
        auto s = make_test_session("s" + std::to_string(i), "fast");
        daemon_->create_session(s);
    }
    EXPECT_EQ(daemon_->sessions().size(), 5u);

    EXPECT_EQ(daemon_->delete_session("s2"), SessionManager::Error::Ok);
    EXPECT_EQ(daemon_->sessions().size(), 4u);
    EXPECT_FALSE(daemon_->find_session("s2").has_value());

    // Others should still exist
    EXPECT_TRUE(daemon_->find_session("s0").has_value());
    EXPECT_TRUE(daemon_->find_session("s1").has_value());
    EXPECT_TRUE(daemon_->find_session("s3").has_value());
    EXPECT_TRUE(daemon_->find_session("s4").has_value());
}

TEST_F(SessionLifecycleTest, RunDefInActiveSession) {
    daemon_->create_session(make_test_session("run-test", "fast"));

    RunDef run;
    run.session = "run-test";
    run.exec.command = "/opt/rocm/bin/hipcc";
    run.exec.args = {"--offload-arch=gfx942", "kernel.hip"};
    run.exec.env = {{"HIP_VISIBLE_DEVICES", "0,1,2,3"}};

    auto exec = daemon_->get_run_def(run);
    ASSERT_TRUE(exec.has_value());
    EXPECT_EQ(exec->command, "/opt/rocm/bin/hipcc");
    EXPECT_EQ(exec->args.size(), 2u);
    // Original env + simulator-injected env
    EXPECT_GE(exec->env.size(), 2u);
}

TEST_F(SessionLifecycleTest, RunDefAfterDeleteFails) {
    daemon_->create_session(make_test_session("ephemeral", "fast"));
    daemon_->delete_session("ephemeral");

    RunDef run;
    run.session = "ephemeral";
    run.exec.command = "ls";
    EXPECT_FALSE(daemon_->get_run_def(run).has_value());
}

TEST_F(SessionLifecycleTest, ContainerImageFromSession) {
    auto s = make_test_session("custom-img", "fast");
    s.image = "ghcr.io/custom/image:v2";
    daemon_->create_session(s);

    auto state = daemon_->find_session("custom-img");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->container.image, "ghcr.io/custom/image:v2");
}

TEST_F(SessionLifecycleTest, SessionRemembersSimulatorName) {
    daemon_->create_session(make_test_session("sim-track", "fast"));
    auto state = daemon_->find_session("sim-track");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->simulator_name, "test-sim");
}

} // namespace
} // namespace mirage
