#include "mirage/daemon.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

namespace mirage {
namespace {

using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;

class DaemonTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto profile = make_test_profile();
        DaemonConfig config;
        config.profiles.push_back(profile);
        daemon_ = std::make_unique<Daemon>(std::move(config));
        daemon_->register_simulator(make_mock_simulator());
    }

    std::unique_ptr<Daemon> daemon_;
};

// ── Profiles ───────────────────────────────────────────────────────────────

TEST_F(DaemonTest, InitialProfiles) {
    auto profiles = daemon_->profiles();
    ASSERT_EQ(profiles.size(), 1u);
    EXPECT_EQ(profiles[0].name, "test-profile");
}

TEST_F(DaemonTest, AddProfile) {
    auto p = make_test_profile("new-profile");
    EXPECT_EQ(daemon_->add_profile(p), ProfileRegistry::Error::Ok);
    EXPECT_EQ(daemon_->profiles().size(), 2u);
}

TEST_F(DaemonTest, AddDuplicateProfile) {
    EXPECT_EQ(daemon_->add_profile(make_test_profile()),
              ProfileRegistry::Error::DuplicateName);
}

TEST_F(DaemonTest, FindProfile) {
    auto found = daemon_->find_profile("test-profile");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->gpu, "MI300X");
}

TEST_F(DaemonTest, FindProfileNotFound) {
    EXPECT_FALSE(daemon_->find_profile("nope").has_value());
}

TEST_F(DaemonTest, RemoveProfile) {
    EXPECT_EQ(daemon_->remove_profile("test-profile"),
              ProfileRegistry::Error::Ok);
    EXPECT_TRUE(daemon_->profiles().empty());
}

TEST_F(DaemonTest, RemoveProfileNotFound) {
    EXPECT_EQ(daemon_->remove_profile("nope"),
              ProfileRegistry::Error::NotFound);
}

// ── Simulators ─────────────────────────────────────────────────────────────

TEST_F(DaemonTest, SimulatorRegistered) {
    auto names = daemon_->simulator_names();
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "test-sim");
}

TEST_F(DaemonTest, FindSimulator) {
    auto sim = daemon_->find_simulator("test-sim");
    ASSERT_NE(sim, nullptr);
    EXPECT_EQ(sim->info().name, "test-sim");
}

TEST_F(DaemonTest, UnregisterSimulator) {
    EXPECT_TRUE(daemon_->unregister_simulator("test-sim"));
    EXPECT_TRUE(daemon_->simulator_names().empty());
}

// ── Sessions ───────────────────────────────────────────────────────────────

TEST_F(DaemonTest, CreateSession) {
    EXPECT_EQ(daemon_->create_session(make_test_session()),
              SessionManager::Error::Ok);
    auto sessions = daemon_->sessions();
    ASSERT_EQ(sessions.size(), 1u);
    EXPECT_EQ(sessions[0].definition.name, "test-session");
}

TEST_F(DaemonTest, CreateSessionResolvesProfile) {
    auto s = make_test_session("s1", "test-profile");
    EXPECT_EQ(daemon_->create_session(s), SessionManager::Error::Ok);
    auto state = daemon_->find_session("s1");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->profile.name, "test-profile");
    EXPECT_EQ(state->profile.gpu, "MI300X");
}

TEST_F(DaemonTest, CreateSessionProfileNotFound) {
    auto s = make_test_session("bad", "nonexistent-profile");
    EXPECT_EQ(daemon_->create_session(s), SessionManager::Error::EmptyProfile);
}

TEST_F(DaemonTest, CreateDuplicateSession) {
    daemon_->create_session(make_test_session());
    EXPECT_EQ(daemon_->create_session(make_test_session()),
              SessionManager::Error::DuplicateSession);
}

TEST_F(DaemonTest, DeleteSession) {
    daemon_->create_session(make_test_session());
    EXPECT_EQ(daemon_->delete_session("test-session"),
              SessionManager::Error::Ok);
    EXPECT_TRUE(daemon_->sessions().empty());
}

TEST_F(DaemonTest, DeleteSessionNotFound) {
    EXPECT_EQ(daemon_->delete_session("nope"),
              SessionManager::Error::SessionNotFound);
}

TEST_F(DaemonTest, FindSession) {
    daemon_->create_session(make_test_session());
    auto state = daemon_->find_session("test-session");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->definition.image, "ghcr.io/rocm/pytorch:latest");
}

TEST_F(DaemonTest, FindSessionNotFound) {
    EXPECT_FALSE(daemon_->find_session("nope").has_value());
}

// ── Health / Perf ──────────────────────────────────────────────────────────

TEST_F(DaemonTest, SessionHealth) {
    daemon_->create_session(make_test_session());
    auto health = daemon_->session_health("test-session");
    ASSERT_TRUE(health.has_value());
    EXPECT_EQ(health->status, HealthStatus::Healthy);
}

TEST_F(DaemonTest, SessionHealthNotFound) {
    EXPECT_FALSE(daemon_->session_health("nope").has_value());
}

TEST_F(DaemonTest, SessionPerf) {
    daemon_->create_session(make_test_session());
    auto perf = daemon_->session_perf("test-session");
    ASSERT_TRUE(perf.has_value());
    EXPECT_EQ(perf->ticks, 1000u);
    EXPECT_DOUBLE_EQ(perf->ipc, 2.5);
}

TEST_F(DaemonTest, SessionPerfNotFound) {
    EXPECT_FALSE(daemon_->session_perf("nope").has_value());
}

// ── Run support ────────────────────────────────────────────────────────────

TEST_F(DaemonTest, GetRunDef) {
    daemon_->create_session(make_test_session());

    RunDef run;
    run.session = "test-session";
    run.exec.command = "python";
    run.exec.args = {"train.py"};

    auto exec = daemon_->get_run_def(run);
    ASSERT_TRUE(exec.has_value());
    EXPECT_EQ(exec->command, "python");
    ASSERT_GE(exec->env.size(), 1u);
    EXPECT_EQ(exec->env.back().key, "LD_PRELOAD");
}

TEST_F(DaemonTest, GetRunDefSessionNotFound) {
    RunDef run;
    run.session = "nonexistent";
    run.exec.command = "ls";
    EXPECT_FALSE(daemon_->get_run_def(run).has_value());
}

// ── Empty daemon ───────────────────────────────────────────────────────────

TEST(DaemonEmptyTest, DefaultConstruction) {
    Daemon d;
    EXPECT_TRUE(d.profiles().empty());
    EXPECT_TRUE(d.sessions().empty());
    EXPECT_TRUE(d.simulator_names().empty());
}

TEST(DaemonEmptyTest, MultipleProfilesInConfig) {
    DaemonConfig config;
    config.profiles.push_back(make_test_profile("p1"));
    config.profiles.push_back(make_test_profile("p2"));
    config.profiles.push_back(make_test_profile("p3"));
    Daemon d(config);
    EXPECT_EQ(d.profiles().size(), 3u);
}

} // namespace
} // namespace mirage
