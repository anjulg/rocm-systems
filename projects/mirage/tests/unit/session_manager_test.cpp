#include "mirage/session_manager.h"

#include "../mock_simulator.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace mirage {
namespace {

using Error = SessionManager::Error;
using mirage::testing::make_mock_simulator;
using mirage::testing::make_test_profile;
using mirage::testing::make_test_session;

class SessionManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mgr_.register_simulator(make_mock_simulator());
    }

    SessionManager mgr_;
};

// ── Simulator registration ─────────────────────────────────────────────────

TEST_F(SessionManagerTest, RegisterSimulator) {
    auto names = mgr_.simulator_names();
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "test-sim");
}

TEST_F(SessionManagerTest, FindRegisteredSimulator) {
    auto sim = mgr_.find_simulator("test-sim");
    ASSERT_NE(sim, nullptr);
    EXPECT_EQ(sim->info().name, "test-sim");
}

TEST_F(SessionManagerTest, FindUnknownSimulator) {
    EXPECT_EQ(mgr_.find_simulator("unknown"), nullptr);
}

TEST_F(SessionManagerTest, UnregisterSimulator) {
    EXPECT_TRUE(mgr_.unregister_simulator("test-sim"));
    EXPECT_EQ(mgr_.find_simulator("test-sim"), nullptr);
    EXPECT_TRUE(mgr_.simulator_names().empty());
}

TEST_F(SessionManagerTest, UnregisterNonexistent) {
    EXPECT_FALSE(mgr_.unregister_simulator("nope"));
}

TEST_F(SessionManagerTest, MultipleSimulators) {
    mgr_.register_simulator(make_mock_simulator("sim-2"));
    auto names = mgr_.simulator_names();
    EXPECT_EQ(names.size(), 2u);
}

// ── Session creation ───────────────────────────────────────────────────────

TEST_F(SessionManagerTest, CreateSession) {
    auto err = mgr_.create_session(make_test_session(), make_test_profile());
    EXPECT_EQ(err, Error::Ok);
    EXPECT_EQ(mgr_.session_count(), 1u);
}

TEST_F(SessionManagerTest, CreateSessionDuplicate) {
    mgr_.create_session(make_test_session(), make_test_profile());
    auto err = mgr_.create_session(make_test_session(), make_test_profile());
    EXPECT_EQ(err, Error::DuplicateSession);
}

TEST_F(SessionManagerTest, CreateSessionEmptyName) {
    auto s = make_test_session();
    s.name = "";
    EXPECT_EQ(mgr_.create_session(s, make_test_profile()), Error::EmptyName);
}

TEST_F(SessionManagerTest, CreateSessionEmptyProfile) {
    auto s = make_test_session();
    s.profile = "";
    EXPECT_EQ(mgr_.create_session(s, make_test_profile()),
              Error::EmptyProfile);
}

TEST_F(SessionManagerTest, CreateSessionSimulatorNotFound) {
    auto p = make_test_profile();
    p.simulator = "nonexistent";
    EXPECT_EQ(mgr_.create_session(make_test_session(), p),
              Error::SimulatorNotFound);
}

TEST_F(SessionManagerTest, CreateSessionSimulatorError) {
    auto mock =
        std::dynamic_pointer_cast<mirage::testing::MockSimulator>(
            mgr_.find_simulator("test-sim"));
    mock->set_fail_create(true);
    EXPECT_EQ(mgr_.create_session(make_test_session(), make_test_profile()),
              Error::SimulatorError);
}

// ── Session lookup ─────────────────────────────────────────────────────────

TEST_F(SessionManagerTest, FindSession) {
    mgr_.create_session(make_test_session("s1"), make_test_profile());
    auto state = mgr_.find_session("s1");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->definition.name, "s1");
    EXPECT_EQ(state->simulator_name, "test-sim");
    EXPECT_EQ(state->health, HealthStatus::Unknown);
}

TEST_F(SessionManagerTest, FindSessionNotFound) {
    EXPECT_FALSE(mgr_.find_session("nope").has_value());
}

TEST_F(SessionManagerTest, AllSessions) {
    mgr_.create_session(make_test_session("s1"), make_test_profile());
    mgr_.create_session(make_test_session("s2"), make_test_profile());
    auto all = mgr_.all_sessions();
    EXPECT_EQ(all.size(), 2u);
}

// ── Session deletion ───────────────────────────────────────────────────────

TEST_F(SessionManagerTest, DeleteSession) {
    mgr_.create_session(make_test_session("del"), make_test_profile());
    EXPECT_EQ(mgr_.delete_session("del"), Error::Ok);
    EXPECT_EQ(mgr_.session_count(), 0u);
    EXPECT_FALSE(mgr_.find_session("del").has_value());
}

TEST_F(SessionManagerTest, DeleteSessionNotFound) {
    EXPECT_EQ(mgr_.delete_session("nope"), Error::SessionNotFound);
}

// ── Session health ─────────────────────────────────────────────────────────

TEST_F(SessionManagerTest, GetSessionHealth) {
    mgr_.create_session(make_test_session("h1"), make_test_profile());
    auto health = mgr_.get_session_health("h1");
    ASSERT_TRUE(health.has_value());
    EXPECT_EQ(health->session_id, "h1");
    EXPECT_EQ(health->status, HealthStatus::Healthy);
    EXPECT_EQ(health->uptime.seconds, 42u);
}

TEST_F(SessionManagerTest, GetSessionHealthNotFound) {
    EXPECT_FALSE(mgr_.get_session_health("nope").has_value());
}

// ── Session perf ───────────────────────────────────────────────────────────

TEST_F(SessionManagerTest, GetSessionPerf) {
    mgr_.create_session(make_test_session("perf1"), make_test_profile());
    auto perf = mgr_.get_session_perf("perf1");
    ASSERT_TRUE(perf.has_value());
    EXPECT_EQ(perf->session_id, "perf1");
    EXPECT_EQ(perf->ticks, 1000u);
    EXPECT_DOUBLE_EQ(perf->ipc, 2.5);
    EXPECT_EQ(perf->active_contexts, 64u);
}

TEST_F(SessionManagerTest, GetSessionPerfNotFound) {
    EXPECT_FALSE(mgr_.get_session_perf("nope").has_value());
}

// ── Container creation ─────────────────────────────────────────────────────

TEST_F(SessionManagerTest, ContainerHasSimulatorEnv) {
    mgr_.create_session(make_test_session("c1"), make_test_profile());
    auto state = mgr_.find_session("c1");
    ASSERT_TRUE(state.has_value());
    const auto& env = state->container.env;
    ASSERT_GE(env.size(), 2u);
    EXPECT_EQ(env[0].key, "SIMULATOR");
    EXPECT_EQ(env[0].value, "test-sim");
    EXPECT_EQ(env[1].key, "GPU");
    EXPECT_EQ(env[1].value, "MI300X");
}

TEST_F(SessionManagerTest, ContainerUsesSessionImage) {
    auto s = make_test_session("img-test");
    s.image = "custom:v1";
    mgr_.create_session(s, make_test_profile());
    auto state = mgr_.find_session("img-test");
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(state->container.image, "custom:v1");
}

// ── Error strings ──────────────────────────────────────────────────────────

TEST(SessionManagerErrorTest, ErrorStrings) {
    EXPECT_STREQ(SessionManager::error_string(Error::Ok), "Ok");
    EXPECT_STREQ(SessionManager::error_string(Error::DuplicateSession),
                 "DuplicateSession");
    EXPECT_STREQ(SessionManager::error_string(Error::SimulatorNotFound),
                 "SimulatorNotFound");
    EXPECT_STREQ(SessionManager::error_string(Error::SimulatorError),
                 "SimulatorError");
}

// ── Thread safety ──────────────────────────────────────────────────────────

TEST_F(SessionManagerTest, ConcurrentSessionCreation) {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 25;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([this, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                auto name = "t" + std::to_string(t) + "-s" + std::to_string(i);
                mgr_.create_session(make_test_session(name),
                                    make_test_profile());
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_EQ(mgr_.session_count(),
              static_cast<size_t>(kThreads * kPerThread));
}

} // namespace
} // namespace mirage
