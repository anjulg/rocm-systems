#include "mirage/profile_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <thread>
#include <vector>

namespace mirage {
namespace {

using Error = ProfileRegistry::Error;

ProfileDef valid_profile(const std::string& name = "profile-1") {
    return {name, "test-sim", SimulatorMode::Functional, "MI300X", 4, 2};
}

// ── Add ────────────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, AddSuccess) {
    ProfileRegistry reg;
    EXPECT_EQ(reg.add(valid_profile()), Error::Ok);
    EXPECT_EQ(reg.size(), 1u);
}

TEST(ProfileRegistryTest, AddDuplicate) {
    ProfileRegistry reg;
    EXPECT_EQ(reg.add(valid_profile()), Error::Ok);
    EXPECT_EQ(reg.add(valid_profile()), Error::DuplicateName);
    EXPECT_EQ(reg.size(), 1u);
}

TEST(ProfileRegistryTest, AddMultiple) {
    ProfileRegistry reg;
    EXPECT_EQ(reg.add(valid_profile("p1")), Error::Ok);
    EXPECT_EQ(reg.add(valid_profile("p2")), Error::Ok);
    EXPECT_EQ(reg.add(valid_profile("p3")), Error::Ok);
    EXPECT_EQ(reg.size(), 3u);
}

// ── Validation ─────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, RejectsEmptyName) {
    ProfileRegistry reg;
    auto p = valid_profile();
    p.name = "";
    EXPECT_EQ(reg.add(p), Error::EmptyName);
    EXPECT_EQ(reg.size(), 0u);
}

TEST(ProfileRegistryTest, RejectsEmptySimulator) {
    ProfileRegistry reg;
    auto p = valid_profile();
    p.simulator = "";
    EXPECT_EQ(reg.add(p), Error::EmptySimulator);
}

TEST(ProfileRegistryTest, RejectsEmptyGpu) {
    ProfileRegistry reg;
    auto p = valid_profile();
    p.gpu = "";
    EXPECT_EQ(reg.add(p), Error::EmptyGpu);
}

TEST(ProfileRegistryTest, RejectsZeroGpuCount) {
    ProfileRegistry reg;
    auto p = valid_profile();
    p.num_gpus = 0;
    EXPECT_EQ(reg.add(p), Error::InvalidGpuCount);
}

TEST(ProfileRegistryTest, RejectsZeroNodeCount) {
    ProfileRegistry reg;
    auto p = valid_profile();
    p.num_nodes = 0;
    EXPECT_EQ(reg.add(p), Error::InvalidNodeCount);
}

// ── Find ───────────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, FindExisting) {
    ProfileRegistry reg;
    auto p = valid_profile("find-me");
    reg.add(p);
    auto found = reg.find("find-me");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->name, "find-me");
    EXPECT_EQ(found->gpu, "MI300X");
}

TEST(ProfileRegistryTest, FindNonExisting) {
    ProfileRegistry reg;
    EXPECT_FALSE(reg.find("nope").has_value());
}

TEST(ProfileRegistryTest, FindReturnsCorrectProfile) {
    ProfileRegistry reg;
    reg.add(valid_profile("a"));
    auto pb = valid_profile("b");
    pb.gpu = "MI325X";
    pb.num_gpus = 8;
    reg.add(pb);

    auto found = reg.find("b");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->gpu, "MI325X");
    EXPECT_EQ(found->num_gpus, 8u);
}

// ── Remove ─────────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, RemoveExisting) {
    ProfileRegistry reg;
    reg.add(valid_profile("del-me"));
    EXPECT_EQ(reg.remove("del-me"), Error::Ok);
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_FALSE(reg.find("del-me").has_value());
}

TEST(ProfileRegistryTest, RemoveNonExisting) {
    ProfileRegistry reg;
    EXPECT_EQ(reg.remove("nope"), Error::NotFound);
}

TEST(ProfileRegistryTest, RemoveDoesNotAffectOthers) {
    ProfileRegistry reg;
    reg.add(valid_profile("a"));
    reg.add(valid_profile("b"));
    reg.remove("a");
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_TRUE(reg.find("b").has_value());
}

// ── All ────────────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, AllEmpty) {
    ProfileRegistry reg;
    EXPECT_TRUE(reg.all().empty());
}

TEST(ProfileRegistryTest, AllReturnsAllProfiles) {
    ProfileRegistry reg;
    reg.add(valid_profile("x"));
    reg.add(valid_profile("y"));
    reg.add(valid_profile("z"));
    auto all = reg.all();
    EXPECT_EQ(all.size(), 3u);

    std::vector<std::string> names;
    for (const auto& p : all) names.push_back(p.name);
    std::sort(names.begin(), names.end());
    EXPECT_EQ(names, (std::vector<std::string>{"x", "y", "z"}));
}

// ── Clear ──────────────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, Clear) {
    ProfileRegistry reg;
    reg.add(valid_profile("a"));
    reg.add(valid_profile("b"));
    reg.clear();
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_TRUE(reg.all().empty());
}

// ── Error strings ──────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, ErrorStrings) {
    EXPECT_STREQ(ProfileRegistry::error_string(Error::Ok), "Ok");
    EXPECT_STREQ(ProfileRegistry::error_string(Error::DuplicateName),
                 "DuplicateName");
    EXPECT_STREQ(ProfileRegistry::error_string(Error::EmptyName), "EmptyName");
    EXPECT_STREQ(ProfileRegistry::error_string(Error::NotFound), "NotFound");
}

// ── Thread safety ──────────────────────────────────────────────────────────

TEST(ProfileRegistryTest, ConcurrentAdds) {
    ProfileRegistry reg;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 100;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&reg, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                auto name = "t" + std::to_string(t) + "-" + std::to_string(i);
                reg.add(valid_profile(name));
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_EQ(reg.size(), kThreads * kPerThread);
}

TEST(ProfileRegistryTest, ConcurrentReads) {
    ProfileRegistry reg;
    for (int i = 0; i < 50; ++i) {
        reg.add(valid_profile("p" + std::to_string(i)));
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&reg]() {
            for (int i = 0; i < 50; ++i) {
                auto r = reg.find("p" + std::to_string(i));
                EXPECT_TRUE(r.has_value());
            }
        });
    }
    for (auto& t : threads) t.join();
}

} // namespace
} // namespace mirage
