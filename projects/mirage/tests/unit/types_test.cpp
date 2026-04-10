#include "mirage/types.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace mirage {
namespace {

// ── GpuFamily ──────────────────────────────────────────────────────────────

TEST(GpuFamilyTest, DefaultIsUnknown) {
    GpuDef gpu;
    EXPECT_EQ(gpu.family, GpuFamily::Unknown);
}

TEST(GpuFamilyTest, AllValuesHaveStrings) {
    EXPECT_STREQ(to_string(GpuFamily::Unknown), "Unknown");
    EXPECT_STREQ(to_string(GpuFamily::AmdCdna), "AmdCdna");
    EXPECT_STREQ(to_string(GpuFamily::AmdRdna), "AmdRdna");
    EXPECT_STREQ(to_string(GpuFamily::RiscV), "RiscV");
}

// ── SimulatorMode ──────────────────────────────────────────────────────────

TEST(SimulatorModeTest, AllValuesHaveStrings) {
    EXPECT_STREQ(to_string(SimulatorMode::Functional), "Functional");
    EXPECT_STREQ(to_string(SimulatorMode::Clocked), "Clocked");
    EXPECT_STREQ(to_string(SimulatorMode::CycleAccurate), "CycleAccurate");
}

TEST(SimulatorModeTest, DefaultIsFunctional) {
    ProfileDef p;
    EXPECT_EQ(p.mode, SimulatorMode::Functional);
}

// ── Stream ─────────────────────────────────────────────────────────────────

TEST(StreamTest, AllValuesHaveStrings) {
    EXPECT_STREQ(to_string(Stream::Stdin), "Stdin");
    EXPECT_STREQ(to_string(Stream::Stdout), "Stdout");
    EXPECT_STREQ(to_string(Stream::Stderr), "Stderr");
}

// ── HealthStatus ───────────────────────────────────────────────────────────

TEST(HealthStatusTest, AllValuesHaveStrings) {
    EXPECT_STREQ(to_string(HealthStatus::Unknown), "Unknown");
    EXPECT_STREQ(to_string(HealthStatus::Healthy), "Healthy");
    EXPECT_STREQ(to_string(HealthStatus::Unhealthy), "Unhealthy");
}

// ── Time ───────────────────────────────────────────────────────────────────

TEST(TimeTest, DefaultIsZero) {
    Time t;
    EXPECT_EQ(t.seconds, 0u);
    EXPECT_EQ(t.picoseconds, 0u);
}

TEST(TimeTest, Equality) {
    Time a{10, 500};
    Time b{10, 500};
    Time c{10, 501};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(TimeTest, Ordering) {
    Time a{1, 0};
    Time b{2, 0};
    Time c{1, 1};
    EXPECT_LT(a, b);
    EXPECT_LT(a, c);
    EXPECT_GT(b, a);
}

// ── SetEnv ─────────────────────────────────────────────────────────────────

TEST(SetEnvTest, Equality) {
    SetEnv a{"KEY", "VAL"};
    SetEnv b{"KEY", "VAL"};
    SetEnv c{"KEY", "OTHER"};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── GpuDef ─────────────────────────────────────────────────────────────────

TEST(GpuDefTest, Construction) {
    GpuDef gpu{"MI300X", "gfx942", GpuFamily::AmdCdna, "AMD MI300X"};
    EXPECT_EQ(gpu.name, "MI300X");
    EXPECT_EQ(gpu.arch, "gfx942");
    EXPECT_EQ(gpu.family, GpuFamily::AmdCdna);
    EXPECT_EQ(gpu.description, "AMD MI300X");
}

TEST(GpuDefTest, Equality) {
    GpuDef a{"MI300X", "gfx942", GpuFamily::AmdCdna, "AMD MI300X"};
    GpuDef b{"MI300X", "gfx942", GpuFamily::AmdCdna, "AMD MI300X"};
    GpuDef c{"MI325X", "gfx950", GpuFamily::AmdCdna, "AMD MI325X"};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── ProfileDef ─────────────────────────────────────────────────────────────

TEST(ProfileDefTest, Defaults) {
    ProfileDef p;
    EXPECT_TRUE(p.name.empty());
    EXPECT_EQ(p.mode, SimulatorMode::Functional);
    EXPECT_EQ(p.num_gpus, 1u);
    EXPECT_EQ(p.num_nodes, 1u);
}

TEST(ProfileDefTest, FullConstruction) {
    ProfileDef p{"perf-mi300x", "rocjitsu", SimulatorMode::CycleAccurate,
                 "MI300X", 8, 4};
    EXPECT_EQ(p.name, "perf-mi300x");
    EXPECT_EQ(p.simulator, "rocjitsu");
    EXPECT_EQ(p.mode, SimulatorMode::CycleAccurate);
    EXPECT_EQ(p.gpu, "MI300X");
    EXPECT_EQ(p.num_gpus, 8u);
    EXPECT_EQ(p.num_nodes, 4u);
}

TEST(ProfileDefTest, Equality) {
    ProfileDef a{"p1", "sim", SimulatorMode::Functional, "gpu1", 1, 1};
    ProfileDef b{"p1", "sim", SimulatorMode::Functional, "gpu1", 1, 1};
    ProfileDef c{"p2", "sim", SimulatorMode::Functional, "gpu1", 1, 1};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── ExecDef ────────────────────────────────────────────────────────────────

TEST(ExecDefTest, Construction) {
    ExecDef exec;
    exec.command = "python";
    exec.args = {"-c", "print('hello')"};
    exec.env = {{"CUDA_VISIBLE_DEVICES", "0"}};
    EXPECT_EQ(exec.command, "python");
    EXPECT_EQ(exec.args.size(), 2u);
    EXPECT_EQ(exec.env.size(), 1u);
}

// ── StreamData ─────────────────────────────────────────────────────────────

TEST(StreamDataTest, DefaultStream) {
    StreamData sd;
    EXPECT_EQ(sd.stream, Stream::Stdout);
    EXPECT_TRUE(sd.data.empty());
}

TEST(StreamDataTest, WithData) {
    StreamData sd{Stream::Stderr, {72, 101, 108, 108, 111}};
    EXPECT_EQ(sd.stream, Stream::Stderr);
    EXPECT_EQ(sd.data.size(), 5u);
}

// ── RunDef ─────────────────────────────────────────────────────────────────

TEST(RunDefTest, Construction) {
    RunDef run;
    run.session = "my-session";
    run.exec.command = "ls";
    EXPECT_EQ(run.session, "my-session");
    EXPECT_EQ(run.exec.command, "ls");
}

// ── SessionDef ─────────────────────────────────────────────────────────────

TEST(SessionDefTest, Defaults) {
    SessionDef s;
    EXPECT_TRUE(s.name.empty());
    EXPECT_TRUE(s.profile.empty());
    EXPECT_TRUE(s.image.empty());
}

TEST(SessionDefTest, Equality) {
    SessionDef a{"s1", "p1", "img:1"};
    SessionDef b{"s1", "p1", "img:1"};
    SessionDef c{"s2", "p1", "img:1"};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── ClusterDef ─────────────────────────────────────────────────────────────

TEST(ClusterDefTest, Construction) {
    ClusterDef c;
    c.name = "cluster-1";
    c.head_address = "10.0.0.1";
    c.workers_address = {"10.0.0.2", "10.0.0.3"};
    EXPECT_EQ(c.name, "cluster-1");
    EXPECT_EQ(c.workers_address.size(), 2u);
}

// ── RunExit ────────────────────────────────────────────────────────────────

TEST(RunExitTest, Defaults) {
    RunExit re;
    EXPECT_TRUE(re.run_id.empty());
    EXPECT_EQ(re.exit_code, 0);
}

TEST(RunExitTest, NonZeroExit) {
    RunExit re{"run-1", 137};
    EXPECT_EQ(re.run_id, "run-1");
    EXPECT_EQ(re.exit_code, 137);
}

} // namespace
} // namespace mirage
