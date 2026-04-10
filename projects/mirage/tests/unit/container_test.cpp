#include "mirage/container.h"

#include <gtest/gtest.h>

namespace mirage {
namespace {

// ── BindMount ──────────────────────────────────────────────────────────────

TEST(BindMountTest, Defaults) {
    BindMount bm;
    EXPECT_TRUE(bm.host_path.empty());
    EXPECT_TRUE(bm.container_path.empty());
    EXPECT_TRUE(bm.readonly);
}

TEST(BindMountTest, Construction) {
    BindMount bm{"/host/lib", "/container/lib", false};
    EXPECT_EQ(bm.host_path, "/host/lib");
    EXPECT_EQ(bm.container_path, "/container/lib");
    EXPECT_FALSE(bm.readonly);
}

TEST(BindMountTest, Equality) {
    BindMount a{"/a", "/b", true};
    BindMount b{"/a", "/b", true};
    BindMount c{"/a", "/c", true};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── InjectedFile ───────────────────────────────────────────────────────────

TEST(InjectedFileTest, Defaults) {
    InjectedFile f;
    EXPECT_TRUE(f.container_path.empty());
    EXPECT_TRUE(f.content.empty());
    EXPECT_TRUE(f.readonly);
    EXPECT_TRUE(f.label.empty());
}

TEST(InjectedFileTest, WithContent) {
    std::vector<uint8_t> data = {'{', '}', '\n'};
    InjectedFile f{"/etc/config.json", data, true, "sim-config"};
    EXPECT_EQ(f.container_path, "/etc/config.json");
    EXPECT_EQ(f.content.size(), 3u);
    EXPECT_EQ(f.label, "sim-config");
}

// ── PortMapping ────────────────────────────────────────────────────────────

TEST(PortMappingTest, Defaults) {
    PortMapping pm;
    EXPECT_EQ(pm.container_port, 0u);
    EXPECT_EQ(pm.host_port, 0u);
    EXPECT_EQ(pm.protocol, "tcp");
    EXPECT_TRUE(pm.label.empty());
}

TEST(PortMappingTest, Construction) {
    PortMapping pm{8080, 9090, "tcp", "metrics"};
    EXPECT_EQ(pm.container_port, 8080u);
    EXPECT_EQ(pm.host_port, 9090u);
    EXPECT_EQ(pm.protocol, "tcp");
    EXPECT_EQ(pm.label, "metrics");
}

TEST(PortMappingTest, Equality) {
    PortMapping a{80, 80, "tcp", ""};
    PortMapping b{80, 80, "tcp", ""};
    PortMapping c{443, 443, "tcp", ""};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── ContainerDef ───────────────────────────────────────────────────────────

TEST(ContainerDefTest, Defaults) {
    ContainerDef c;
    EXPECT_TRUE(c.image.empty());
    EXPECT_TRUE(c.env.empty());
    EXPECT_TRUE(c.mounts.empty());
    EXPECT_TRUE(c.injected_files.empty());
    EXPECT_TRUE(c.entrypoint.empty());
    EXPECT_TRUE(c.working_dir.empty());
    EXPECT_TRUE(c.ports.empty());
    EXPECT_FALSE(c.privileged);
    EXPECT_TRUE(c.resource_limits_json.empty());
}

TEST(ContainerDefTest, FullConstruction) {
    ContainerDef c;
    c.image = "pytorch:latest";
    c.env = {{"LD_PRELOAD", "/usr/lib/libsim.so"}};
    c.mounts = {{"/host/lib", "/lib/sim", true}};
    c.entrypoint = {"/bin/bash", "-c", "echo hello"};
    c.working_dir = "/workspace";
    c.ports = {{8080, 0, "tcp", "app"}};
    c.privileged = false;
    c.resource_limits_json = R"({"cpu": "4"})";

    EXPECT_EQ(c.image, "pytorch:latest");
    EXPECT_EQ(c.env.size(), 1u);
    EXPECT_EQ(c.mounts.size(), 1u);
    EXPECT_EQ(c.entrypoint.size(), 3u);
    EXPECT_EQ(c.ports.size(), 1u);
    EXPECT_FALSE(c.privileged);
}

TEST(ContainerDefTest, MultipleEnvVars) {
    ContainerDef c;
    c.env = {
        {"LD_PRELOAD", "/usr/lib/libsim.so"},
        {"HSA_OVERRIDE_GFX_VERSION", "11.0.0"},
        {"ROCJITSU_CONFIG", "/etc/sim.json"},
    };
    EXPECT_EQ(c.env.size(), 3u);
    EXPECT_EQ(c.env[0].key, "LD_PRELOAD");
    EXPECT_EQ(c.env[2].key, "ROCJITSU_CONFIG");
}

TEST(ContainerDefTest, MultipleMounts) {
    ContainerDef c;
    c.mounts = {
        {"/host/lib1", "/lib1", true},
        {"/host/lib2", "/lib2", false},
        {"/dev/kfd", "/dev/kfd", false},
    };
    EXPECT_EQ(c.mounts.size(), 3u);
    EXPECT_TRUE(c.mounts[0].readonly);
    EXPECT_FALSE(c.mounts[1].readonly);
}

} // namespace
} // namespace mirage
