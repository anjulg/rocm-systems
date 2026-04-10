#include "mirage/time_utils.h"

#include <gtest/gtest.h>

namespace mirage {
namespace {

// ── time_add ───────────────────────────────────────────────────────────────

TEST(TimeAddTest, SimpleAdd) {
    Time a{1, 500};
    Time b{2, 300};
    auto r = time_add(a, b);
    EXPECT_EQ(r.seconds, 3u);
    EXPECT_EQ(r.picoseconds, 800u);
}

TEST(TimeAddTest, CarryPicoseconds) {
    Time a{0, kPicosecondsPerSecond - 1};
    Time b{0, 2};
    auto r = time_add(a, b);
    EXPECT_EQ(r.seconds, 1u);
    EXPECT_EQ(r.picoseconds, 1u);
}

TEST(TimeAddTest, LargeCarry) {
    Time a{0, kPicosecondsPerSecond * 3 + 42};
    Time b{0, kPicosecondsPerSecond * 2 + 58};
    auto r = time_add(a, b);
    // 3+2 = 5 seconds carry, 42+58=100 ps
    // But a and b already have overflow in ps fields, so time_add
    // handles this correctly
    EXPECT_EQ(r.seconds, 5u);
    EXPECT_EQ(r.picoseconds, 100u);
}

TEST(TimeAddTest, BothZero) {
    auto r = time_add({0, 0}, {0, 0});
    EXPECT_EQ(r.seconds, 0u);
    EXPECT_EQ(r.picoseconds, 0u);
}

TEST(TimeAddTest, AddZero) {
    Time a{5, 1000};
    auto r = time_add(a, {0, 0});
    EXPECT_EQ(r, a);
}

// ── time_sub ───────────────────────────────────────────────────────────────

TEST(TimeSubTest, SimpleSub) {
    Time a{5, 300};
    Time b{2, 100};
    auto r = time_sub(a, b);
    EXPECT_EQ(r.seconds, 3u);
    EXPECT_EQ(r.picoseconds, 200u);
}

TEST(TimeSubTest, BorrowFromSeconds) {
    Time a{5, 100};
    Time b{2, 300};
    auto r = time_sub(a, b);
    EXPECT_EQ(r.seconds, 2u);
    EXPECT_EQ(r.picoseconds, kPicosecondsPerSecond - 200);
}

TEST(TimeSubTest, SubtractSelf) {
    Time a{42, 12345};
    auto r = time_sub(a, a);
    EXPECT_EQ(r.seconds, 0u);
    EXPECT_EQ(r.picoseconds, 0u);
}

TEST(TimeSubTest, SubtractZero) {
    Time a{10, 500};
    auto r = time_sub(a, {0, 0});
    EXPECT_EQ(r, a);
}

// ── time_to_nanoseconds ────────────────────────────────────────────────────

TEST(TimeToNsTest, Zero) {
    EXPECT_EQ(time_to_nanoseconds({0, 0}), 0u);
}

TEST(TimeToNsTest, SecondsOnly) {
    EXPECT_EQ(time_to_nanoseconds({1, 0}), 1'000'000'000u);
}

TEST(TimeToNsTest, PicosecondsOnly) {
    // 5000 picoseconds = 5 nanoseconds
    EXPECT_EQ(time_to_nanoseconds({0, 5000}), 5u);
}

TEST(TimeToNsTest, Combined) {
    // 2 seconds + 3000 ps = 2,000,000,003 ns
    EXPECT_EQ(time_to_nanoseconds({2, 3000}), 2'000'000'003u);
}

TEST(TimeToNsTest, SubNsTruncated) {
    // 500 ps < 1 ns → truncated to 0
    EXPECT_EQ(time_to_nanoseconds({0, 500}), 0u);
}

// ── time_from_nanoseconds ──────────────────────────────────────────────────

TEST(TimeFromNsTest, Zero) {
    auto t = time_from_nanoseconds(0);
    EXPECT_EQ(t.seconds, 0u);
    EXPECT_EQ(t.picoseconds, 0u);
}

TEST(TimeFromNsTest, ExactSecond) {
    auto t = time_from_nanoseconds(1'000'000'000);
    EXPECT_EQ(t.seconds, 1u);
    EXPECT_EQ(t.picoseconds, 0u);
}

TEST(TimeFromNsTest, WithRemainder) {
    auto t = time_from_nanoseconds(2'500'000'042);
    EXPECT_EQ(t.seconds, 2u);
    EXPECT_EQ(t.picoseconds, 500'000'042'000u);
}

TEST(TimeFromNsTest, RoundTrip) {
    uint64_t original = 123'456'789'012;
    auto t = time_from_nanoseconds(original);
    EXPECT_EQ(time_to_nanoseconds(t), original);
}

// ── time_from_seconds ──────────────────────────────────────────────────────

TEST(TimeFromSecondsTest, Basic) {
    auto t = time_from_seconds(42);
    EXPECT_EQ(t.seconds, 42u);
    EXPECT_EQ(t.picoseconds, 0u);
}

// ── time_from_parts ────────────────────────────────────────────────────────

TEST(TimeFromPartsTest, NormalValues) {
    auto t = time_from_parts(1, 500);
    EXPECT_EQ(t.seconds, 1u);
    EXPECT_EQ(t.picoseconds, 500u);
}

TEST(TimeFromPartsTest, PicosecondOverflow) {
    auto t = time_from_parts(0, kPicosecondsPerSecond * 3 + 42);
    EXPECT_EQ(t.seconds, 3u);
    EXPECT_EQ(t.picoseconds, 42u);
}

TEST(TimeFromPartsTest, BothOverflowAndBase) {
    auto t = time_from_parts(10, kPicosecondsPerSecond + 1);
    EXPECT_EQ(t.seconds, 11u);
    EXPECT_EQ(t.picoseconds, 1u);
}

// ── time_is_zero ───────────────────────────────────────────────────────────

TEST(TimeIsZeroTest, Zero) {
    EXPECT_TRUE(time_is_zero({0, 0}));
}

TEST(TimeIsZeroTest, NonZeroSeconds) {
    EXPECT_FALSE(time_is_zero({1, 0}));
}

TEST(TimeIsZeroTest, NonZeroPicoseconds) {
    EXPECT_FALSE(time_is_zero({0, 1}));
}

// ── Constants ──────────────────────────────────────────────────────────────

TEST(ConstantsTest, PicosecondsPerSecond) {
    EXPECT_EQ(kPicosecondsPerSecond, 1'000'000'000'000ULL);
}

TEST(ConstantsTest, PicosecondsPerNanosecond) {
    EXPECT_EQ(kPicosecondsPerNanosecond, 1'000ULL);
}

TEST(ConstantsTest, Consistency) {
    EXPECT_EQ(kPicosecondsPerSecond,
              1'000'000'000ULL * kPicosecondsPerNanosecond);
}

} // namespace
} // namespace mirage
