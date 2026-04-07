// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @brief Comprehensive V4 schema validation - one test per table
 * Each test validates: all columns, keys, V4-specific features, and absence of V3-only
 * features
 */

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <rocpdsna/storage.hpp>
#include <rocpdsna/writer.hpp>
#include <rocpdsna/writer_types.hpp>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <vector>
#define VERSION                                                                          \
    rocpdsna::version_t { 4, 0, 0 }
class SchemaV4ComprehensiveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = "test_v4_comprehensive.db";
        m_uuid    = "v4comp";
        std::filesystem::remove(m_db_path);

        // Create V4 database
        auto storage = std::make_unique<rocpdsna::storage_t>(m_db_path, m_uuid, VERSION);
        auto writer  = std::make_unique<rocpdsna::writer_t>(std::move(storage));

        // Write minimal data to initialize schema
        rocpdsna::writer_types::node_info_t node;
        node.hash          = 1;
        node.machine_id    = "test";
        node.hostname      = "test";
        node.system_name   = "test";
        node.release       = "1.0";
        node.version       = "1.0";
        node.hardware_name = "test";
        node.domain_name   = "test";
        writer->register_node_info(node);
        writer->flush_in_memory_data_to_disk();
        writer.reset();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    sqlite3* open_db()
    {
        sqlite3* db = nullptr;
        int      rc = sqlite3_open(m_db_path.c_str(), &db);
        return (rc == SQLITE_OK) ? db : nullptr;
    }

    bool table_exists(sqlite3* db, const std::string& table_name)
    {
        std::string query =
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table_name.c_str(), -1, SQLITE_STATIC);
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return exists;
    }

    bool view_exists(sqlite3* db, const std::string& table_name)
    {
        std::string query = "SELECT name FROM sqlite_master WHERE type='view' AND name=?";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table_name.c_str(), -1, SQLITE_STATIC);
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return exists;
    }

    bool column_exists(sqlite3*           db,
                       const std::string& table_name,
                       const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        bool found = false;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                found = true;
                break;
            }
        }
        sqlite3_finalize(stmt);
        return found;
    }

    std::string m_db_path;
    std::string m_uuid;
};

// Test 1: rocpd_metadata table
TEST_F(SchemaV4ComprehensiveTest, metadata_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_metadata_" + m_uuid;
    // if(!table_exists(db, table_name))
    // {
    //     table_name = "rocpd_metadata";
    // }
    ASSERT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "tag"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));

    sqlite3_close(db);
}

// Test 2: rocpd_string table
TEST_F(SchemaV4ComprehensiveTest, string_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_string_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "string"));

    sqlite3_close(db);
}

// Test 3: rocpd_info_node table
TEST_F(SchemaV4ComprehensiveTest, info_node_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_node_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "hash"));
    EXPECT_TRUE(column_exists(db, table_name, "machine_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "system_name"));
    EXPECT_TRUE(column_exists(db, table_name, "hostname"));
    EXPECT_TRUE(column_exists(db, table_name, "release"));
    EXPECT_TRUE(column_exists(db, table_name, "version"));
    EXPECT_TRUE(column_exists(db, table_name, "hardware_name"));
    EXPECT_TRUE(column_exists(db, table_name, "domain_name"));

    sqlite3_close(db);
}

// Test 4: rocpd_info_process table
TEST_F(SchemaV4ComprehensiveTest, info_process_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_process_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "init"));
    EXPECT_TRUE(column_exists(db, table_name, "fini"));
    EXPECT_TRUE(column_exists(db, table_name, "start"));
    EXPECT_TRUE(column_exists(db, table_name, "end"));
    EXPECT_TRUE(column_exists(db, table_name, "command"));
    EXPECT_TRUE(column_exists(db, table_name, "environment"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 5: rocpd_info_thread table
TEST_F(SchemaV4ComprehensiveTest, info_thread_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_thread_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "tid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "start"));
    EXPECT_TRUE(column_exists(db, table_name, "end"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 6: rocpd_info_category table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_category_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_category_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 7: rocpd_info_agent table
TEST_F(SchemaV4ComprehensiveTest, info_agent_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_agent_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "absolute_index"));
    EXPECT_TRUE(column_exists(db, table_name, "logical_index"));
    EXPECT_TRUE(column_exists(db, table_name, "type_index"));
    EXPECT_TRUE(column_exists(db, table_name, "uuid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "generic_name"));  // V4: replaces user_name
    EXPECT_TRUE(column_exists(db, table_name, "model_name"));
    EXPECT_TRUE(column_exists(db, table_name, "vendor_name"));
    EXPECT_TRUE(column_exists(db, table_name, "product_name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had "user_name", V4 doesn't
    EXPECT_FALSE(column_exists(db, table_name, "user_name"));

    sqlite3_close(db);
}

// Test 8: rocpd_info_queue table
TEST_F(SchemaV4ComprehensiveTest, info_queue_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_queue_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 9: rocpd_info_stream table
TEST_F(SchemaV4ComprehensiveTest, info_stream_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_stream_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 10: rocpd_info_pmc table
TEST_F(SchemaV4ComprehensiveTest, info_pmc_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_pmc_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "target_arch"));
    EXPECT_TRUE(column_exists(db, table_name, "event_code"));
    EXPECT_TRUE(column_exists(db, table_name, "instance_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "symbol"));
    EXPECT_TRUE(column_exists(db, table_name, "qualifier"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "description"));
    EXPECT_TRUE(column_exists(db, table_name, "long_description"));
    EXPECT_TRUE(column_exists(db, table_name, "component"));
    EXPECT_TRUE(column_exists(db, table_name, "units"));
    EXPECT_TRUE(column_exists(db, table_name, "value_type"));
    EXPECT_TRUE(column_exists(db, table_name, "block"));
    EXPECT_TRUE(column_exists(db, table_name, "expression"));
    EXPECT_TRUE(column_exists(db, table_name, "is_constant"));
    EXPECT_TRUE(column_exists(db, table_name, "is_derived"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 11: rocpd_info_code_object table
TEST_F(SchemaV4ComprehensiveTest, info_code_object_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_code_object_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "uri"));
    EXPECT_TRUE(column_exists(db, table_name, "load_base"));
    EXPECT_TRUE(column_exists(db, table_name, "load_size"));
    EXPECT_TRUE(column_exists(db, table_name, "load_delta"));
    EXPECT_TRUE(column_exists(db, table_name, "storage_type"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 12: rocpd_info_kernel_symbol table
TEST_F(SchemaV4ComprehensiveTest, info_kernel_symbol_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_kernel_symbol_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "code_object_id"));
    EXPECT_TRUE(column_exists(db, table_name, "kernel_name"));
    EXPECT_TRUE(column_exists(db, table_name, "display_name"));
    EXPECT_TRUE(column_exists(db, table_name, "kernel_object"));
    EXPECT_TRUE(column_exists(db, table_name, "kernarg_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "kernarg_segment_alignment"));
    EXPECT_TRUE(column_exists(db, table_name, "group_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "private_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "sgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "arch_vgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "accum_vgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 13: rocpd_info_address_range table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_address_range_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_address_range_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "address_base"));
    EXPECT_TRUE(column_exists(db, table_name, "address_low"));
    EXPECT_TRUE(column_exists(db, table_name, "address_high"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 14: rocpd_info_source_code table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_source_code_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_source_code_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "address_id"));
    EXPECT_TRUE(column_exists(db, table_name, "file"));
    EXPECT_TRUE(column_exists(db, table_name, "line_number"));
    EXPECT_TRUE(column_exists(db, table_name, "lines"));
    EXPECT_TRUE(column_exists(db, table_name, "instructions"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 15: rocpd_info_pc table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_pc_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_pc_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "function"));
    EXPECT_TRUE(column_exists(db, table_name, "address_id"));
    EXPECT_TRUE(column_exists(db, table_name, "file"));
    EXPECT_TRUE(column_exists(db, table_name, "line"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 16: rocpd_track table (V4 enhanced)
TEST_F(SchemaV4ComprehensiveTest, track_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_track_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (enhanced from V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "tid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "queue_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "stream_id"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 17: rocpd_timestamp table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, timestamp_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_timestamp_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 for timestamp normalization)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "phase"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));

    sqlite3_close(db);
}

// Test 18: rocpd_event table
TEST_F(SchemaV4ComprehensiveTest, event_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_event_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "category_id"));
    EXPECT_TRUE(column_exists(db, table_name, "stack_id"));
    EXPECT_TRUE(column_exists(db, table_name, "parent_stack_id"));
    EXPECT_TRUE(column_exists(db, table_name, "correlation_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had embedded call_stack and line_info JSONB, V4 doesn't
    EXPECT_FALSE(column_exists(db, table_name, "call_stack"));
    EXPECT_FALSE(column_exists(db, table_name, "line_info"));

    sqlite3_close(db);
}

// Test 19: rocpd_arg table
TEST_F(SchemaV4ComprehensiveTest, arg_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_arg_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "position"));
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 20: rocpd_line_info table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, line_info_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_line_info_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 - normalized from V3's embedded JSONB)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "source_code_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 21: rocpd_call_stack table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, call_stack_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_call_stack_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 - normalized from V3's embedded JSONB)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "depth"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 22: rocpd_pmc_event table
TEST_F(SchemaV4ComprehensiveTest, pmc_event_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_pmc_event_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pmc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 23: rocpd_region table
TEST_F(SchemaV4ComprehensiveTest, region_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_region_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized with foreign keys
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(
        column_exists(db, table_name, "start_id"));        // V4: foreign key to timestamp
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));  // V4: foreign key to timestamp
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and nid/pid/tid
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));

    sqlite3_close(db);
}

// Test 24: rocpd_sample table
TEST_F(SchemaV4ComprehensiveTest, sample_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_sample_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));       // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "timestamp_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamp
    EXPECT_FALSE(column_exists(db, table_name, "timestamp"));

    sqlite3_close(db);
}

// Test 25: rocpd_kernel_dispatch table
TEST_F(SchemaV4ComprehensiveTest, kernel_dispatch_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_kernel_dispatch_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "kernel_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dispatch_id"));
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "private_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "group_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_x"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_y"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_z"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_x"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_y"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_z"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "agent_id"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}

// Test 26: rocpd_memory_copy table
TEST_F(SchemaV4ComprehensiveTest, memory_copy_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_memory_copy_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dst_agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dst_address"));
    EXPECT_TRUE(column_exists(db, table_name, "src_agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "src_address"));
    EXPECT_TRUE(column_exists(db, table_name, "size"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}

// Test 27: rocpd_memory_allocate table
TEST_F(SchemaV4ComprehensiveTest, memory_allocate_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_memory_allocate_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "level"));
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "address"));
    EXPECT_TRUE(column_exists(db, table_name, "size"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "agent_id"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}