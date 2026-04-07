# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import sqlite3
from contextlib import ExitStack, closing

from utils.logger import console_error, console_warning

# From schema definition in source/share/rocprofiler-sdk-rocpd/data_views.sql
# in rocprofiler-sdk repository
COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
    guid as GUID,
    stack_id as Correlation_Id,
    dispatch_id as Dispatch_ID,
    pid as PID,
    grid_size as Grid_Size,
    workgroup_size as Workgroup_Size,
    lds_block_size as LDS_Per_Workgroup,
    scratch_size as Scratch_Per_Workitem,
    vgpr_count as Arch_VGPR,
    accum_vgpr_count as Accum_VGPR,
    sgpr_count as SGPR,
    kernel_name as Kernel_Name,
    start as Start_Timestamp,
    end as End_Timestamp,
    kernel_id as Kernel_ID,
    counter_name as Counter_Name,
    value as Counter_Value
FROM counters_collection
"""
MARKER_API_TRACE_QUERY = """
SELECT
    category AS Domain,
    json_extract(extdata, '$.message') AS Function,
    pid AS Process_Id,
    tid AS Thread_Id,
    stack_id AS Correlation_Id,
    guid AS GUID,
    start AS Start_Timestamp,
    end AS End_Timestamp
FROM regions
ORDER BY start
"""
KERNEL_DISPATCH_QUERY = """
SELECT dispatch_id, event_id, guid
FROM rocpd_kernel_dispatch
WHERE guid = ?
"""
ROCPD_PMC_EVENT_TABLE_NAME_PREFIX = "rocpd_pmc_event_"
TABLE_NAME_PREFIX_QUERY = (
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name LIKE '{table_name_prefix}%'"
)
INSERT_QUERY = "INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
STAGING_TABLE_NAME = "temp_rocpd_pmc_stage"
DISPATCH_MAP_TABLE_NAME = "temp_rocpd_dispatch_event_map"
STAGING_INSERT_QUERY = (
    f"INSERT INTO {STAGING_TABLE_NAME} (dispatch_id, pmc_id, value) VALUES (?, ?, ?)"
)
STAGING_CLEAR_QUERY = f"DELETE FROM {STAGING_TABLE_NAME}"
STAGING_CREATE_QUERY = f"""
CREATE TEMP TABLE IF NOT EXISTS {STAGING_TABLE_NAME} (
    dispatch_id TEXT,
    pmc_id TEXT,
    value TEXT
)
"""
DISPATCH_MAP_CREATE_QUERY = f"""
CREATE TEMP TABLE IF NOT EXISTS {DISPATCH_MAP_TABLE_NAME} (
    dispatch_id TEXT PRIMARY KEY,
    event_id INTEGER NOT NULL
)
"""
DISPATCH_MAP_LOAD_QUERY = (
    f"INSERT INTO {DISPATCH_MAP_TABLE_NAME} (dispatch_id, event_id) "
    "SELECT CAST(dispatch_id AS TEXT), event_id "
    "FROM rocpd_kernel_dispatch WHERE guid = ?"
)
STAGING_TO_PMC_QUERY = """
INSERT INTO {table_name} (event_id, pmc_id, value)
SELECT
    M.event_id,
    CAST(S.pmc_id AS INTEGER),
    CAST(S.value AS REAL)
FROM
    {staging_table_name} S
    INNER JOIN {dispatch_map_table_name} M ON M.dispatch_id = S.dispatch_id
"""
UNMATCHED_STAGE_ROWS_QUERY = """
SELECT COUNT(*)
FROM
    {staging_table_name} S
    LEFT JOIN {dispatch_map_table_name} M ON M.dispatch_id = S.dispatch_id
WHERE
    M.event_id IS NULL
"""


def convert_dbs_to_csv(
    db_paths: list[str],
    counter_collection_csv_path: str,
    marker_trace_csv_path: str,
) -> None:
    queries = {
        counter_collection_csv_path: COUNTERS_COLLECTION_QUERY,
        marker_trace_csv_path: MARKER_API_TRACE_QUERY,
    }
    header_written = {path: False for path in queries}

    with ExitStack() as stack:
        writers = {
            path: csv.writer(stack.enter_context(open(path, "w", newline="")))
            for path in queries
        }
        for db_path in db_paths:
            with closing(sqlite3.connect(db_path)) as conn:
                for file_path, query in queries.items():
                    try:
                        with closing(conn.execute(query)) as cursor:
                            if cursor.description is None:
                                continue
                            if not header_written[file_path]:
                                writers[file_path].writerow([
                                    desc[0] for desc in cursor.description
                                ])
                                header_written[file_path] = True
                            writers[file_path].writerows(cursor)
                    except OSError as e:
                        console_error(
                            f"Database error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )
                    except Exception as e:
                        console_error(
                            f"Unexpected error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )


def _flush_staged_pmc_rows(conn: sqlite3.Connection, table_name: str) -> int:
    with closing(
        conn.execute(
            UNMATCHED_STAGE_ROWS_QUERY.format(
                staging_table_name=STAGING_TABLE_NAME,
                dispatch_map_table_name=DISPATCH_MAP_TABLE_NAME,
            )
        )
    ) as cursor:
        unmatched_rows = cursor.fetchone()
    conn.execute(
        STAGING_TO_PMC_QUERY.format(
            table_name=table_name,
            staging_table_name=STAGING_TABLE_NAME,
            dispatch_map_table_name=DISPATCH_MAP_TABLE_NAME,
        )
    )
    conn.execute(STAGING_CLEAR_QUERY)
    return int(unmatched_rows[0]) if unmatched_rows else 0


def update_rocpd_pmc_events(
    counter_csv_path: str, rocpd_db_path: str, chunk_size: int = 50000
) -> None:
    """Updates pmc_event table in the given rocpd database path."""
    try:
        with closing(sqlite3.connect(rocpd_db_path)) as conn:
            # Get pmc_event table name
            with closing(
                conn.execute(
                    TABLE_NAME_PREFIX_QUERY.format(
                        table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
                    )
                )
            ) as cursor:
                table_name = cursor.fetchone()
            if table_name is None:
                console_error("No pmc_event table found in the rocpd database")
            table_name = table_name[0]

            guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace(
                "_", "-"
            )
            with conn:
                conn.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE_NAME}")
                conn.execute(f"DROP TABLE IF EXISTS {DISPATCH_MAP_TABLE_NAME}")
                conn.execute(STAGING_CREATE_QUERY)
                conn.execute(DISPATCH_MAP_CREATE_QUERY)
                conn.execute(DISPATCH_MAP_LOAD_QUERY, (guid,))

                with closing(
                    conn.execute(f"SELECT COUNT(*) FROM {DISPATCH_MAP_TABLE_NAME}")
                ) as cursor:
                    dispatch_rows = cursor.fetchone()
                if not dispatch_rows or dispatch_rows[0] == 0:
                    console_error("No kernel dispatch data found.")
                    return

                unmatched_rows = 0
                staged_rows: list[tuple[str | None, str | None, str | None]] = []

                with open(counter_csv_path, newline="") as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        staged_rows.append(
                            (
                                row.get("dispatch_id"),
                                row.get("counter_id"),
                                row.get("counter_value"),
                            )
                        )
                        if len(staged_rows) >= chunk_size:
                            conn.executemany(STAGING_INSERT_QUERY, staged_rows)
                            unmatched_rows += _flush_staged_pmc_rows(conn, table_name)
                            staged_rows.clear()

                if staged_rows:
                    conn.executemany(STAGING_INSERT_QUERY, staged_rows)
                    unmatched_rows += _flush_staged_pmc_rows(conn, table_name)

                conn.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE_NAME}")
                conn.execute(f"DROP TABLE IF EXISTS {DISPATCH_MAP_TABLE_NAME}")

                if unmatched_rows:
                    console_warning(
                        f"Skipped {unmatched_rows} counter rows with no matching dispatch_id."
                    )
    except FileNotFoundError as e:
        console_error(f"Counter CSV not found while updating pmc_event table: {e}")
    except OSError as e:
        console_error(f"Database error while updating pmc_event table: {e}")
    except Exception as e:
        console_error(f"Unexpected error updating pmc_event table: {e}")
