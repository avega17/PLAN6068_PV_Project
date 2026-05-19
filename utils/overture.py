"""Shared Overture-building helpers for PLAN6068 notebooks."""

from __future__ import annotations

DEFAULT_OVERTURE_BUILDINGS_TABLE = "pr_overture_buildings"


def sql_quote(value: str) -> str:
    """Escape a Python string for safe inline SQL literal usage."""

    return "'" + str(value).replace("'", "''") + "'"


def occupied_h3_cells_sql(buildings_table: str = DEFAULT_OVERTURE_BUILDINGS_TABLE) -> str:
    """Return SQL that derives occupied H3 cells from the base Overture table."""

    return f"""
        WITH municipality_counts AS (
            SELECT
                h3_cell_id,
                h3_cell_uint,
                h3_resolution,
                municipality_name,
                municipality_geoid,
                COUNT(*)::INTEGER AS municipality_building_count
            FROM {buildings_table}
            WHERE geometry IS NOT NULL
              AND h3_cell_id IS NOT NULL
              AND h3_cell_uint IS NOT NULL
            GROUP BY ALL
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY h3_cell_id
                    ORDER BY municipality_building_count DESC, municipality_name
                ) AS municipality_rank
            FROM municipality_counts
        ),
        totals AS (
            SELECT
                h3_cell_id,
                MIN(h3_cell_uint) AS h3_cell_uint,
                MIN(h3_resolution) AS h3_resolution,
                COUNT(*)::INTEGER AS building_count
            FROM {buildings_table}
            WHERE geometry IS NOT NULL
              AND h3_cell_id IS NOT NULL
              AND h3_cell_uint IS NOT NULL
            GROUP BY h3_cell_id
        )
        SELECT
            CAST(r.h3_cell_id AS VARCHAR) AS h3_cell_id,
            CAST(r.h3_cell_uint AS UBIGINT) AS h3_cell_uint,
            CAST(r.h3_resolution AS INTEGER) AS h3_resolution,
            CAST(r.municipality_name AS VARCHAR) AS municipality_name,
            CAST(r.municipality_geoid AS VARCHAR) AS municipality_geoid,
            CAST(t.building_count AS INTEGER) AS building_count,
            CAST(r.municipality_building_count AS INTEGER) AS municipality_building_count,
            CAST(t.building_count > r.municipality_building_count AS BOOLEAN) AS crosses_municipality_boundary,
            CAST(h3_cell_to_lng(r.h3_cell_uint) AS DOUBLE) AS cell_center_lon,
            CAST(h3_cell_to_lat(r.h3_cell_uint) AS DOUBLE) AS cell_center_lat,
            ST_GeomFromWKB(h3_cell_to_boundary_wkb(r.h3_cell_uint)) AS geometry
        FROM ranked AS r
        JOIN totals AS t USING (h3_cell_id)
        WHERE r.municipality_rank = 1
    """
