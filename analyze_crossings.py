#!/usr/bin/env python3
"""
analyze_crossings.py - Script to analyze crosswalk crossing data

This script reads the generated crossing log files and produces
statistical summaries and reports for pedestrian and vehicle crossings.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter


def load_crossing_data(log_file):
    """Load crossing data from a JSONL log file."""
    crossings = []

    if not Path(log_file).exists():
        print(f"Warning: Log file {log_file} does not exist")
        return crossings

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Skip header lines
                if 'log_type' in data and 'created' in data:
                    continue
                crossings.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")

    return crossings


def analyze_crossings(crossings, entity_type="All"):
    """Analyze crossing data and generate statistics."""
    if not crossings:
        print(f"No {entity_type.lower()} crossing data found")
        return

    print(f"\n{'='*50}")
    print(f"{entity_type.upper()} CROSSING ANALYSIS")
    print(f"{'='*50}")

    # Basic statistics
    total_crossings = len(crossings)
    print(f"Total {entity_type.lower()} crossings: {total_crossings}")

    # Direction analysis
    directions = Counter(crossing.get('direction', 'unknown') for crossing in crossings)
    print(f"\nDirection breakdown:")
    for direction, count in directions.most_common():
        percentage = (count / total_crossings) * 100
        print(f"  {direction}: {count} ({percentage:.1f}%)")

    # Object type analysis
    object_types = Counter(crossing.get('object_type', 'unknown') for crossing in crossings)
    print(f"\nObject type breakdown:")
    for obj_type, count in object_types.most_common():
        percentage = (count / total_crossings) * 100
        print(f"  {obj_type}: {count} ({percentage:.1f}%)")

    # Duration analysis
    durations = [crossing.get('duration', 0) for crossing in crossings if crossing.get('duration')]
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"\nCrossing duration statistics:")
        print(f"  Average: {avg_duration:.2f} seconds")
        print(f"  Minimum: {min_duration:.2f} seconds")
        print(f"  Maximum: {max_duration:.2f} seconds")

    # Time analysis
    timestamps = []
    for crossing in crossings:
        if 'timestamp' in crossing:
            try:
                timestamps.append(datetime.fromisoformat(crossing['timestamp']))
            except ValueError:
                continue

    if timestamps:
        timestamps.sort()
        print(f"\nTime range:")
        print(f"  First crossing: {timestamps[0].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Last crossing: {timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}")

        # Hour analysis
        hours = Counter(ts.hour for ts in timestamps)
        print(f"\nBusiest hours:")
        for hour, count in hours.most_common(5):
            print(f"  {hour:02d}:00 - {hour:02d}:59: {count} crossings")

    # Zone path analysis
    zone_paths = Counter(
        ' → '.join(crossing.get('zone_path', []))
        for crossing in crossings
        if crossing.get('zone_path')
    )

    if zone_paths:
        print(f"\nMost common crossing paths:")
        for path, count in zone_paths.most_common(5):
            percentage = (count / total_crossings) * 100
            print(f"  {path}: {count} ({percentage:.1f}%)")


def generate_csv_report(crossings, output_file):
    """Generate a CSV report from crossing data."""
    import csv

    if not crossings:
        print("No crossing data to export")
        return

    with open(output_file, 'w', newline='') as csvfile:
        # Get all possible fieldnames
        fieldnames = set()
        for crossing in crossings:
            fieldnames.update(crossing.keys())

        # Ensure consistent ordering
        fieldnames = sorted(fieldnames)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for crossing in crossings:
            # Convert lists to strings for CSV
            row = {}
            for key, value in crossing.items():
                if isinstance(value, list):
                    row[key] = ' → '.join(map(str, value))
                else:
                    row[key] = value
            writer.writerow(row)

    print(f"CSV report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze crosswalk crossing data")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory containing log files")
    parser.add_argument("--zone-config", type=str,
                       help="Specific zone configuration to analyze (e.g., 'chapel-crosswalk')")
    parser.add_argument("--csv", action="store_true",
                       help="Generate CSV reports")
    parser.add_argument("--pedestrians-only", action="store_true",
                       help="Analyze only pedestrian crossings")
    parser.add_argument("--vehicles-only", action="store_true",
                       help="Analyze only vehicle crossings")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return

    # If zone-config specified, look in that subdirectory
    if args.zone_config:
        zone_log_dir = log_dir / args.zone_config
        if not zone_log_dir.exists():
            print(f"Error: Zone log directory {zone_log_dir} does not exist")
            return
        search_dirs = [zone_log_dir]
        print(f"Analyzing logs for zone configuration: {args.zone_config}")
    else:
        # Search all subdirectories and main directory
        search_dirs = [log_dir] + [d for d in log_dir.iterdir() if d.is_dir()]
        print(f"Analyzing logs in all directories under {log_dir}")

    # Find log files across all search directories
    pedestrian_logs = []
    vehicle_logs = []
    all_logs = []

    for search_dir in search_dirs:
        pedestrian_logs.extend(search_dir.glob("*pedestrians.jsonl"))
        vehicle_logs.extend(search_dir.glob("*vehicles.jsonl"))
        all_files = list(search_dir.glob("*all.jsonl"))
        all_logs.extend(all_files)

    print(f"\nFound log files:")
    print(f"  Pedestrian logs: {len(pedestrian_logs)}")
    print(f"  Vehicle logs: {len(vehicle_logs)}")
    print(f"  Combined logs: {len(all_logs)}")

    if pedestrian_logs:
        print(f"  Pedestrian log locations:")
        for log in pedestrian_logs:
            print(f"    {log}")

    if vehicle_logs:
        print(f"  Vehicle log locations:")
        for log in vehicle_logs:
            print(f"    {log}")

    # Load and analyze data
    if not args.vehicles_only:
        for ped_log in pedestrian_logs:
            pedestrian_data = load_crossing_data(ped_log)
            analyze_crossings(pedestrian_data, "Pedestrian")

            if args.csv:
                csv_file = ped_log.with_suffix('.csv')
                generate_csv_report(pedestrian_data, csv_file)

    if not args.pedestrians_only:
        for veh_log in vehicle_logs:
            vehicle_data = load_crossing_data(veh_log)
            analyze_crossings(vehicle_data, "Vehicle")

            if args.csv:
                csv_file = veh_log.with_suffix('.csv')
                generate_csv_report(vehicle_data, csv_file)

    # Analyze combined data if no specific filter
    if not args.pedestrians_only and not args.vehicles_only:
        for all_log in all_logs:
            all_data = load_crossing_data(all_log)
            analyze_crossings(all_data, "All")

            if args.csv:
                csv_file = all_log.with_suffix('.csv')
                generate_csv_report(all_data, csv_file)


if __name__ == "__main__":
    main()