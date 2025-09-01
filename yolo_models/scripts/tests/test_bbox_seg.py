
2), str(output_dir))

    # Check if output file exists
    output_file = output_dir / "sample.txt"
    assert output_file.exists(), "Output file not created."

    # Check content of the output file
    with open(output_file, "r") as f:
        lines = f.readlines()

    parts = lines[0].strip().split() #whitespace
    assert parts[0] == "0"
    assert len(parts[1:]) == 8  # 4 points × 2 coordinates

    # Check that all coords are in range [0.0, 1.0]
    coords = list(map(float, parts[1:]))
    assert all(0.0 <= c <= 1.0 for c in coords), "Coords not in [0.0, 1.0]"
