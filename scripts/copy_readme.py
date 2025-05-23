import sys

def copy_markdown_lines(source_file, target_file, start_line=10):
    try:
        # Read source file
        with open(source_file, 'r', encoding='utf-8') as src:
            lines = src.readlines()

        # Check if source file has enough lines
        if len(lines) < start_line:
            print(f"Error: Source file '{source_file}' has fewer than {start_line} lines.")
            sys.exit(1)

        # Get content from start_line onward (0-based index)
        content_to_copy = lines[start_line - 1:]

        # Write to target file
        with open(target_file, 'w', encoding='utf-8') as tgt:
            tgt.writelines(content_to_copy)

        print(f"Successfully copied content from '{source_file}' (line {start_line} onward) to '{target_file}'.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":

    source_file = "../README.md"  # Path to the source file
    target_file = "../doxyfiles/doxyfile_readme.md"  # Path to the target file
    copy_markdown_lines(source_file, target_file)