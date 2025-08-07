import os
import tempfile
import re
import pytest

from scripts.rename_files import rename_images_labels  # Replace with actual import

def test_rename_images_labels():
    with tempfile.TemporaryDirectory() as temp_img_dir, tempfile.TemporaryDirectory() as temp_lbl_dir:
        # Create image files with complex names
        img_names = [
            "image1_jpg123.jpg",
            "image2_jpg456.jpg",
            "image3.png",
            "not_an_image.txt",  # Should be ignored
        ]

        # Create label files corresponding to images
        lbl_names = [
            "image1_jpg123.txt",
            "image2_jpg456.txt",
            "image3.txt",
        ]

        # Create files in temp dirs
        for name in img_names:
            open(os.path.join(temp_img_dir, name), 'w').close()
        for name in lbl_names:
            open(os.path.join(temp_lbl_dir, name), 'w').close()

        # Run your renaming function
        rename_images_labels(temp_img_dir, temp_lbl_dir)

        # List resulting files
        renamed_images = sorted(os.listdir(temp_img_dir))
        renamed_labels = sorted(os.listdir(temp_lbl_dir))

        # Assert images were renamed correctly (suffix _jpg* removed)
        assert "image1.jpg" in renamed_images
        assert "image2.jpg" in renamed_images
        assert "image3.png" in renamed_images
        assert "not_an_image.txt" in renamed_images  # Not renamed

        # Assert labels renamed accordingly
        assert "image1.txt" in renamed_labels
        assert "image2.txt" in renamed_labels
        assert "image3.txt" in renamed_labels

        # Ensure old label names are gone
        assert "image1_jpg123.txt" not in renamed_labels
        assert "image2_jpg456.txt" not in renamed_labels
