from scripts.training import train_objects_model, train_seame_model
from unittest.mock import patch

def test_train_objects_model():
    with patch("scripts.training.YOLO") as MockYOLO:
        instance = MockYOLO.return_value
        instance.train.return_value = "train_success"
        result = train_objects_model()
        instance.train.assert_called_once()
        assert result == "train_success"

def test_train_seame_model():
    with patch("scripts.training.YOLO") as MockYOLO:
        instance = MockYOLO.return_value
        instance.train.return_value = "train_success"
        result = train_seame_model()
        instance.train.assert_called_once()
        assert result == "train_success"
