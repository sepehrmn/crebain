//! COCO Dataset class labels.
//!
//! The COCO (Common Objects in Context) dataset defines 80 object classes.
//! This module provides the canonical class list used by YOLOv8 and other
//! COCO-trained models.

/// COCO dataset class labels (80 classes).
///
/// These labels match the output indices from YOLOv8 and other
/// models trained on the COCO dataset.
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Number of COCO classes.
pub const NUM_CLASSES: usize = 80;

/// Get the class name for a given class index.
///
/// Returns "unknown" for out-of-range indices.
#[inline]
pub fn get_class_name(index: usize) -> String {
    COCO_CLASSES
        .get(index)
        .map(|s| (*s).to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get the class name as a static str reference.
///
/// Returns None for out-of-range indices.
#[inline]
pub fn get_class_name_ref(index: usize) -> Option<&'static str> {
    COCO_CLASSES.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_count() {
        assert_eq!(COCO_CLASSES.len(), NUM_CLASSES);
        assert_eq!(COCO_CLASSES.len(), 80);
    }

    #[test]
    fn test_get_class_name() {
        assert_eq!(get_class_name(0), "person");
        assert_eq!(get_class_name(79), "toothbrush");
        assert_eq!(get_class_name(80), "unknown");
        assert_eq!(get_class_name(1000), "unknown");
    }

    #[test]
    fn test_get_class_name_ref() {
        assert_eq!(get_class_name_ref(0), Some("person"));
        assert_eq!(get_class_name_ref(79), Some("toothbrush"));
        assert_eq!(get_class_name_ref(80), None);
    }
}
