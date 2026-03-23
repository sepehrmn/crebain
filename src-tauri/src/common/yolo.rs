//! YOLO model output helpers.
//!
//! CREBAIN currently expects Ultralytics-style YOLOv8 outputs with 84 features:
//! 4 bbox coords (cx, cy, w, h) + 80 class scores (COCO).
//!
//! Different export paths may produce either:
//! - `[1, 84, N]` (channels-first)
//! - `[1, N, 84]` (anchors-first)

/// YOLOv8 COCO output features: 4 box coords + 80 class scores.
pub const YOLOV8_OUTPUT_FEATURES: usize = 84;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputLayout {
    ChannelsFirst,
    AnchorsFirst,
}

pub fn infer_yolov8_output_layout(shape_dims: &[usize]) -> Result<(OutputLayout, usize), String> {
    if shape_dims.len() == 3 {
        if shape_dims[1] == YOLOV8_OUTPUT_FEATURES {
            Ok((OutputLayout::ChannelsFirst, shape_dims[2]))
        } else if shape_dims[2] == YOLOV8_OUTPUT_FEATURES {
            Ok((OutputLayout::AnchorsFirst, shape_dims[1]))
        } else {
            Err(format!("Unexpected output shape: {:?}", shape_dims))
        }
    } else {
        Err(format!("Unexpected output shape: {:?}", shape_dims))
    }
}

pub fn read_bbox(
    layout: OutputLayout,
    output_data: &[f32],
    num_anchors: usize,
    anchor_idx: usize,
) -> (f32, f32, f32, f32) {
    match layout {
        // Layout: [1, 84, N]
        // Index [0, j, i] = j * N + i
        OutputLayout::ChannelsFirst => (
            output_data[anchor_idx],
            output_data[num_anchors + anchor_idx],
            output_data[2 * num_anchors + anchor_idx],
            output_data[3 * num_anchors + anchor_idx],
        ),
        // Layout: [1, N, 84]
        // Index [0, i, j] = i * 84 + j
        OutputLayout::AnchorsFirst => {
            let base = anchor_idx * YOLOV8_OUTPUT_FEATURES;
            (
                output_data[base],
                output_data[base + 1],
                output_data[base + 2],
                output_data[base + 3],
            )
        }
    }
}

pub fn read_class_score(
    layout: OutputLayout,
    output_data: &[f32],
    num_anchors: usize,
    anchor_idx: usize,
    class_idx: usize,
) -> f32 {
    match layout {
        OutputLayout::ChannelsFirst => output_data[(4 + class_idx) * num_anchors + anchor_idx],
        OutputLayout::AnchorsFirst => output_data[anchor_idx * YOLOV8_OUTPUT_FEATURES + 4 + class_idx],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolov8_layout_channels_first_indexes_correctly() {
        let num_anchors = 2usize;
        let shape = [1, YOLOV8_OUTPUT_FEATURES, num_anchors];
        let (layout, anchors) = infer_yolov8_output_layout(&shape).unwrap();
        assert_eq!(layout, OutputLayout::ChannelsFirst);
        assert_eq!(anchors, num_anchors);

        let mut data = vec![0.0f32; YOLOV8_OUTPUT_FEATURES * num_anchors];
        // bbox for anchor 1
        data[1] = 11.0;
        data[num_anchors + 1] = 21.0;
        data[2 * num_anchors + 1] = 31.0;
        data[3 * num_anchors + 1] = 41.0;
        // class score (class 5) for anchor 0
        data[(4 + 5) * num_anchors] = 0.9;

        let (cx, cy, w, h) = read_bbox(layout, &data, anchors, 1);
        assert_eq!((cx, cy, w, h), (11.0, 21.0, 31.0, 41.0));
        assert_eq!(read_class_score(layout, &data, anchors, 0, 5), 0.9);
    }

    #[test]
    fn yolov8_layout_anchors_first_indexes_correctly() {
        let num_anchors = 2usize;
        let shape = [1, num_anchors, YOLOV8_OUTPUT_FEATURES];
        let (layout, anchors) = infer_yolov8_output_layout(&shape).unwrap();
        assert_eq!(layout, OutputLayout::AnchorsFirst);
        assert_eq!(anchors, num_anchors);

        let mut data = vec![0.0f32; YOLOV8_OUTPUT_FEATURES * num_anchors];
        // bbox for anchor 1
        let base = YOLOV8_OUTPUT_FEATURES;
        data[base] = 11.0;
        data[base + 1] = 21.0;
        data[base + 2] = 31.0;
        data[base + 3] = 41.0;
        // class score (class 5) for anchor 0
        data[4 + 5] = 0.9;

        let (cx, cy, w, h) = read_bbox(layout, &data, anchors, 1);
        assert_eq!((cx, cy, w, h), (11.0, 21.0, 31.0, 41.0));
        assert_eq!(read_class_score(layout, &data, anchors, 0, 5), 0.9);
    }

    #[test]
    fn yolov8_layout_rejects_unexpected_shapes() {
        assert!(infer_yolov8_output_layout(&[1, 85, 8400]).is_err());
        assert!(infer_yolov8_output_layout(&[1, 8400]).is_err());
    }
}
