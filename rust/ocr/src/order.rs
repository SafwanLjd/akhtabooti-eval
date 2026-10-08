//! Output lines built from recognised boxes.

use ocr_rs::OcrResult_;
use serde::Serialize;

#[derive(Debug, PartialEq, Serialize)]
pub struct Line {
    pub text: String,
    pub score: f32,
}

/// One line per box, in the order the engine returned them.
pub fn engine_lines(items: Vec<OcrResult_>) -> Vec<Line> {
    items
        .into_iter()
        .map(|item| Line {
            text: item.text,
            score: item.confidence,
        })
        .collect()
}

/// One line per visual row, rows from top to bottom.
///
/// A box joins the current row when its vertical centre lies above the bottom of the row's first box. Boxes in a
/// row are joined left to right with a space, so a right-to-left row comes out in visual order. The row score is
/// the mean box confidence.
pub fn reading_lines(mut items: Vec<OcrResult_>) -> Vec<Line> {
    items.sort_by_key(|item| (item.bbox.rect.top(), item.bbox.rect.left()));
    let mut rows: Vec<Vec<OcrResult_>> = Vec::new();
    for item in items {
        match rows.last_mut() {
            Some(row) if same_row(&row[0], &item) => row.push(item),
            _ => rows.push(vec![item]),
        }
    }
    rows.into_iter().map(merge_row).collect()
}

fn same_row(first: &OcrResult_, item: &OcrResult_) -> bool {
    let row = first.bbox.rect;
    let centre = item.bbox.rect.top() + item.bbox.rect.height() as i32 / 2;
    centre < row.top() + row.height() as i32
}

fn merge_row(mut row: Vec<OcrResult_>) -> Line {
    row.sort_by_key(|item| item.bbox.rect.left());
    let score = row.iter().map(|item| item.confidence).sum::<f32>() / row.len() as f32;
    let text = row.into_iter().map(|item| item.text).collect::<Vec<_>>().join(" ");
    Line { text, score }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::rect::Rect;
    use ocr_rs::TextBox;

    fn item(text: &str, left: i32, top: i32, score: f32) -> OcrResult_ {
        OcrResult_::new(
            text.to_owned(),
            score,
            TextBox::new(Rect::at(left, top).of_size(50, 20), score),
        )
    }

    #[test]
    fn merges_rows_top_to_bottom_and_left_to_right() {
        let items = vec![
            item("d", 80, 42, 0.8),
            item("b", 80, 3, 0.6),
            item("c", 10, 40, 1.0),
            item("a", 10, 0, 1.0),
        ];
        let lines = reading_lines(items);
        assert_eq!(
            lines,
            [
                Line {
                    text: "a b".into(),
                    score: 0.8
                },
                Line {
                    text: "c d".into(),
                    score: 0.9
                }
            ]
        );
    }

    #[test]
    fn keeps_engine_order() {
        let lines = engine_lines(vec![item("z", 90, 90, 0.5), item("y", 0, 0, 0.7)]);
        assert_eq!(
            lines,
            [
                Line {
                    text: "z".into(),
                    score: 0.5
                },
                Line {
                    text: "y".into(),
                    score: 0.7
                }
            ]
        );
    }
}
