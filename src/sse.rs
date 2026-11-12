//! Server-sent events for lair's streams. Every stream is an enum tagged `event` with its payload in
//! `data`; a frame's event name is the tag and its data is the payload, so a frame reads back into the
//! same enum.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// The event name and data line for one stream item.
pub fn frame(item: &impl Serialize) -> (String, String) {
    let mut v = serde_json::to_value(item).expect("stream items serialize");
    let name = v["event"].as_str().expect("stream items are tagged `event`").to_owned();
    (name, v["data"].take().to_string())
}

/// One raw frame (`event: …` / `data: …` lines) back as a stream item; `None` for anything else.
pub fn parse<T: DeserializeOwned>(raw: &str) -> Option<T> {
    let mut name = "";
    let mut data = String::new();
    for line in raw.lines() {
        if let Some(n) = line.strip_prefix("event:") {
            name = n.trim();
        } else if let Some(d) = line.strip_prefix("data:") {
            data.push_str(d.trim_start());
        }
    }
    let data: Value = serde_json::from_str(&data).ok()?;
    serde_json::from_value(json!({ "event": name, "data": data })).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Done, Event, Finish};

    #[test]
    fn generation_events_survive_the_trip_between_nodes() {
        let events = [
            Event::Thinking { text: "hm".into() },
            Event::Answer { text: "Oi".into() },
            Event::Done(Done {
                prompt_tokens: 3,
                completion_tokens: 2,
                finish_reason: Finish::Length,
                ..Default::default()
            }),
            Event::Failed { message: "boom".into() },
        ];
        for e in events {
            let (name, data) = frame(&e);
            assert_eq!(parse::<Event>(&format!("event: {name}\ndata: {data}\n\n")), Some(e));
        }
        assert_eq!(parse::<Event>("event: other\ndata: {}\n\n"), None);
    }

    #[test]
    fn frames_keep_the_wire_format_older_nodes_speak() {
        assert_eq!(frame(&Event::Answer { text: "Oi".into() }), ("answer".into(), r#"{"text":"Oi"}"#.into()));
    }
}
