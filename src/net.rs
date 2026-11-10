//! How lair reaches the network: its HTTP clients and size-capped downloads.

use std::time::Duration;

use anyhow::Result;

const USER_AGENT: &str = concat!("lair/", env!("CARGO_PKG_VERSION"));

/// For requests that answer quickly: tools, skill downloads, peers.
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(USER_AGENT)
        .build()
        .expect("an HTTP client with default settings builds")
}

/// For model downloads, which take as long as gigabytes take; only connecting is bounded.
pub fn download_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .user_agent(USER_AGENT)
        .build()
        .expect("an HTTP client with default settings builds")
}

/// The body of `url`, up to `limit` bytes; `true` when it was longer and got cut there.
pub async fn get_capped(
    http: &reqwest::Client,
    url: &str,
    limit: usize,
) -> Result<(reqwest::header::HeaderMap, Vec<u8>, bool)> {
    let mut resp = http.get(url).send().await?.error_for_status()?;
    let headers = resp.headers().clone();
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        body.extend_from_slice(&chunk);
        if body.len() > limit {
            body.truncate(limit);
            return Ok((headers, body, true));
        }
    }
    Ok((headers, body, false))
}
