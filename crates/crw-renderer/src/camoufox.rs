//! Camoufox stealth renderer tier (opt-in, REST — not CDP).
//!
//! Backed by a [`camofox-browser`](https://github.com/jo-inc/camofox-browser)
//! REST sidecar (default port `9377`) which drives a Camoufox (Firefox-fork)
//! browser with C++-level fingerprint spoofing. This tier covers
//! fingerprint/bot-challenge blocks (e.g. Cloudflare 403) that the CDP tiers
//! cannot pass.
//!
//! ## Why REST, not CDP
//! Camoufox does not expose a usable Chromium-CDP `Fetch.*` surface, so it is
//! reached over the sidecar's HTTP API instead of the [`crate::cdp`] path. The
//! renderer is a plain [`reqwest::Client`] — it never touches `cdp.rs`.
//!
//! ## Sequence (per request, fresh session — no cookie carry-over)
//! 1. `POST {base}/tabs` `{userId, sessionKey, url}` → `{tabId, url}` (the
//!    `url` body navigates the new tab as it is created).
//! 2. `POST {base}/tabs/{tabId}/wait` `{userId, timeout}` → `{ok, ready}`
//!    (best-effort — `waitForPageReady`: `domcontentloaded` + `networkidle` +
//!    hydration poll + settle — so JS has finished before we read the DOM;
//!    any failure proceeds to step 3 anyway).
//! 3. `POST {base}/tabs/{tabId}/evaluate` `{userId, expression}` with
//!    `expression = "document.documentElement.outerHTML"` → `{ok, result}`.
//!    `result` is the fully JS-rendered DOM string — exactly what CRW's
//!    post-render pipeline (`only_main_content`, tag filters, markdown) needs.
//! 4. `DELETE {base}/sessions/{userId}` — ALWAYS, even on error, so the sidecar
//!    never leaks a server-side session.
//!
//! The endpoint contract above is from the camofox-browser `openapi.json`.

use async_trait::async_trait;
use crw_core::Deadline;
use crw_core::error::{CrwError, CrwResult};
use crw_core::types::FetchResult;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::traits::PageFetcher;

/// JS expression evaluated to retrieve the post-render DOM.
const OUTER_HTML_EXPR: &str = "document.documentElement.outerHTML";
/// Fixed budget for the best-effort session cleanup. Deliberately NOT derived
/// from the request deadline: a hung cleanup must never consume the caller's
/// remaining budget.
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
/// Budget for the `is_available` health probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Default server-side budget (ms) sent as `timeout` to `POST
/// /tabs/{tabId}/wait` when the caller did not supply `wait_for_ms`. Mirrors
/// the camofox-browser default (`timeout = 10000` in `server.js`): covers
/// `domcontentloaded` + up-to-5s `networkidle` + hydration poll (capped at
/// `min(timeout, 10000)`) + settle.
const DEFAULT_WAIT_TIMEOUT_MS: u64 = 10_000;

/// Compute the server-side `timeout` to send to `POST /tabs/{tabId}/wait`:
/// the caller's `wait_for_ms` when present, else [`DEFAULT_WAIT_TIMEOUT_MS`],
/// clamped to the current call budget so the wait can never overrun the
/// request deadline. Pure function so the clamping is unit-testable without
/// HTTP mocks.
fn wait_timeout_ms(wait_for_ms: Option<u64>, budget: Duration) -> u64 {
    let desired = wait_for_ms.unwrap_or(DEFAULT_WAIT_TIMEOUT_MS);
    (budget.as_millis().min(desired as u128)) as u64
}

/// Opt-in Camoufox stealth renderer. Construct via [`CamoufoxRenderer::new`].
pub struct CamoufoxRenderer {
    name: String,
    /// Trimmed base URL, no trailing slash (e.g. `http://localhost:9377`).
    base_url: String,
    /// Bearer token; empty string means no `Authorization` header is sent.
    api_key: String,
    /// Overall per-request REST budget (`config.camoufox_timeout()`).
    timeout: Duration,
    client: reqwest::Client,
}

impl CamoufoxRenderer {
    pub fn new(name: &str, base_url: &str, api_key: &str, timeout_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            timeout: Duration::from_millis(timeout_ms),
            client: reqwest::Client::new(),
        }
    }

    /// Attach the bearer header when an API key is configured.
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.api_key.is_empty() {
            rb
        } else {
            rb.bearer_auth(&self.api_key)
        }
    }

    /// Random opaque identifier (`<prefix>` + 32 hex chars). Fresh per request
    /// so sessions never share cookies — safer for stealth.
    fn rand_id(prefix: &str) -> String {
        let n: u128 = rand::random();
        format!("{prefix}{n:032x}")
    }

    /// Remaining budget for the next REST call: the smaller of the request
    /// deadline's remaining time and the configured camoufox timeout. This is
    /// the single source of the clamp formula — every REST call uses it.
    fn call_budget(&self, deadline: &Deadline) -> Duration {
        deadline.remaining().min(self.timeout)
    }

    /// Send a JSON POST, honoring the deadline, and parse a JSON response.
    /// Maps transport/timeout/non-2xx into the appropriate [`CrwError`].
    async fn post_json(
        &self,
        path: &str,
        body: &serde_json::Value,
        deadline: &Deadline,
    ) -> CrwResult<serde_json::Value> {
        let budget = self.call_budget(deadline);
        if budget.is_zero() {
            return Err(CrwError::Timeout(deadline.requested_ms()));
        }
        let url = format!("{}{}", self.base_url, path);
        let fut = self.auth(self.client.post(&url).json(body)).send();
        let resp = tokio::time::timeout(budget, fut)
            .await
            .map_err(|_| CrwError::Timeout(budget.as_millis() as u64))?
            .map_err(|e| CrwError::RendererError(format!("camoufox POST {path}: {e}")))?;
        let status = resp.status();
        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CrwError::RendererError(format!("camoufox POST {path} body: {e}")))?;
        if !status.is_success() {
            let msg = value
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error");
            return Err(CrwError::RendererError(format!(
                "camoufox POST {path} -> HTTP {}: {msg}",
                status.as_u16()
            )));
        }
        Ok(value)
    }

    /// Create a fresh tab navigated to `url`. Returns its `tabId`. Some sidecar
    /// builds intermittently omit `tabId` under proxy-IP races on the first
    /// attempt, so we retry exactly once before failing.
    async fn create_tab(
        &self,
        url: &str,
        user_id: &str,
        session_key: &str,
        deadline: &Deadline,
    ) -> CrwResult<String> {
        let body = serde_json::json!({
            "userId": user_id,
            "sessionKey": session_key,
            "url": url,
        });
        for attempt in 0..2u8 {
            let value = self.post_json("/tabs", &body, deadline).await?;
            if let Some(tab_id) = value.get("tabId").and_then(|t| t.as_str()) {
                return Ok(tab_id.to_string());
            }
            tracing::debug!(
                renderer = %self.name,
                attempt,
                "camoufox create tab returned no tabId, retrying"
            );
        }
        Err(CrwError::RendererError(
            "camoufox: create tab returned no tabId after retry".into(),
        ))
    }

    /// Best-effort wait for the page to finish loading and executing JS via
    /// `POST {base}/tabs/{tabId}/wait` (`waitForPageReady` server-side:
    /// `domcontentloaded` + `networkidle` + hydration poll + settle).
    ///
    /// FAIL-OPEN by design: every failure mode — expired budget (wait is
    /// skipped), transport/timeout/non-2xx error, `ok=false`, or
    /// `ready=false` (page did not fully settle) — only logs and returns, so
    /// the caller always proceeds to `evaluate_outer_html` with whatever DOM
    /// exists. A partially-rendered page is strictly better than no page.
    ///
    /// Body is `{userId, timeout}` only. The sidecar's `/wait` handler
    /// (`server.js`) destructures just `{userId, timeout, waitForNetwork,
    /// dismissConsent}` and looks the tab up by `userId` + `tabId`, so
    /// `sessionKey` is intentionally omitted (the server would ignore it).
    /// `timeout` is [`wait_timeout_ms`]: `wait_for_ms` when the caller
    /// supplied one, else [`DEFAULT_WAIT_TIMEOUT_MS`], clamped to the
    /// remaining call budget (which is itself `min(deadline.remaining(),
    /// camoufox_timeout)`). [`Self::post_json`] recomputes that same budget as
    /// the HTTP timeout, so the two agree modulo microseconds of clock drift;
    /// a client-side timeout firing marginally first is fail-open anyway.
    async fn wait_for_ready(
        &self,
        tab_id: &str,
        user_id: &str,
        wait_for_ms: Option<u64>,
        deadline: &Deadline,
    ) {
        let budget = self.call_budget(deadline);
        if budget.is_zero() {
            tracing::debug!(
                renderer = %self.name,
                tab_id,
                "skipping camoufox wait: deadline exhausted (fail-open)"
            );
            return;
        }
        let timeout_ms = wait_timeout_ms(wait_for_ms, budget);
        if timeout_ms == 0 {
            return;
        }
        let body = serde_json::json!({
            "userId": user_id,
            "timeout": timeout_ms,
        });
        match self
            .post_json(&format!("/tabs/{tab_id}/wait"), &body, deadline)
            .await
        {
            Ok(value) => {
                let ok = value.get("ok").and_then(|o| o.as_bool()).unwrap_or(true);
                let ready = value.get("ready").and_then(|r| r.as_bool()).unwrap_or(true);
                if !ok || !ready {
                    tracing::debug!(
                        renderer = %self.name,
                        tab_id,
                        ok,
                        ready,
                        "camoufox wait settled without ready (fail-open, proceeding to evaluate)"
                    );
                }
            }
            Err(e) => tracing::warn!(
                renderer = %self.name,
                tab_id,
                "camoufox wait failed (fail-open, proceeding to evaluate): {e}"
            ),
        }
    }

    /// Evaluate `document.documentElement.outerHTML` and return the DOM string.
    async fn evaluate_outer_html(
        &self,
        tab_id: &str,
        user_id: &str,
        deadline: &Deadline,
    ) -> CrwResult<String> {
        let body = serde_json::json!({
            "userId": user_id,
            "expression": OUTER_HTML_EXPR,
        });
        let value = self
            .post_json(&format!("/tabs/{tab_id}/evaluate"), &body, deadline)
            .await?;
        // Shape: { ok: bool, result: <any> }. For outerHTML, result is a string.
        if value.get("ok").and_then(|o| o.as_bool()) == Some(false) {
            return Err(CrwError::RendererError(
                "camoufox: evaluate returned ok=false".into(),
            ));
        }
        match value.get("result").and_then(|r| r.as_str()) {
            Some(html) => Ok(html.to_string()),
            None => Err(CrwError::RendererError(
                "camoufox: evaluate result was not an HTML string".into(),
            )),
        }
    }

    /// Best-effort session teardown. Uses [`CLEANUP_TIMEOUT`] (NOT the request
    /// deadline) and swallows every error — a failed cleanup is logged, never
    /// propagated.
    async fn destroy_session(&self, user_id: &str) {
        let url = format!("{}/sessions/{}", self.base_url, user_id);
        let fut = self.auth(self.client.delete(&url)).send();
        match tokio::time::timeout(CLEANUP_TIMEOUT, fut).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => tracing::warn!(
                renderer = %self.name, user_id, "camoufox session cleanup failed: {e}"
            ),
            Err(_) => tracing::warn!(
                renderer = %self.name,
                user_id,
                "camoufox session cleanup timed out after {}s (leaked server-side session)",
                CLEANUP_TIMEOUT.as_secs()
            ),
        }
    }

    /// Full create → wait → evaluate sequence with GUARANTEED cleanup. The inner
    /// result is bound first, then `destroy_session` runs at statement level so
    /// it fires on both the Ok and Err paths, and only afterward is the result
    /// returned.
    async fn run_sequence(
        &self,
        url: &str,
        user_id: &str,
        session_key: &str,
        wait_for_ms: Option<u64>,
        deadline: &Deadline,
    ) -> CrwResult<(u16, String)> {
        let inner = self
            .run_sequence_inner(url, user_id, session_key, wait_for_ms, deadline)
            .await;
        self.destroy_session(user_id).await;
        inner
    }

    async fn run_sequence_inner(
        &self,
        url: &str,
        user_id: &str,
        session_key: &str,
        wait_for_ms: Option<u64>,
        deadline: &Deadline,
    ) -> CrwResult<(u16, String)> {
        let tab_id = self.create_tab(url, user_id, session_key, deadline).await?;
        // Best-effort: never errors — on any wait failure the evaluate below
        // still runs against whatever DOM exists.
        self.wait_for_ready(&tab_id, user_id, wait_for_ms, deadline)
            .await;
        let html = self.evaluate_outer_html(&tab_id, user_id, deadline).await?;
        // A bot wall or an empty body is a failure for THIS tier — surface it as
        // a retryable RendererError so the fallback loop / breaker can react.
        if html.trim().is_empty() {
            return Err(CrwError::RendererError(
                "camoufox: evaluate returned empty HTML".into(),
            ));
        }
        if let Some(kind) = looks_like_wall(&html) {
            return Err(CrwError::RendererError(format!(
                "camoufox: bot {kind} detected in rendered HTML"
            )));
        }
        // The /evaluate path does not surface the page's HTTP status; the
        // navigation in `create_tab` already succeeded server-side, so 200.
        Ok((200, html))
    }
}

/// Heuristic detection of a bot wall / challenge interstitial in rendered HTML.
/// Returns the wall kind when matched. These markers are CRW-specific (the
/// stock camofox-browser client does not classify walls); they let the tier
/// report a retryable failure instead of returning a useless challenge page.
fn looks_like_wall(html: &str) -> Option<&'static str> {
    let h = html.to_ascii_lowercase();
    const NEEDLES: &[(&str, &str)] = &[
        ("just a moment", "challenge"),
        ("verifying you are human", "challenge"),
        ("checking your browser before", "challenge"),
        ("cf-challenge", "challenge"),
        ("/cdn-cgi/challenge-platform", "challenge"),
        ("attention required! | cloudflare", "wall"),
        ("enable javascript and cookies to continue", "wall"),
    ];
    NEEDLES
        .iter()
        .find(|(needle, _)| h.contains(needle))
        .map(|(_, kind)| *kind)
}

#[async_trait]
impl PageFetcher for CamoufoxRenderer {
    async fn fetch(
        &self,
        url: &str,
        _headers: &HashMap<String, String>,
        wait_for_ms: Option<u64>,
        deadline: Deadline,
    ) -> CrwResult<FetchResult> {
        if deadline.expired() {
            return Err(CrwError::Timeout(deadline.requested_ms()));
        }
        let start = Instant::now();
        let user_id = Self::rand_id("crw_");
        let session_key = Self::rand_id("task_");
        let (status, html) = self
            .run_sequence(url, &user_id, &session_key, wait_for_ms, &deadline)
            .await?;

        Ok(FetchResult {
            url: url.to_string(),
            final_url: None,
            status_code: status,
            html,
            content_type: None,
            raw_bytes: None,
            rendered_with: Some(self.name.clone()),
            elapsed_ms: start.elapsed().as_millis() as u64,
            warning: None,
            render_decision: None,
            credit_cost: 0,
            warnings: Vec::new(),
            truncated: false,
            deadline_exceeded: deadline.remaining().is_zero(),
            captured_responses: Vec::new(),
            // Camoufox is an HTTP sidecar (no CDP) — it never captures.
            screenshot: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_js(&self) -> bool {
        true
    }

    /// Health probe against `GET {base}/health`. Mirrors the camofox-browser
    /// client's availability check. Cheap (~5s cap) so the FallbackRenderer can
    /// skip a down sidecar instead of paying the full per-request budget.
    async fn is_available(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        let fut = self.auth(self.client.get(&url)).send();
        matches!(
            tokio::time::timeout(PROBE_TIMEOUT, fut).await,
            Ok(Ok(r)) if r.status().is_success()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn renderer(base_url: &str) -> CamoufoxRenderer {
        CamoufoxRenderer::new("camoufox", base_url, "", 30_000)
    }

    fn deadline() -> Deadline {
        Deadline::from_request_ms(30_000)
    }

    /// A spent deadline must report the budget the caller was given, not how
    /// many milliseconds late we noticed. Prod told customers holding a 30s
    /// budget that their scrape "timed out after 1ms" 326 times in ten days.
    /// This exercises the real call site: `fetch` checks `expired()` before it
    /// touches the network, so no server is needed.
    #[tokio::test]
    async fn expired_deadline_reports_the_requested_budget() {
        let r = renderer("http://127.0.0.1:1");
        let spent = Deadline::from_request_ms(0);
        assert!(spent.expired());

        let err = r
            .fetch("https://example.com", &HashMap::new(), None, spent)
            .await
            .expect_err("an expired deadline must not attempt a fetch");

        match err {
            CrwError::Timeout(ms) => assert_eq!(
                ms, 0,
                "must echo the caller's budget, not the overrun since it lapsed"
            ),
            other => panic!("expected Timeout, got {other:?}"),
        }

        // And the message a customer actually reads.
        let long = Deadline::from_request_ms(30_000);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let budget = long.requested_ms();
        assert_eq!(
            CrwError::Timeout(budget).to_string(),
            "Timeout after 30000ms"
        );
    }

    async fn mount_delete_session(server: &MockServer) {
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true, "closed": 1
            })))
            .mount(server)
            .await;
    }

    /// Mock a successful best-effort wait: `POST /tabs/{tab_id}/wait` →
    /// `{ok: true, ready: true}`. Body-agnostic (any `userId`/`timeout`) so
    /// per-request random IDs do not need matching; timeout-value propagation
    /// is covered by the pure `wait_timeout_ms` unit tests below.
    async fn mount_wait_ok(server: &MockServer, tab_id: &str) {
        mount_wait_body(
            server,
            tab_id,
            serde_json::json!({"ok": true, "ready": true}),
        )
        .await;
    }

    async fn mount_wait_body(server: &MockServer, tab_id: &str, body: serde_json::Value) {
        Mock::given(method("POST"))
            .and(path(format!("/tabs/{tab_id}/wait")))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(server)
            .await;
    }

    /// Mount a wait mock that MUST fire exactly once — use on the happy path
    /// to prove the fetch sequence actually waits before evaluating.
    async fn mount_wait_ok_expect_once(server: &MockServer, tab_id: &str) {
        Mock::given(method("POST"))
            .and(path(format!("/tabs/{tab_id}/wait")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true, "ready": true
            })))
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn happy_path_returns_html_and_cleans_up() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "tabId": "t1", "url": "https://example.com"
            })))
            .mount(&server)
            .await;
        // Best-effort wait between navigation and evaluate — must fire once.
        mount_wait_ok_expect_once(&server, "t1").await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": "<html><body>real content well over the empty threshold here</body></html>"
            })))
            .mount(&server)
            .await;
        // Expect cleanup to fire exactly once.
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .expect(1)
            .mount(&server)
            .await;

        let r = renderer(&server.uri());
        let res = r
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect("fetch should succeed");
        assert_eq!(res.rendered_with.as_deref(), Some("camoufox"));
        assert_eq!(res.status_code, 200);
        assert!(res.html.contains("real content"));
        // server drop verifies the expect(1) on the wait + DELETE mocks
    }

    #[tokio::test]
    async fn cleanup_fires_on_evaluate_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        mount_wait_ok(&server, "t1").await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(
                ResponseTemplate::new(500).set_body_json(serde_json::json!({"error": "boom"})),
            )
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .expect(1) // cleanup MUST fire even though evaluate failed
            .mount(&server)
            .await;

        let r = renderer(&server.uri());
        let err = r
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("evaluate 500 should error");
        assert!(matches!(err, CrwError::RendererError(_)));
    }

    /// Helper: mock create-tab OK + wait OK + evaluate returning `eval_body`,
    /// with a cleanup mock that MUST fire once. Returns the started server.
    async fn server_with_evaluate(eval_body: serde_json::Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        mount_wait_ok(&server, "t1").await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(eval_body))
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .expect(1) // cleanup must fire on every error path below
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn empty_html_errors_and_cleans_up() {
        let server = server_with_evaluate(serde_json::json!({"ok": true, "result": "   "})).await;
        let err = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("empty HTML should error");
        match err {
            CrwError::RendererError(m) => assert!(m.contains("empty HTML")),
            other => panic!("expected RendererError, got {other:?}"),
        }
        // server drop verifies cleanup expect(1)
    }

    #[tokio::test]
    async fn evaluate_ok_false_errors_and_cleans_up() {
        let server = server_with_evaluate(serde_json::json!({"ok": false})).await;
        let err = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("ok=false should error");
        match err {
            CrwError::RendererError(m) => assert!(m.contains("ok=false")),
            other => panic!("expected RendererError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn evaluate_non_string_result_errors_and_cleans_up() {
        // result is a number, not the expected HTML string.
        let server = server_with_evaluate(serde_json::json!({"ok": true, "result": 42})).await;
        let err = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("non-string result should error");
        match err {
            CrwError::RendererError(m) => assert!(m.contains("not an HTML string")),
            other => panic!("expected RendererError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_tab_id_retries_once_then_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(2) // first attempt + one retry
            .mount(&server)
            .await;
        // Wait must never fire: create-tab never produced a tabId.
        Mock::given(method("POST"))
            .and(path("/tabs/t1/wait"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true, "ready": true
            })))
            .expect(0)
            .mount(&server)
            .await;
        mount_delete_session(&server).await;

        let r = renderer(&server.uri());
        let err = r
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("missing tabId should error");
        match err {
            CrwError::RendererError(m) => assert!(m.contains("no tabId")),
            other => panic!("expected RendererError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn wall_detection_returns_retryable_renderer_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        mount_wait_ok(&server, "t1").await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": "<html><head><title>Just a moment...</title></head><body>verifying you are human</body></html>"
            })))
            .mount(&server)
            .await;
        mount_delete_session(&server).await;

        let r = renderer(&server.uri());
        let err = r
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect_err("wall should error");
        match err {
            CrwError::RendererError(m) => assert!(m.contains("challenge")),
            other => panic!("expected RendererError, got {other:?}"),
        }
    }

    /// Fail-open: `ready=false` means the page did not fully settle, but the
    /// fetch must still return whatever DOM `evaluate` produces.
    #[tokio::test]
    async fn wait_ready_false_still_returns_html() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        mount_wait_body(
            &server,
            "t1",
            serde_json::json!({"ok": true, "ready": false}),
        )
        .await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": "<html><body>late-hydrated content after unsettled wait</body></html>"
            })))
            .expect(1)
            .mount(&server)
            .await;
        mount_delete_session(&server).await;

        let res = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect("ready=false must not fail the fetch");
        assert!(res.html.contains("late-hydrated"));
    }

    /// Fail-open: a transport-level wait failure (HTTP 500 here; timeouts and
    /// 404s take the same path) must not fail the fetch — evaluate still runs
    /// and cleanup still fires.
    #[tokio::test]
    async fn wait_http_error_is_fail_open() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/wait"))
            .respond_with(
                ResponseTemplate::new(500).set_body_json(serde_json::json!({"error": "boom"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": "<html><body>content despite wait transport failure</body></html>"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .expect(1)
            .mount(&server)
            .await;

        let res = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect("wait 500 must not fail the fetch");
        assert!(res.html.contains("despite wait"));
    }

    /// Fail-open: `ok=false` from `/wait` (unexpected per the sidecar
    /// contract, which always returns `ok:true` here) still proceeds.
    #[tokio::test]
    async fn wait_ok_false_still_returns_html() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        mount_wait_body(&server, "t1", serde_json::json!({"ok": false})).await;
        Mock::given(method("POST"))
            .and(path("/tabs/t1/evaluate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": "<html><body>content despite ok=false wait</body></html>"
            })))
            .mount(&server)
            .await;
        mount_delete_session(&server).await;

        let res = renderer(&server.uri())
            .fetch("https://example.com", &HashMap::new(), None, deadline())
            .await
            .expect("wait ok=false must not fail the fetch");
        assert!(res.html.contains("despite ok=false"));
    }

    #[test]
    fn wait_timeout_uses_wait_for_ms_else_default_clamped_to_budget() {
        // No caller preference → server default.
        assert_eq!(
            wait_timeout_ms(None, Duration::from_secs(30)),
            DEFAULT_WAIT_TIMEOUT_MS
        );
        // Caller preference wins when budget allows.
        assert_eq!(wait_timeout_ms(Some(2_000), Duration::from_secs(30)), 2_000);
        // Clamped to the remaining budget so the wait never overruns the
        // request deadline.
        assert_eq!(wait_timeout_ms(None, Duration::from_millis(500)), 500);
        assert_eq!(wait_timeout_ms(Some(20_000), Duration::from_secs(5)), 5_000);
        // Zero budget → zero (the caller skips the wait entirely).
        assert_eq!(wait_timeout_ms(None, Duration::ZERO), 0);
    }

    #[tokio::test]
    async fn expired_deadline_short_circuits_without_http() {
        // No mocks mounted: if any HTTP call were made it would 404 and the
        // error message would differ from a Timeout.
        let server = MockServer::start().await;
        let r = renderer(&server.uri());
        let err = r
            .fetch(
                "https://example.com",
                &HashMap::new(),
                None,
                Deadline::from_request_ms(0),
            )
            .await
            .expect_err("expired deadline should error immediately");
        assert!(matches!(err, CrwError::Timeout(_)));
    }

    #[tokio::test]
    async fn is_available_true_when_health_ok() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;
        assert!(renderer(&server.uri()).is_available().await);
    }

    #[tokio::test]
    async fn is_available_false_when_health_503() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(
                ResponseTemplate::new(503).set_body_json(serde_json::json!({"ok": false})),
            )
            .mount(&server)
            .await;
        assert!(!renderer(&server.uri()).is_available().await);
    }

    #[test]
    fn wall_needles_match_and_clean_html_passes() {
        assert_eq!(looks_like_wall("Just a moment..."), Some("challenge"));
        assert_eq!(
            looks_like_wall("<script src=/cdn-cgi/challenge-platform/x>"),
            Some("challenge")
        );
        assert_eq!(
            looks_like_wall("<div class=cf-challenge>"),
            Some("challenge")
        );
        assert_eq!(
            looks_like_wall("Verifying you are human"),
            Some("challenge")
        );
        assert_eq!(
            looks_like_wall("Checking your browser before access"),
            Some("challenge")
        );
        assert_eq!(
            looks_like_wall("<h1>Attention Required! | Cloudflare</h1>"),
            Some("wall")
        );
        assert_eq!(
            looks_like_wall("Please enable JavaScript and cookies to continue"),
            Some("wall")
        );
        assert_eq!(
            looks_like_wall("<html><body>normal page</body></html>"),
            None
        );
    }

    #[tokio::test]
    async fn cleanup_uses_fixed_timeout_not_request_deadline() {
        // A near-exhausted request deadline must NOT prevent cleanup from
        // running: cleanup has its own fixed budget. We give a tiny deadline so
        // the fetch fails fast (budget exhausted mid-sequence), then assert the
        // DELETE cleanup still fired exactly once.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tabs"))
            .respond_with(
                // Large delay (2s) vs the 5ms deadline gives a ~400x margin, so
                // the create-tab call deterministically times out before its
                // response arrives even under heavy CI load — no flakiness.
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(2))
                    .set_body_json(serde_json::json!({"tabId": "t1"})),
            )
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/sessions/.+"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .expect(1)
            .mount(&server)
            .await;

        let r = renderer(&server.uri());
        // 5ms deadline << 2s create-tab delay -> create_tab times out, but
        // cleanup (fixed 5s budget) must still fire.
        let err = r
            .fetch(
                "https://example.com",
                &HashMap::new(),
                None,
                Deadline::from_request_ms(5),
            )
            .await
            .expect_err("tiny deadline should error");
        assert!(matches!(err, CrwError::Timeout(_)));
        // server drop verifies cleanup expect(1) despite the exhausted deadline
    }
}
