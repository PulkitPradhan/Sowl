# browser

Web browsing and automation capabilities, split cleanly into two modes:

1. **System Browser (`browser.open_url`)**:
   - Opens an HTTP/HTTPS URL in your default desktop browser (e.g. Chrome, Firefox) via OS launcher.
   - Uncontrolled: does not interact with page DOM or take snapshots.
   - Use when the user simply wants to view a website themselves.

2. **Controlled Browser (`browser.goto`, `snapshot`, `click`, `type`, `scroll`, `links`, `read_text`, `back`, `wait`)**:
   - Operates a dedicated Chrome instance attached via Playwright over CDP to `~/.ag/chrome-profile`.
   - Accessible only while `ag browser` is running.
   - Uses numbered refs (`@e1`, `@e2`) from accessibility snapshots instead of CSS selectors.
   - Refuses password, one-time-code, and payment inputs.
   - Escalates sensitive buttons (send, submit, delete, pay) to R2 and blocks engagement on social platforms (R3).

## Examples
- `browser.open_url(url="https://docs.python.org")` -> Opens documentation in the user's regular browser.
- `browser.goto(url="https://news.ycombinator.com")` -> Navigates the controlled browser page.
- `browser.snapshot()` -> Captures interactive elements with refs (@e1, @e2).
- `browser.click(ref="@e1")` -> Clicks element with ref @e1.
