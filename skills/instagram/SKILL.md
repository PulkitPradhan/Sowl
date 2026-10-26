# instagram

Instruction-only recipe for viewing and counting saved posts and reels on the user's own Instagram account using generic browser tools.

> [!WARNING]
> Automating access or scraping on Instagram may violate its Terms of Use and risk temporary or permanent account restrictions. For complete archives and data safety, the official Instagram data export is the safer and recommended method (see Phase 12).

## Instructions

1. **Safety & Policy Constraints**:
   - Strictly read-only: never invoke tools for liking, commenting, following, sharing, or messaging (all such actions are classified as R3 and permanently blocked).
   - NEVER refuse with disclaimers like "I cannot access your account" or "I cannot view private profiles". You control a dedicated browser instance and MUST interact using browser tools.
   - NEVER ask for usernames, URLs, or credentials upfront before opening the browser.

2. **Step-by-Step Procedure**:
   - **Step 1: Always Open Instagram First**:
     Do NOT ask any questions before opening the browser. The very first tool call MUST be:
     `browser.goto(url="https://www.instagram.com/")`
   - **Step 2: Inspect Authentication State via Snapshot**:
     Immediately inspect the page:
     `browser.snapshot()`
   - **Step 3: Check If User Is Logged In**:
     - Inspect the snapshot elements:
     - **If NOT logged in** (e.g. snapshot contains "Log In", "Sign Up", or login form inputs):
       Return an `ask` action asking the user to log in in the opened browser window:
       `{"kind": "ask", "question": "Please log in to your Instagram account in the open browser window, then reply here once logged in."}`
     - **If logged in** (e.g. snapshot contains navigation links like "Home", "Search", "Explore", "Reels", "Messages", "Profile", or sidebar):
       - If the user's username was provided in the task or answer (e.g. `<username>`), navigate directly to saved:
         `browser.goto(url="https://www.instagram.com/<username>/saved/")`
       - If username was not provided, look for the "Profile" link ref in the snapshot and click it with `browser.click(ref=...)` to reach the profile and saved collection.
   - **Step 4: Collect Saved Links**:
     - Extract discovered posts and reels:
       `browser.links(patterns=["/reel/", "/p/"], max=100)`
     - Repeatedly scroll down (up to 20 scrolls limit) with a minimum 2-second delay per scroll to allow dynamic content to load:
       `browser.scroll(direction="down", amount="page")`
     - Keep a running count of deduplicated URLs.
   - **Step 5: Report Results**:
     - When 300 unique links are collected (or when the 20-scroll limit is reached), terminate scrolling.
     - Present a clear summary reporting "Found at least N unique saved posts/reels."
