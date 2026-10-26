# instagram

Instruction-only recipe for counting saved posts and reels on the user's own Instagram account using generic browser tools.

> [!WARNING]
> Automating access or scraping on Instagram may violate its Terms of Use and risk temporary or permanent account restrictions. For complete archives and data safety, the official Instagram data export is the safer and recommended method (see Phase 12).

## Instructions

1. **Safety & Policy Constraints**:
   - Strictly read-only: never invoke tools for liking, commenting, following, sharing, or messaging (all such actions are classified as R3 and permanently blocked).
   - Operates only on the user's own logged-in account in the dedicated browser profile.

2. **Procedure**:
   - Navigate to your saved collection:
     `browser.goto(url="https://www.instagram.com/<your_username>/saved/")`
   - Take a snapshot to confirm page render:
     `browser.snapshot()`
   - Repeatedly scroll down (up to the maximum 20 scrolls limit) with a minimum 2-second delay per scroll to allow dynamic content to load:
     `browser.scroll(direction="down", amount="page")`
   - Collect discovered posts and reels:
     `browser.links(patterns=["/reel/", "/p/"], max=100)`
   - Keep a running count of deduplicated URLs.
   - When 300 unique links are collected (or when the 20-scroll limit is reached), terminate scrolling.
   - Present a clear summary reporting "Found at least N unique saved posts/reels."
