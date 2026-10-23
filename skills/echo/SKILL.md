# echo

Safely echo back inputs without invoking external services or modifying files.

## Examples
- `echo.echo(text="ping")` -> Returns {"ok": true, "data": {"echo": "ping"}}.
- `echo.echo(text="test payload")` -> Returns {"ok": true, "data": {"echo": "test payload"}}.
