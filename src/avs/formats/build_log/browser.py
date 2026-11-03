"""Playwright helpers: a local static server, a browser launcher that works with
either `playwright install` browsers or a system Chromium, and the tiny test DSL
used by recipes (goto / click / press / wait / expect_text / expect_not_text /
expect_count)."""
from __future__ import annotations

import contextlib
import functools
import glob
import http.server
import os
import threading
from pathlib import Path
from urllib.parse import urlparse


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the CLI output readable
        pass


@contextlib.contextmanager
def serve(root: Path):
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def launch(pw):
    """Launch Chromium. Order: AVS_CHROMIUM env, Playwright's own download, known system paths."""
    exe = os.environ.get("AVS_CHROMIUM")
    if exe:
        return pw.chromium.launch(executable_path=exe)
    try:
        return pw.chromium.launch()
    except Exception:
        for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux*/chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
                        "/usr/bin/google-chrome"):
            for cand in sorted(glob.glob(pattern), reverse=True):
                return pw.chromium.launch(executable_path=cand)
        raise


def run_tests(base_url: str, tests: list[dict]) -> dict:
    """Run recipe tests headless. Returns pass/fail per test plus console errors
    and any request that left localhost (should be none)."""
    from playwright.sync_api import sync_playwright

    results, console_errors, external = [], [], set()
    with sync_playwright() as pw:
        browser = launch(pw)
        for test in tests:
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))
            page.on("request", lambda r: external.add(r.url) if urlparse(r.url).hostname not in ("127.0.0.1", "localhost") else None)
            ok, err = True, ""
            try:
                for action in test["do"]:
                    apply_action(page, base_url, action, page)
            except Exception as exc:
                ok, err = False, str(exc).splitlines()[0][:300]
            results.append({"name": test["name"], "passed": ok, "error": err})
            ctx.close()
        browser.close()
    return {"tests": results, "passed": sum(r["passed"] for r in results), "total": len(results),
            "console_errors": console_errors, "external_requests": sorted(external)}


def apply_action(page, base_url: str, action: dict, target, timeout: int = 3000) -> None:
    """`target` is a Page or FrameLocator to act on; navigation always goes through `page`."""
    from playwright.sync_api import expect

    (verb, arg), = action.items()
    if verb == "goto":
        page.goto(f"{base_url}/{arg}")
    elif verb == "click":
        target.locator(arg).click(timeout=timeout)
    elif verb == "press":
        target.locator("body").press(arg, timeout=timeout)
    elif verb == "wait":
        page.wait_for_timeout(arg)
    elif verb == "expect_text":
        expect(target.locator(arg[0])).to_have_text(str(arg[1]), timeout=timeout)
    elif verb == "expect_not_text":
        expect(target.locator(arg[0])).not_to_have_text(str(arg[1]), timeout=timeout)
    elif verb == "expect_count":
        expect(target.locator(arg[0])).to_have_count(int(arg[1]), timeout=timeout)
    else:
        raise ValueError(f"unknown test action {verb!r}")
