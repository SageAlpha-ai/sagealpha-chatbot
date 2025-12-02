"""
SageAlpha.ai PDF Blueprint
PDF generation and report handling with async support
"""

import os
import subprocess
import tempfile
from io import BytesIO

from flask import Blueprint, current_app, make_response, render_template, send_file

pdf_bp = Blueprint("pdf", __name__, template_folder="../templates")

# Optional pdfkit support
try:
    import pdfkit

    PDFKIT_AVAILABLE = True
except ImportError:
    pdfkit = None
    PDFKIT_AVAILABLE = False


def find_wkhtmltopdf() -> str | None:
    """Return first usable wkhtmltopdf binary path or None."""
    candidates = [
        os.path.join(os.getcwd(), "bin", "wkhtmltopdf"),
        "/usr/local/bin/wkhtmltopdf",
        "/usr/bin/wkhtmltopdf",
        "/usr/bin/wkhtmltopdf-amd64",
        "/usr/local/bin/wkhtmltox",
    ]
    for p in candidates:
        if p and os.path.exists(p) and os.access(p, os.X_OK):
            return p

    for cmd in ("wkhtmltopdf",):
        try:
            which = subprocess.run(["which", cmd], capture_output=True, text=True)
            if which.returncode == 0:
                path = which.stdout.strip().splitlines()[0]
                if path and os.path.exists(path) and os.access(path, os.X_OK):
                    return path
        except Exception:
            continue
    return None


def try_pdfkit_from_string(html: str) -> bytes | None:
    """Attempt to generate PDF using pdfkit. Return bytes or None."""
    if not PDFKIT_AVAILABLE:
        return None

    config = None
    try:
        wk = find_wkhtmltopdf()
        if wk:
            config = pdfkit.configuration(wkhtmltopdf=wk)

        options = {
            "page-size": "A4",
            "encoding": "UTF-8",
            "enable-local-file-access": None,
            "quiet": "",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
            "margin-left": "10mm",
            "margin-right": "10mm",
        }
        pdf_bytes = pdfkit.from_string(html, False, configuration=config, options=options)
        if isinstance(pdf_bytes, (bytes, bytearray)):
            return bytes(pdf_bytes)
    except Exception as e:
        current_app.logger.exception(f"pdfkit.from_string failed: {e}")
    return None


def try_wkhtmltopdf_subprocess(html: str) -> bytes | None:
    """Run wkhtmltopdf as subprocess. Returns bytes or None."""
    wk = find_wkhtmltopdf()
    if not wk:
        return None

    try:
        proc = subprocess.Popen(
            [wk, "-", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate(input=html.encode("utf-8"), timeout=60)
        if proc.returncode == 0 and out:
            return out
        current_app.logger.warning(
            f"wkhtmltopdf stdin/stdout failed (rc={proc.returncode}), "
            f"stderr: {err.decode('utf-8', errors='ignore')}"
        )
    except Exception as e:
        current_app.logger.exception(f"wkhtmltopdf stdin/stdout attempt failed: {e}")

    # Tempfile fallback
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as in_f:
            in_f.write(html.encode("utf-8"))
            in_path = in_f.name
        out_fd, out_path = tempfile.mkstemp(suffix=".pdf")
        os.close(out_fd)
        try:
            run = subprocess.run([wk, in_path, out_path], capture_output=True, timeout=60)
            if run.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    data = f.read()
                return data
            else:
                current_app.logger.warning(
                    f"wkhtmltopdf returned non-zero ({run.returncode}). "
                    f"stderr: {run.stderr.decode('utf-8', errors='ignore')}"
                )
        finally:
            try:
                os.remove(in_path)
            except Exception:
                pass
            try:
                os.remove(out_path)
            except Exception:
                pass
    except Exception as e:
        current_app.logger.exception(f"wkhtmltopdf tempfile method failed: {e}")
    return None


def try_playwright_pdf(html: str) -> bytes | None:
    """Generate PDF using Playwright Chromium."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox"])
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={
                    "top": "10mm",
                    "bottom": "10mm",
                    "left": "10mm",
                    "right": "10mm",
                },
            )
            browser.close()
            return pdf_bytes
    except Exception as e:
        current_app.logger.exception(f"Playwright PDF generation failed: {e}")
    return None


@pdf_bp.route("/download-report")
def download_report():
    """
    Render report template. Try pdfkit -> wkhtmltopdf -> Playwright -> HTML fallback.
    """
    template_name = "sagealpha_reports.html"
    rendered = render_template(template_name)

    # 1) Try pdfkit
    if PDFKIT_AVAILABLE:
        pdf_bytes = try_pdfkit_from_string(rendered)
        if pdf_bytes:
            resp = make_response(pdf_bytes)
            resp.headers["Content-Type"] = "application/pdf"
            resp.headers["Content-Disposition"] = (
                'attachment; filename="SageAlpha_CRH_Report.pdf"'
            )
            resp.headers["X-PDF-Generated"] = "yes (pdfkit)"
            return resp
        current_app.logger.info(
            "pdfkit available but failed, falling back to wkhtmltopdf subprocess."
        )

    # 2) Try wkhtmltopdf subprocess
    pdf_bytes = try_wkhtmltopdf_subprocess(rendered)
    if pdf_bytes:
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = (
            'attachment; filename="SageAlpha_CRH_Report.pdf"'
        )
        resp.headers["X-PDF-Generated"] = "yes (wkhtmltopdf)"
        return resp

    # 3) Try Playwright
    pdf_bytes = try_playwright_pdf(rendered)
    if pdf_bytes:
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = (
            'attachment; filename="SageAlpha_CRH_Report.pdf"'
        )
        resp.headers["X-PDF-Generated"] = "yes (playwright)"
        return resp

    # 4) Fallback: return HTML inline
    current_app.logger.warning(
        "PDF generation failed (no pdfkit/wkhtmltopdf/playwright); returning HTML."
    )
    resp = make_response(rendered)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = (
        'inline; filename="SageAlpha_CRH_Report.html"'
    )
    resp.headers["X-PDF-Generated"] = "no"
    return resp


@pdf_bp.route("/download-report-static")
def download_report_static():
    """Serve pre-generated static PDF if available."""
    static_pdf = os.path.join(
        current_app.static_folder or "static", "sagealpha_report.pdf"
    )
    if os.path.exists(static_pdf):
        return send_file(
            static_pdf, as_attachment=True, download_name="SageAlpha_CRH_Report.pdf"
        )
    return download_report()


@pdf_bp.route("/download-report-test")
def download_report_test():
    """Serve test PDF for client testing."""
    static_pdf = os.path.join(current_app.static_folder or "static", "test_report.pdf")
    if os.path.exists(static_pdf):
        return send_file(
            static_pdf, as_attachment=True, download_name="test_report.pdf"
        )
    return ("Test PDF not found. Place a small PDF at static/test_report.pdf.", 404)


@pdf_bp.route("/download-report-playwright")
def download_report_playwright():
    """Render report to PDF using Playwright Chromium."""
    try:
        html = render_template("sagealpha_reports.html")
        pdf_bytes = try_playwright_pdf(html)

        if pdf_bytes:
            resp = make_response(pdf_bytes)
            resp.headers["Content-Type"] = "application/pdf"
            resp.headers["Content-Disposition"] = (
                'attachment; filename="SageAlpha_CRH_Report.pdf"'
            )
            resp.headers["X-PDF-Generated"] = "yes (playwright)"
            return resp

        # Fallback to HTML
        resp = make_response(html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Content-Disposition"] = (
            'inline; filename="SageAlpha_CRH_Report.html"'
        )
        resp.headers["X-PDF-Generated"] = "no (fallback)"
        return resp
    except Exception as e:
        current_app.logger.exception(f"Playwright PDF route failed: {e}")
        html = render_template("sagealpha_reports.html")
        resp = make_response(html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Content-Disposition"] = (
            'inline; filename="SageAlpha_CRH_Report.html"'
        )
        resp.headers["X-PDF-Generated"] = "no (error fallback)"
        return resp

