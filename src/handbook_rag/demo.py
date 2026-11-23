"""The demo page: a question box over ``POST /ask``, served by the API itself.

Three files written by hand, in ``static/`` next to this module: the page, its script and
its style. There is no framework and no build step, so what is served is what is in the
repository, and a reader can read all of it. Why, and how the page keeps what a model
wrote from becoming code or a link, is in ADR 0016.
"""

from collections.abc import Awaitable, Callable
from importlib.resources import files

from fastapi import FastAPI, Response

# The files of the page, by the address each is served at: its name in ``static/`` and
# its content type. Each type names its charset, so that the browser never has to guess.
FILES = {
    "/": ("demo.html", "text/html; charset=utf-8"),
    "/static/demo.js": ("demo.js", "text/javascript; charset=utf-8"),
    "/static/demo.css": ("demo.css", "text/css; charset=utf-8"),
}

# What the browser may do on the page. Nothing is allowed unless it is named: the script,
# the style and the requests to the API, all from this server. So no inline script or
# style runs, nothing is loaded from another site, no other page can frame this one, and
# no form sends anywhere. The last two lines switch on Trusted Types, where a browser has
# them: it then refuses any HTML written from a string, which the script never does.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
        "require-trusted-types-for 'script'",
        "trusted-types 'none'",
    )
)

HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    # The browser uses each file as the content type says, and never as something else.
    "X-Content-Type-Options": "nosniff",
    # A source's page, opened from the page, is not told where the reader came from.
    "Referrer-Policy": "no-referrer",
    # The browser may keep a copy but checks it with the server before using it, so a
    # changed file reaches the reader at the next load.
    "Cache-Control": "no-cache",
}


def read_files() -> dict[str, tuple[bytes, str]]:
    """The page's files with their content types, by the address each is served at.

    They are read through the package's resources, not from a path, so they are found
    wherever the package is installed. In the container image that is a wheel in a
    virtual environment, far from this repository, and a file missing from the wheel
    fails here, when the application is built.
    """
    static = files("handbook_rag").joinpath("static")
    return {
        path: (static.joinpath(name).read_bytes(), media_type)
        for path, (name, media_type) in FILES.items()
    }


def add_demo_page(app: FastAPI) -> None:
    """Serve the page and its two files, outside the API's documentation.

    The files are read once, here. They are small, and a request then reads no file.
    """
    for path, (body, media_type) in read_files().items():
        app.add_api_route(path, _serve(body, media_type), methods=["GET"], include_in_schema=False)


def _serve(body: bytes, media_type: str) -> Callable[[], Awaitable[Response]]:
    async def serve() -> Response:
        return Response(content=body, media_type=media_type, headers=HEADERS)

    return serve
