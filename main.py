from __future__ import annotations

import uvicorn

from config import API_SETTINGS


def run() -> None:
    uvicorn.run("server:app", host=API_SETTINGS.host, port=API_SETTINGS.port, reload=False)


if __name__ == "__main__":
    run()
