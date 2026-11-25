"""入口：python server.py ... / MCP サーバー。本体は crystal/server.py。"""

import sys

from shido.server import main

if __name__ == "__main__":
    sys.exit(main())
