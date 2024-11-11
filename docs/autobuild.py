#!/usr/bin/env python

from livereload import Server
import os
import subprocess


def rebuild():
    print("Rebuilding documentation...")
    subprocess.run(["make", "html"], check=True)
    print("Done!")


server = Server()

# Watch RST files
server.watch("source/**/*.rst", rebuild)
# Watch Python files
server.watch("../quantammsim/**/*.py", rebuild)
# Watch static files
server.watch("source/_static/*", rebuild)
# Watch templates
server.watch("source/_templates/*", rebuild)

# Initial build
rebuild()

# Serve the documentation
server.serve(root="build/html", port=8000, host="localhost", open_url_delay=1)
